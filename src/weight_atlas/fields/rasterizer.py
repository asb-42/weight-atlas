"""Stats table → 2D matrices per channel (rows=layer, cols=slot)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from weight_atlas.core.name_map import (
    extract_expert_id,
    get_moe_slot,
    is_expert_tensor,
    is_shared_expert,
    map_name,
    map_vision,
)
from weight_atlas.core.types import AtlasSpec, ExpertPanel, Field2D, TensorStats
from weight_atlas.fields.scaling import apply_scale
from weight_atlas.fields.tif_io import read_tif


def _vision_row_labels(out_dir: Path, n_logical_rows: int) -> list[str]:
    """Reconstruct vision-sheet row labels from the fingerprint.

    Vision rows are the block indices 0..n_blocks-1 plus a final ``"global"``
    row when the scan had global vision tensors (patch_embed, pos_embed,
    projector). Without a fingerprint (or an old scan), fall back to plain
    numeric labels.
    """
    fp_path = out_dir / "fingerprint.json"
    fp: dict[str, Any] = {}
    if fp_path.exists():
        try:
            fp = json.loads(fp_path.read_text())
        except (OSError, ValueError):
            fp = {}
    vision = fp.get("model", {}).get("vision", {})
    n_blocks = vision.get("n_blocks") if isinstance(vision, dict) else None
    if n_blocks is None or n_blocks > n_logical_rows:
        return [str(i) for i in range(n_logical_rows)]
    labels = [str(i) for i in range(n_blocks)]
    if n_blocks < n_logical_rows:
        labels.append("global")
    return labels


def load_channel_field(
    out_dir: Path,
    channel: str,
    spec: AtlasSpec,
    *,
    prefer_smooth: bool = True,
    model_name: str = "",
) -> Field2D | None:
    """Build a Field2D for rendering from a scan output directory.

    Uses the smooth TIFF when present (already channel-scaled by the scan
    pipeline), otherwise the raw TIFF with the channel scale applied on the
    fly. Row labels are the true layer indices and column labels the spec
    slot names, so rendered sheets stay interpretable even after upsampling.

    ``vision_<ch>`` channels use the vision slot taxonomy and the vision
    block labels from the fingerprint (block indices + an optional
    ``"global"`` row).

    Returns None if neither a smooth nor a raw field exists.
    """
    smooth = out_dir / f"field_{channel}_smooth.tif"
    raw = out_dir / f"field_{channel}_raw.tif"
    if prefer_smooth and smooth.exists():
        path = smooth
    elif raw.exists():
        path = raw
    else:
        return None

    is_vision = channel.startswith("vision_")
    is_expert = channel.startswith("expert_")
    if is_vision:
        base_channel = channel[len("vision_"):]
    elif is_expert:
        # expert_<slot>_<channel>  (slot ∈ {mlp_gate, mlp_up, mlp_down};
        # channel names themselves contain no underscore, so the channel is
        # the LAST segment — split("_", 1) misparsed "mlp_gate_height" into
        # ("mlp", "gate_height") and the scale lookup silently missed).
        parts = channel[len("expert_"):].rsplit("_", 1)
        _panel_slot, base_channel = parts if len(parts) == 2 else (parts[0], "")
    else:
        base_channel = channel

    is_smooth = path.name.endswith("_smooth.tif")
    data = read_tif(path)
    if not is_smooth:
        ch_spec = spec.channels.get(channel, {})
        if not ch_spec and is_vision:
            ch_spec = spec.vision_channels.get(base_channel, {})
        if not ch_spec and is_expert:
            ch_spec = (spec.expert_channels or spec.channels).get(base_channel, {})
        if "scale" in ch_spec:
            data = apply_scale(data, ch_spec["scale"])

    upsample = max(1, int(spec.grid.get("upsample", 1)))
    n_rows, n_cols = data.shape
    n_logical_rows = n_rows // upsample if is_smooth else n_rows
    if is_vision:
        col_labels = list(spec.vision_slots)
        row_labels = _vision_row_labels(out_dir, n_logical_rows)
    elif is_expert:
        # Expert panels: rows are transformer layers, columns are expert ids.
        n_experts = n_cols // upsample if is_smooth else n_cols
        col_labels = [str(i) for i in range(n_experts)]
        row_labels = [str(i) for i in range(n_logical_rows)]
    else:
        col_labels = list(spec.slots)
        row_labels = [str(i) for i in range(n_logical_rows)]
    return Field2D(
        channel=channel,
        data=data,
        row_labels=row_labels,
        col_labels=col_labels,
        spec_version=spec.spec_version,
        model_name=model_name,
    )


def rasterize(
    stats: Iterable[TensorStats],
    spec: AtlasSpec,
    stat_key: str,
) -> Field2D:
    """Rasterize a single statistic into a (n_layers × n_slots) field.

    Missing combinations become ``NaN``; no implicit fill.
    MoE expert tensors are excluded from the main raster (they go into ExpertPanel).
    """
    slot_idx = {s: i for i, s in enumerate(spec.slots)}
    layers: list[int] = []
    seen_layers: set[int] = set()
    cells: dict[tuple[int, int], float] = {}

    for ts in stats:
        # Skip expert tensors and shared experts (they go into panels)
        if is_expert_tensor(ts.name) or is_shared_expert(ts.name):
            continue
        layer, slot = map_name(ts.name)
        if layer is None:
            continue  # skip non-layer tensors like embed/lm_head
        if layer not in seen_layers:
            layers.append(layer)
            seen_layers.add(layer)
        value = getattr(ts, stat_key, None)
        if value is None:
            continue
        col = slot_idx.get(slot)
        if col is None:
            continue
        cells[(layer, col)] = float(value)

    n_rows = len(layers)
    n_cols = len(spec.slots)
    grid = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    row_idx = {layer: i for i, layer in enumerate(layers)}
    for (layer, col), value in cells.items():
        grid[row_idx[layer], col] = value

    row_labels = [str(idx) for idx in layers]
    col_labels = list(spec.slots)
    return Field2D(
        channel=stat_key,
        data=grid,
        row_labels=row_labels,
        col_labels=col_labels,
        spec_version=spec.spec_version,
    )


def rasterize_flat(
    stats: Iterable[TensorStats],
    spec: AtlasSpec,
    stat_key: str,
) -> Field2D | None:
    """Rasterize non-layered tensors into a (1 × n_slots) flat field.

    For architectures like BDH where all tensors are shared across the model
    (no per-layer structure), this produces a single-row grid with one column
    per mapped slot.  Returns ``None`` if no non-layer tensors match any slot.
    """
    slot_idx = {s: i for i, s in enumerate(spec.slots)}
    cells: dict[int, float] = {}
    col_labels_set: set[str] = set()

    for ts in stats:
        layer, slot = map_name(ts.name)
        if layer is not None:
            continue  # only non-layer tensors
        col = slot_idx.get(slot)
        if col is None:
            continue
        value = getattr(ts, stat_key, None)
        if value is None:
            continue
        cells[col] = float(value)
        col_labels_set.add(slot)

    if not cells:
        return None

    # Build a compact grid with only the columns that have data
    cols_sorted = sorted(cells.keys())
    col_labels = [spec.slots[c] for c in cols_sorted]
    grid = np.full((1, len(cols_sorted)), np.nan, dtype=np.float64)
    for i, col in enumerate(cols_sorted):
        grid[0, i] = cells[col]

    return Field2D(
        channel=stat_key,
        data=grid,
        row_labels=["model"],
        col_labels=col_labels,
        spec_version=spec.spec_version,
    )


def rasterize_vision(
    stats: Iterable[TensorStats],
    spec: AtlasSpec,
    stat_key: str,
) -> Field2D | None:
    """Rasterize a single statistic into a (vision_block × vision_slot) field.

    Rows are the vision tower's block indices (e.g. ``v.blk.N`` / HF encoder
    layers), plus a final ``"global"`` row for non-block vision tensors
    (patch_embed, pos_embed, multimodal projector). Missing combinations are
    ``NaN``. Returns None when the model has no vision tensors for this stat.
    """
    if not spec.vision_slots:
        return None
    slot_idx = {s: i for i, s in enumerate(spec.vision_slots)}
    blocks: set[int] = set()
    cells: dict[tuple[int | None, int], float] = {}
    has_global = False

    for ts in stats:
        mapped = map_vision(ts.name)
        if mapped is None:
            continue
        block, slot = mapped
        col = slot_idx.get(slot)
        if col is None:
            continue  # slot not in the vision taxonomy
        value = getattr(ts, stat_key, None)
        if value is None or not np.isfinite(value):
            continue
        if block is None:
            has_global = True
        else:
            blocks.add(block)
        key = (block, col)
        if key in cells:
            # Multiple tensors share a cell (e.g. mm.model.mlp.0/1/2 in the
            # global projector row): aggregate by mean, deterministically.
            cells[key] = (cells[key] + float(value)) / 2.0
        else:
            cells[key] = float(value)

    if not cells:
        return None

    blocks_sorted = sorted(blocks)
    rows: list[int | None] = list(blocks_sorted)
    if has_global:
        rows.append(None)  # global row last
    grid = np.full((len(rows), len(spec.vision_slots)), np.nan, dtype=np.float64)
    row_idx = {block: i for i, block in enumerate(rows)}
    for (block, col), value in cells.items():
        grid[row_idx[block], col] = value

    row_labels = [str(b) for b in blocks_sorted]
    if has_global:
        row_labels.append("global")
    return Field2D(
        channel=stat_key,
        data=grid,
        row_labels=row_labels,
        col_labels=list(spec.vision_slots),
        spec_version=spec.spec_version,
    )


def detect_vision(stats: Iterable[TensorStats]) -> dict[str, Any] | None:
    """Summarize the vision subsystem of a model, if present.

    Returns a dict with ``present``, ``n_tensors``, ``n_blocks`` (distinct
    vision block indices) and ``n_global`` (non-block tensors such as
    patch_embed / pos_embed / projector), or None for text-only models.
    """
    blocks: set[int] = set()
    n_global = 0
    n_tensors = 0
    for ts in stats:
        mapped = map_vision(ts.name)
        if mapped is None:
            continue
        n_tensors += 1
        block, _slot = mapped
        if block is None:
            n_global += 1
        else:
            blocks.add(block)
    if n_tensors == 0:
        return None
    return {
        "present": True,
        "n_tensors": n_tensors,
        "n_blocks": len(blocks),
        "n_global": n_global,
    }


def rasterize_expert_panels(
    stats: Iterable[TensorStats],
    spec: AtlasSpec,
    stat_key: str,
) -> list[ExpertPanel]:
    """Rasterize MoE expert statistics into Layer × Expert panels.

    Returns a list of ExpertPanel objects, one per mlp slot (gate/up/down).
    Only includes tensors that are expert tensors (not shared experts).
    """
    # Collect expert stats by slot
    expert_stats: dict[str, dict[tuple[int, int], float]] = {
        "mlp_gate": {},
        "mlp_up": {},
        "mlp_down": {},
    }
    layers: set[int] = set()
    experts: set[int] = set()

    for ts in stats:
        if not is_expert_tensor(ts.name):
            continue
        slot = get_moe_slot(ts.name)
        if slot is None:
            continue
        # Map gate/up/down to mlp_gate/mlp_up/mlp_down
        slot_map = {"gate": "mlp_gate", "up": "mlp_up", "down": "mlp_down"}
        target_slot = slot_map[slot]

        expert_id = ts.expert_id if ts.expert_id is not None else extract_expert_id(ts.name)
        if expert_id is None:
            continue

        # Get layer from name
        layer, _ = map_name(ts.name)
        if layer is None:
            continue

        layers.add(layer)
        experts.add(expert_id)

        value = getattr(ts, stat_key, None)
        if value is None:
            continue
        expert_stats[target_slot][(layer, expert_id)] = float(value)

    if not layers or not experts:
        return []

    # Create panels
    layers_sorted = sorted(layers)
    experts_sorted = sorted(experts)
    layer_idx = {layer: i for i, layer in enumerate(layers_sorted)}
    expert_idx = {e: i for i, e in enumerate(experts_sorted)}

    panels: list[ExpertPanel] = []
    for slot in ["mlp_gate", "mlp_up", "mlp_down"]:
        grid = np.full((len(layers_sorted), len(experts_sorted)), np.nan, dtype=np.float64)
        for (layer, expert), value in expert_stats[slot].items():
            grid[layer_idx[layer], expert_idx[expert]] = value

        panels.append(ExpertPanel(
            slot=slot,
            channel=stat_key,
            data=grid,
            row_labels=[str(layer) for layer in layers_sorted],
            col_labels=[str(e) for e in experts_sorted],
            spec_version=spec.spec_version,
        ))

    return panels


def detect_moe(stats: Iterable[TensorStats]) -> dict[str, Any]:
    """Detect MoE configuration from tensor statistics.

    Returns:
        Dict with num_experts, shared_expert, num_layers if MoE detected,
        empty dict otherwise.
    """
    expert_ids: set[int] = set()
    has_shared_expert = False
    layers: set[int] = set()

    for ts in stats:
        if is_expert_tensor(ts.name):
            expert_id = ts.expert_id if ts.expert_id is not None else extract_expert_id(ts.name)
            if expert_id is not None:
                expert_ids.add(expert_id)
            layer, _ = map_name(ts.name)
            if layer is not None:
                layers.add(layer)
        if is_shared_expert(ts.name):
            has_shared_expert = True

    if not expert_ids:
        return {}

    return {
        "num_experts": max(expert_ids) + 1,
        "shared_expert": has_shared_expert,
        "num_layers": len(layers),
    }
