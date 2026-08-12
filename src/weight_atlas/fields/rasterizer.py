"""Stats table → 2D matrices per channel (rows=layer, cols=slot)."""

from __future__ import annotations

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
)
from weight_atlas.core.types import AtlasSpec, ExpertPanel, Field2D, TensorStats
from weight_atlas.fields.scaling import apply_scale
from weight_atlas.fields.tif_io import read_tif


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

    is_smooth = path.name.endswith("_smooth.tif")
    data = read_tif(path)
    if not is_smooth:
        ch_spec = spec.channels.get(channel, {})
        if "scale" in ch_spec:
            data = apply_scale(data, ch_spec["scale"])

    upsample = max(1, int(spec.grid.get("upsample", 1)))
    n_rows, _ = data.shape
    n_layers = n_rows // upsample if is_smooth else n_rows
    return Field2D(
        channel=channel,
        data=data,
        row_labels=[str(i) for i in range(n_layers)],
        col_labels=list(spec.slots),
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

        expert_id = extract_expert_id(ts.name)
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
            expert_id = extract_expert_id(ts.name)
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
