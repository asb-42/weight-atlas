"""Paired tensor-difference analysis (M9): quant-impact and edit-signature presets.

Compares two weight snapshots tensor-by-tensor (name-level pairing via
``map_name``), accumulating chunked float64 metrics so per-tensor memory stays
bounded regardless of tensor size. Two presets share the pairing, rasterisation
and determinism machinery:

- ``quant``: quantization impact — SQNR, rel-L2, cosine, zero-flip, max-delta,
  optional operator norm (``dspec``). Summary ``impact_summary.json``.
- ``edit``: edit signatures / abliteration — rel-L2, cosine, operator norm and
  the Δ-spectrum metrics ``delta_stable_rank``, ``spectral_share``, opt-in
  ``u1_coherence``, plus a classification heuristic, edit bands and a
  weight-space hotspot ranking. Summary ``compare_summary.json`` (with
  ``edit_signature`` + ``noise_floor`` blocks).

Determinism contract: within a tensor the chunk loop runs sequentially and
accumulators are float64 in a fixed order; tensors may be processed in any
order (parallel jobs) but each tensor's accumulation is order-independent
float64, so results are byte-identical for any ``jobs``. The Δ-spectrum runs
on the shared ``stats.spectrum`` machinery (same lock, same seeded rSVD), so
spectral values are deterministic too.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from weight_atlas.core.name_map import (
    extract_expert_id,
    get_moe_slot,
    is_expert_tensor,
    map_name,
    map_vision,
)
from weight_atlas.core.registry import get_loader
from weight_atlas.core.types import AtlasSpec, TensorHandle, detect_loader
from weight_atlas.fields.smoothing import smooth, upsample
from weight_atlas.fields.tif_io import write_tif

# Default chunk of the flattened tensor processed per accumulator pass. 1M
# float32 elements (4 MB per side) keeps float64 accumulator temporaries small.
_DEFAULT_CHUNK = 1_048_576

# Preset metric sets. ``dspec`` is opt-in for quant (operator_impact); the edit
# preset always computes the Δ-spectrum because classification needs it.
_QUANT_METRICS = ("sqnr_db", "rel_l2", "cos", "zflip", "dmax", "dspec")
_EDIT_METRICS = ("rel_l2", "cos", "dspec", "delta_stable_rank", "spectral_share")


def _resolve_jobs(jobs: int | None) -> int:
    if jobs is not None and jobs > 0:
        return jobs
    import os
    return max(1, min(8, os.cpu_count() or 1))


def _pair_key(name: str, expert_id: int | None) -> tuple[Any, ...]:
    """Deterministic join key for a tensor name.

    Expert tensors key on ``("expert", layer, moe_slot, expert_id)`` so GGUF
    sub-handles (``blk.N.ffn_gate_exps.weight[3]``) and HF experts
    (``mlp.experts.3.gate_proj.weight``) pair up across formats. Vision
    tensors key on their vision block/slot; everything else on
    ``(layer, slot)``. ``layer`` may be None for non-layer tensors (embed,
    lm_head) which still pair by slot.

    ``None`` layer/expert ids are encoded as -1 so the tuple is fully
    sortable (join keys are ``sorted()`` in ``pair_tensors``); no real layer
    or expert id is negative.
    """
    eid = expert_id if expert_id is not None else extract_expert_id(name)
    if is_expert_tensor(name):
        layer, _ = map_name(name)
        moe = get_moe_slot(name)
        return ("expert", layer if layer is not None else -1, moe, eid if eid is not None else -1)
    vision = map_vision(name)
    if vision is not None:
        block, slot = vision
        return ("vision", block if block is not None else -1, slot)
    layer, slot = map_name(name)
    return ("layer", layer if layer is not None else -1, slot)


@dataclass
class TensorImpact:
    """Per-pair difference metrics for one tensor."""

    name_a: str
    name_b: str
    layer: int | None
    slot: str
    expert_id: int | None
    qtype_a: str
    qtype_b: str
    n_elements: int
    sqnr_db: float
    rel_l2: float
    cos: float
    zflip: float
    dmax: float
    dspec: float | None = None
    # Edit-preset Δ-spectrum metrics (None when not computed).
    delta_stable_rank: float | None = None
    spectral_share: float | None = None
    u1: np.ndarray | None = None


@dataclass
class ImpactSummary:
    """Aggregate impact statistics (impact_summary.json body)."""

    ref_side: str
    model_a: dict[str, Any]
    model_b: dict[str, Any]
    alignment: dict[str, Any]
    global_: dict[str, Any]
    per_type: dict[str, dict[str, float]]
    hotspot_ranking: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Accum:
    sum_a2: float = 0.0
    sum_b2: float = 0.0
    sum_ab: float = 0.0
    sum_d2: float = 0.0
    dmax: float = 0.0
    zflip: int = 0
    n_finite: int = 0

    def absorb(self, chunk: tuple[float, float, float, float, float, int, int]) -> None:
        sa2, sb2, sab, sd2, dmax, zflip, n = chunk
        self.sum_a2 += sa2
        self.sum_b2 += sb2
        self.sum_ab += sab
        self.sum_d2 += sd2
        self.dmax = max(self.dmax, dmax)
        self.zflip += zflip
        self.n_finite += n


def _chunk_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float, float, int, int]:
    """Accumulate metrics over one chunk of paired float32 payloads.

    Returns ``(sum_a2, sum_b2, sum_ab, sum_d2, dmax, zflip, n_finite)``. Only
    positions where both sides are finite contribute; NaN is never a value.
    """
    fin = np.isfinite(a) & np.isfinite(b)
    af = a[fin].astype(np.float64)
    bf = b[fin].astype(np.float64)
    if af.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0
    d = bf - af
    sum_d2 = float(np.dot(d, d))
    dmax = float(np.abs(d).max()) if sum_d2 > 0 else 0.0
    zflip = int(np.logical_xor(af == 0.0, bf == 0.0).sum())
    return (
        float(np.dot(af, af)),
        float(np.dot(bf, bf)),
        float(np.dot(af, bf)),
        sum_d2,
        dmax,
        zflip,
        int(af.size),
    )


def _delta_spectrum_metrics(
    a: np.ndarray,
    b: np.ndarray,
    spec: AtlasSpec,
    *,
    want_u1: bool,
) -> tuple[float | None, float | None, float | None, np.ndarray | None]:
    """Compute Δ-spectrum metrics (dspec, delta_stable_rank, spectral_share, u1).

    Materializes ``Δ = b - a`` and runs the shared truncated spectrum
    (exact SVD for ``min(m, n) <= 512``, else seeded Halko rSVD k=16 q=2 in
    float32) on its row-flattened matrix. ``delta_stable_rank`` uses the
    float64-exact ‖Δ‖_F² from the chunk accumulators; ``spectral_share`` uses
    the spectrum's squared-sum (truncated for large matrices — the same
    documented bias as ``effective_rank``). u1 is sign-fixed so its
    largest-|component| entry is positive.
    """
    from weight_atlas.stats.spectrum import spectrum_of_matrix, top_left_singular_vector

    delta = b - a
    if delta.ndim == 1:
        delta = delta.reshape(1, -1)
    seed = int(spec.seeds.get("svd", 0))
    s = spectrum_of_matrix(delta, seed=seed)
    s1 = float(s[0])
    sumsq = float(np.dot(s, s))
    dspec: float | None = s1 if np.isfinite(s1) else None
    delta_stable_rank: float | None = None
    spectral_share: float | None = None
    if dspec is not None and dspec > 0:
        delta_stable_rank = float(np.dot(delta.reshape(-1), delta.reshape(-1)) / (dspec * dspec))
    if sumsq > 0:
        spectral_share = float((s1 * s1) / sumsq)
    u1: np.ndarray | None = None
    if want_u1 and delta.ndim >= 2:
        u1 = top_left_singular_vector(delta, seed=seed)
    return dspec, delta_stable_rank, spectral_share, u1


def _pair_metrics(
    handle_a: TensorHandle,
    handle_b: TensorHandle,
    spec: AtlasSpec,
    *,
    ref_side: str,
    compute_spectrum: bool,
    want_u1: bool,
    chunk_size: int,
) -> TensorImpact:
    """Compute chunked float64 metrics for one tensor pair."""
    shape_a = handle_a.shape
    shape_b = handle_b.shape
    if shape_a != shape_b:
        raise ValueError(
            f"shape mismatch for paired tensors: {handle_a.name} "
            f"(A)={shape_a} vs {handle_b.name} (B)={shape_b}"
        )

    a = handle_a.load()
    b = handle_b.load()
    n_elements = int(np.prod(shape_a))

    acc = _Accum()
    a_flat = np.ascontiguousarray(a).reshape(-1)
    b_flat = np.ascontiguousarray(b).reshape(-1)
    for start in range(0, n_elements, chunk_size):
        end = min(start + chunk_size, n_elements)
        acc.absorb(_chunk_metrics(a_flat[start:end], b_flat[start:end]))

    # Reference side picks the signal energy in the SQNR/rel-L2 ratio. The
    # difference energy is symmetric; only the numerator changes.
    ref_energy = acc.sum_a2 if ref_side == "a" else acc.sum_b2
    if acc.sum_d2 > 0 and ref_energy > 0:
        sqnr_db = 10.0 * math.log10(ref_energy / acc.sum_d2)
        rel_l2 = math.sqrt(acc.sum_d2 / ref_energy)
    elif acc.sum_d2 == 0:
        sqnr_db = math.inf  # identical tensors
        rel_l2 = 0.0
    else:
        sqnr_db = math.nan  # reference is silent (all-zero) but delta is not
        rel_l2 = math.inf

    denom = math.sqrt(acc.sum_a2 * acc.sum_b2)
    cos = float(np.clip(acc.sum_ab / denom, -1.0, 1.0)) if denom > 0 else 0.0
    zflip = (acc.zflip / acc.n_finite) if acc.n_finite > 0 else math.nan
    dmax = acc.dmax

    dspec: float | None = None
    delta_stable_rank: float | None = None
    spectral_share: float | None = None
    u1: np.ndarray | None = None
    if (compute_spectrum or want_u1) and n_elements > 0:
        dspec, delta_stable_rank, spectral_share, u1 = _delta_spectrum_metrics(
            a, b, spec, want_u1=want_u1
        )

    layer, slot = map_name(handle_a.name)
    if slot in ("other",):
        slot = map_name(handle_b.name)[1]
    return TensorImpact(
        name_a=handle_a.name,
        name_b=handle_b.name,
        layer=layer,
        slot=slot,
        expert_id=handle_a.expert_id,
        qtype_a=handle_a.dtype,
        qtype_b=handle_b.dtype,
        n_elements=n_elements,
        sqnr_db=sqnr_db,
        rel_l2=rel_l2,
        cos=cos,
        zflip=zflip,
        dmax=dmax,
        dspec=dspec,
        delta_stable_rank=delta_stable_rank,
        spectral_share=spectral_share,
        u1=u1,
    )


def pair_tensors(
    handles_a: list[TensorHandle],
    handles_b: list[TensorHandle],
) -> tuple[list[tuple[TensorHandle, TensorHandle]], list[dict[str, Any]]]:
    """Pair tensors across two model snapshots by name-level join key.

    Returns ``(pairs, skipped)`` where ``skipped`` is a list of
    ``{"name", "side", "reason"}`` for tensors present on only one side.
    Raises ValueError if a join key maps to multiple tensors on one side
    (ambiguous pairing).
    """
    index_a: dict[tuple[Any, ...], TensorHandle] = {}
    index_b: dict[tuple[Any, ...], TensorHandle] = {}
    for h in handles_a:
        key = _pair_key(h.name, h.expert_id)
        if key in index_a:
            raise ValueError(
                f"ambiguous pairing key {key} on side A: "
                f"{index_a[key].name} and {h.name}"
            )
        index_a[key] = h
    for h in handles_b:
        key = _pair_key(h.name, h.expert_id)
        if key in index_b:
            raise ValueError(
                f"ambiguous pairing key {key} on side B: "
                f"{index_b[key].name} and {h.name}"
            )
        index_b[key] = h

    pairs: list[tuple[TensorHandle, TensorHandle]] = []
    skipped: list[dict[str, Any]] = []
    for key in sorted(index_a):
        if key in index_b:
            pairs.append((index_a[key], index_b[key]))
        else:
            skipped.append({"name": index_a[key].name, "side": "a", "reason": "not in B"})
    for key in sorted(index_b):
        if key not in index_a:
            skipped.append({"name": index_b[key].name, "side": "b", "reason": "not in A"})

    pairs.sort(key=lambda p: _pair_key(p[0].name, p[0].expert_id))
    return pairs, skipped


def _raster_main(
    impacts: list[TensorImpact],
    spec: AtlasSpec,
    metric: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Rasterize a metric into a (layer × slot) grid for dense tensors."""
    slot_idx = {s: i for i, s in enumerate(spec.slots)}
    layers: list[int] = []
    seen: set[int] = set()
    cells: dict[tuple[int, int], float] = {}
    for ti in impacts:
        if ti.layer is None or ti.slot in ("expert", "other"):
            continue
        if is_expert_tensor(ti.name_a):
            continue
        col = slot_idx.get(ti.slot)
        if col is None:
            continue
        if ti.layer not in seen:
            layers.append(ti.layer)
            seen.add(ti.layer)
        val = getattr(ti, metric)
        if val is not None and np.isfinite(val):
            cells[(ti.layer, col)] = float(val)
    n_rows = len(layers)
    n_cols = len(spec.slots)
    grid = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    row_idx = {layer: i for i, layer in enumerate(layers)}
    for (layer, col), value in cells.items():
        grid[row_idx[layer], col] = value
    return grid, [str(lyr) for lyr in layers], list(spec.slots)


def _raster_expert(
    impacts: list[TensorImpact],
    spec: AtlasSpec,
    metric: str,
) -> list[dict[str, Any]]:
    """Rasterize a metric into Layer × Expert panels per moe slot."""
    panels: dict[str, dict[tuple[int, int], float]] = {}
    layers: set[int] = set()
    experts: set[int] = set()
    for ti in impacts:
        if ti.layer is None or not is_expert_tensor(ti.name_a):
            continue
        moe = get_moe_slot(ti.name_a)
        if moe is None or ti.expert_id is None:
            continue
        panels.setdefault(moe, {})
        val = getattr(ti, metric)
        if val is None or not np.isfinite(val):
            continue
        panels[moe][(ti.layer, ti.expert_id)] = float(val)
        layers.add(ti.layer)
        experts.add(ti.expert_id)

    slot_map = {"gate": "mlp_gate", "up": "mlp_up", "down": "mlp_down"}
    result: list[dict[str, Any]] = []
    for moe, cells in panels.items():
        layers_sorted = sorted(layers)
        experts_sorted = sorted(experts)
        layer_idx = {lyr: i for i, lyr in enumerate(layers_sorted)}
        expert_idx = {e: i for i, e in enumerate(experts_sorted)}
        grid = np.full((len(layers_sorted), len(experts_sorted)), np.nan, dtype=np.float64)
        for (layer, expert), value in cells.items():
            grid[layer_idx[layer], expert_idx[expert]] = value
        result.append(
            {
                "slot": slot_map.get(moe, moe),
                "data": grid,
                "row_labels": [str(lyr) for lyr in layers_sorted],
                "col_labels": [str(e) for e in experts_sorted],
            }
        )
    return result


def _raster_vision(
    impacts: list[TensorImpact],
    spec: AtlasSpec,
    metric: str,
) -> np.ndarray | None:
    """Rasterize a metric into a (vision_block × vision_slot) grid."""
    if not spec.vision_slots:
        return None
    slot_idx = {s: i for i, s in enumerate(spec.vision_slots)}
    blocks: set[int] = set()
    cells: dict[tuple[int | None, int], float] = {}
    has_global = False
    for ti in impacts:
        mapped = map_vision(ti.name_a)
        if mapped is None:
            continue
        block, slot = mapped
        col = slot_idx.get(slot)
        if col is None:
            continue
        val = getattr(ti, metric)
        if val is None or not np.isfinite(val):
            continue
        if block is None:
            has_global = True
        else:
            blocks.add(block)
        cells[(block, col)] = float(val)
    if not cells:
        return None
    rows: list[int | None] = [int(b) for b in sorted(blocks)]
    if has_global:
        rows.append(None)
    grid = np.full((len(rows), len(spec.vision_slots)), np.nan, dtype=np.float64)
    row_idx = {b: i for i, b in enumerate(rows)}
    for (block, col), value in cells.items():
        grid[row_idx[block], col] = value
    return grid


def _qtype_code(impacts: list[TensorImpact], ref_side: str) -> dict[str, int]:
    """Deterministic integer codes for the non-reference quantization types."""
    types = sorted({ti.qtype_b if ref_side == "a" else ti.qtype_a for ti in impacts})
    return {t: i for i, t in enumerate(types)}


def _raster_qtype(
    impacts: list[TensorImpact],
    spec: AtlasSpec,
    codes: dict[str, int],
    ref_side: str,
) -> np.ndarray:
    """Rasterize quantization-type codes into a (layer × slot) grid."""
    slot_idx = {s: i for i, s in enumerate(spec.slots)}
    layers: list[int] = []
    seen: set[int] = set()
    cells: dict[tuple[int, int], int] = {}
    for ti in impacts:
        if ti.layer is None or ti.slot in ("expert", "other"):
            continue
        if is_expert_tensor(ti.name_a):
            continue
        col = slot_idx.get(ti.slot)
        if col is None:
            continue
        if ti.layer not in seen:
            layers.append(ti.layer)
            seen.add(ti.layer)
        qtype = ti.qtype_b if ref_side == "a" else ti.qtype_a
        cells[(ti.layer, col)] = codes[qtype]
    n_rows = len(layers)
    n_cols = len(spec.slots)
    grid = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    row_idx = {layer: i for i, layer in enumerate(layers)}
    for (layer, col), value in cells.items():
        grid[row_idx[layer], col] = float(value)
    return grid


def _smoothed(raw: np.ndarray, spec: AtlasSpec) -> np.ndarray:
    """Upsample + smooth a raw impact field for display artefacts."""
    up = upsample(raw, int(spec.grid.get("upsample", 1)))
    return smooth(up, float(spec.grid.get("smooth_sigma", 1.0)))


def _global_stats(impacts: list[TensorImpact]) -> dict[str, float]:
    """Median / p05 SQNR and median rel-L2 across dense tensor pairs."""
    sqnr = np.array([t.sqnr_db for t in impacts if np.isfinite(t.sqnr_db)], dtype=np.float64)
    rel = np.array([t.rel_l2 for t in impacts if np.isfinite(t.rel_l2)], dtype=np.float64)
    return {
        "median_sqnr_db": float(np.median(sqnr)) if sqnr.size else math.nan,
        "p05_sqnr_db": float(np.quantile(sqnr, 0.05)) if sqnr.size else math.nan,
        "median_rel_l2": float(np.median(rel)) if rel.size else math.nan,
    }


def _per_type_stats(impacts: list[TensorImpact], ref_side: str) -> dict[str, dict[str, float]]:
    per: dict[str, list[float]] = {}
    for ti in impacts:
        qtype = ti.qtype_b if ref_side == "a" else ti.qtype_a
        if np.isfinite(ti.sqnr_db):
            per.setdefault(qtype, []).append(ti.sqnr_db)
    return {
        t: {"n": float(len(v)), "median_sqnr_db": float(np.median(v))}
        for t, v in sorted(per.items())
    }


def _hotspot_ranking(impacts: list[TensorImpact], top_k: int = 5) -> list[dict[str, Any]]:
    """Top-k tensor pairs ranked by lowest SQNR (worst quantization impact)."""
    ranked = sorted(
        (t for t in impacts if np.isfinite(t.sqnr_db)),
        key=lambda t: t.sqnr_db,
    )[:top_k]
    return [
        {
            "layer": t.layer,
            "slot": t.slot,
            "name_a": t.name_a,
            "name_b": t.name_b,
            "sqnr_db": t.sqnr_db,
            "rel_l2": t.rel_l2,
        }
        for t in ranked
    ]


def _hotspot_ranking_rel_l2(impacts: list[TensorImpact], top_k: int = 5) -> list[dict[str, Any]]:
    """Top-k tensor pairs ranked by weight-space rel-L2 (largest edit)."""
    ranked = sorted(
        (t for t in impacts if np.isfinite(t.rel_l2)),
        key=lambda t: t.rel_l2,
        reverse=True,
    )[:top_k]
    return [
        {
            "layer": t.layer,
            "slot": t.slot,
            "name_a": t.name_a,
            "name_b": t.name_b,
            "rel_l2": t.rel_l2,
            "delta_stable_rank": t.delta_stable_rank,
        }
        for t in ranked
    ]


def _edit_bands(
    impacts: list[TensorImpact],
    spec: AtlasSpec,
) -> tuple[list[dict[str, Any]], float]:
    """Contiguous layer runs whose per-layer median rel-L2 exceeds the floor.

    Threshold = ``max(band_floor, band_threshold_factor * all-layer median)``
    — the multiplicative factor scales with the background edit level, so a
    uniform full-rank edit produces no bands while a localized edit stands
    out. Each band records its concentrated slots (per-slot median within the
    band above the band's own median). Returns ``(bands, band_mass_share)``.
    """
    edit = spec.edit or {}
    band_floor = float(edit.get("band_floor", 1e-4))
    factor = float(edit.get("band_threshold_factor", 3.0))

    per_layer: dict[int, list[float]] = {}
    all_layers: set[int] = set()
    for ti in impacts:
        if ti.layer is None or ti.slot in ("expert", "other") or is_expert_tensor(ti.name_a):
            continue
        all_layers.add(ti.layer)
        # Only edited tensors shape the layer median — unedited slots would
        # otherwise dilute a slot-concentrated edit (e.g. one mlp_down) to 0.
        if np.isfinite(ti.rel_l2) and ti.rel_l2 > band_floor:
            per_layer.setdefault(ti.layer, []).append(ti.rel_l2)
    # Layers with no edited tensors count as 0 so the all-layer median stays
    # anchored by the untouched majority (a localized edit then stands out).
    layer_median = {lv: float(np.median(per_layer[lv])) if lv in per_layer else 0.0 for lv in all_layers}
    if not layer_median:
        return [], 0.0
    all_median = float(np.median(list(layer_median.values())))
    threshold = max(band_floor, factor * all_median)

    runs: list[list[int]] = []
    run: list[int] = []
    for layer in sorted(layer_median):
        if layer_median[layer] > threshold:
            run.append(layer)
        else:
            if run:
                runs.append(run)
                run = []
    if run:
        runs.append(run)

    bands: list[dict[str, Any]] = []
    for layers_in_band in runs:
        # Band median over every slot's within-band median (unedited slots
        # contribute 0), so the edited slot(s) stand out as concentrated.
        slot_med: dict[str, list[float]] = {}
        for ti in impacts:
            if ti.layer not in set(layers_in_band) or ti.slot in ("expert", "other"):
                continue
            if is_expert_tensor(ti.name_a):
                continue
            if np.isfinite(ti.rel_l2):
                slot_med.setdefault(ti.slot, []).append(ti.rel_l2)
        slot_medians = {s: float(np.median(v)) for s, v in slot_med.items()}
        band_median = float(np.median(list(slot_medians.values()))) if slot_medians else 0.0
        concentrated = sorted(
            s for s, m in slot_medians.items() if m > band_median
        )
        bands.append(
            {
                "start_layer": layers_in_band[0],
                "end_layer": layers_in_band[-1],
                "n_layers": len(layers_in_band),
                "slots": concentrated,
            }
        )

    total_mass = sum(layer_median.values())
    band_mass = sum(layer_median[bl] for b in runs for bl in b)
    band_mass_share = band_mass / total_mass if total_mass > 0 else 0.0
    return bands, band_mass_share


def _classify_edit(
    impacts: list[TensorImpact],
    spec: AtlasSpec,
    bands: list[dict[str, Any]],
    band_mass_share: float,
) -> str:
    """Classify the edit into one of five signatures (first-match-wins).

    Decision tree (documented in ARCHITECTURE.md):
    1. no tensor is edited (every rel-L2 <= ``band_floor``) → ``identical``.
    2. median ``delta_stable_rank`` over edited tensors <= ``rank_low``:
       ``band_mass_share >= band_mass_share`` → ``low_rank_localized``,
       else → ``low_rank_diffuse``.
    3. Full-rank: no bands → ``full_rank_uniform`` (quantization/rounding-like),
       else → ``diffuse`` (full finetune with layer localization).
    """
    edit = spec.edit or {}
    rank_low = float(edit.get("rank_low", 2.0))
    band_floor = float(edit.get("band_floor", 1e-4))
    band_share = float(edit.get("band_mass_share", 0.7))

    edited = np.array(
        [
            t.delta_stable_rank
            for t in impacts
            if t.delta_stable_rank is not None
            and np.isfinite(t.delta_stable_rank)
            and np.isfinite(t.rel_l2)
            and t.rel_l2 > band_floor
        ],
        dtype=np.float64,
    )
    if edited.size == 0:
        return "identical"
    median_rank = float(np.median(edited))
    if np.isfinite(median_rank) and median_rank <= rank_low:
        if band_mass_share >= band_share:
            return "low_rank_localized"
        return "low_rank_diffuse"
    if not bands:
        return "full_rank_uniform"
    return "diffuse"


def _u1_coherence(
    impacts: list[TensorImpact],
    spec: AtlasSpec,
) -> float | None:
    """Mean pairwise cosine of the top left singular vector u1 across edited
    tensors sharing an output dimension (rows of the weight matrix).

    Opt-in (``edit.u1_coherence``). Only tensors above ``band_floor`` count;
    u1 is already sign-fixed. A shared abliteration direction d̂ (Δ = d̂(Wd̂)ᵀ)
    makes every edited u1 point the same way → coherence near 1.
    """
    edit = spec.edit or {}
    band_floor = float(edit.get("band_floor", 1e-4))
    groups: dict[int, list[np.ndarray]] = {}
    for ti in impacts:
        if ti.u1 is None or not (np.isfinite(ti.rel_l2) and ti.rel_l2 > band_floor):
            continue
        groups.setdefault(int(ti.u1.shape[0]), []).append(ti.u1)
    cosines: list[float] = []
    for vecs in groups.values():
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                cosines.append(float(np.dot(vecs[i], vecs[j])))
    if not cosines:
        return None
    return float(np.mean(cosines))


def _noise_floor_policy(
    impacts: list[TensorImpact],
    fp_a: dict[str, Any] | None,
    fp_b: dict[str, Any] | None,
) -> dict[str, Any]:
    """Weight-space noise-floor policy from the two fingerprints.

    ``identical`` — same loader and matching per-tensor dtype for every paired
    tensor: the deltas sit far above any quantization floor, so sub-percent
    rel-L2 values are trustworthy and no veil is applied.

    ``mismatched`` — loader or per-tensor dtype differ: edit-scale deltas can
    sit at or below the quantization noise of the noisier side. Emits the
    warning "edit-scale signal below quantization noise".
    """
    loader_a = (fp_a or {}).get("loader", "unknown")
    loader_b = (fp_b or {}).get("loader", "unknown")
    if loader_a != loader_b:
        return {
            "policy": "mismatched",
            "warning": "edit-scale signal below quantization noise: "
                       "loader differs between sides (A="
                       f"{loader_a}, B={loader_b}).",
        }
    tensors_a = (fp_a or {}).get("tensors", {})
    tensors_b = (fp_b or {}).get("tensors", {})
    for ti in impacts:
        da = tensors_a.get(ti.name_a, {}).get("dtype")
        db = tensors_b.get(ti.name_b, {}).get("dtype")
        if da is not None and db is not None and da == db:
            continue
        return {
            "policy": "mismatched",
            "warning": "edit-scale signal below quantization noise: "
                       "per-tensor dtype differs between sides "
                       f"({ti.name_a}={da} vs {ti.name_b}={db}).",
        }
    return {"policy": "identical", "warning": None}


def _build_summary(
    impacts: list[TensorImpact],
    skipped: list[dict[str, Any]],
    fp_a: dict[str, Any] | None,
    fp_b: dict[str, Any] | None,
    ref_side: str,
    warnings: list[str],
) -> ImpactSummary:
    model_a = {
        "loader": (fp_a or {}).get("loader", "unknown"),
        "n_tensors": (fp_a or {}).get("model", {}).get("n_tensors", 0),
        "quantization": (fp_a or {}).get("quantization", {}),
    }
    model_b = {
        "loader": (fp_b or {}).get("loader", "unknown"),
        "n_tensors": (fp_b or {}).get("model", {}).get("n_tensors", 0),
        "quantization": (fp_b or {}).get("quantization", {}),
    }
    alignment = {
        "mode": "strict",
        "n_pairs": len(impacts),
        "n_skipped": len(skipped),
        "skipped": skipped,
    }
    return ImpactSummary(
        ref_side=ref_side,
        model_a=model_a,
        model_b=model_b,
        alignment=alignment,
        global_=_global_stats(impacts),
        per_type=_per_type_stats(impacts, ref_side),
        hotspot_ranking=_hotspot_ranking(impacts),
        warnings=warnings,
    )


def _write_artefacts(
    impacts: list[TensorImpact],
    spec: AtlasSpec,
    out: Path,
    ref_side: str,
    *,
    preset: str,
    operator_impact: bool,
) -> list[Path]:
    """Write paired TIFF artefacts + summary JSON."""
    out.mkdir(parents=True, exist_ok=True)
    artefacts: list[Path] = []

    if preset == "edit":
        block = spec.edit or {}
        metrics = list(block.get("metrics", _EDIT_METRICS))
        prefix = "edit"
    else:
        block = spec.qimpact or {}
        metrics = list(block.get("metrics", _QUANT_METRICS))
        prefix = "impact"

    for metric in metrics:
        grid, row_labels, col_labels = _raster_main(impacts, spec, metric)
        raw_path = out / f"field_{prefix}_{metric}_raw.tif"
        write_tif(raw_path, grid)
        artefacts.append(raw_path)

        smooth_path = out / f"field_{prefix}_{metric}_smooth.tif"
        write_tif(smooth_path, _smoothed(grid, spec))
        artefacts.append(smooth_path)

        for panel in _raster_expert(impacts, spec, metric):
            slot = panel["slot"]
            raw_path = out / f"field_expert_{prefix}_{slot}_{metric}_raw.tif"
            write_tif(raw_path, panel["data"])
            artefacts.append(raw_path)
            smooth_path = out / f"field_expert_{prefix}_{slot}_{metric}_smooth.tif"
            write_tif(smooth_path, _smoothed(panel["data"], spec))
            artefacts.append(smooth_path)

        vision = _raster_vision(impacts, spec, metric)
        if vision is not None:
            raw_path = out / f"field_vision_{prefix}_{metric}_raw.tif"
            write_tif(raw_path, vision)
            artefacts.append(raw_path)
            smooth_path = out / f"field_vision_{prefix}_{metric}_smooth.tif"
            write_tif(smooth_path, _smoothed(vision, spec))
            artefacts.append(smooth_path)

    if preset == "quant":
        # Quantization-type map (non-reference side).
        codes = _qtype_code(impacts, ref_side)
        if codes:
            qtype_grid = _raster_qtype(impacts, spec, codes, ref_side)
            raw_path = out / "field_qtype_raw.tif"
            write_tif(raw_path, qtype_grid)
            artefacts.append(raw_path)
            qtype_meta_path = out / "qtype_map.json"
            with open(qtype_meta_path, "w") as f:
                json.dump(codes, f, indent=2, sort_keys=True)
                f.write("\n")
            artefacts.append(qtype_meta_path)

    return artefacts


def _build_edit_signature(
    impacts: list[TensorImpact],
    spec: AtlasSpec,
) -> dict[str, Any]:
    """Edit-signature summary block (classification + bands + stats)."""
    edit = spec.edit or {}
    band_floor = float(edit.get("band_floor", 1e-4))
    bands, band_mass_share = _edit_bands(impacts, spec)
    classification = _classify_edit(impacts, spec, bands, band_mass_share)

    rel = np.array([t.rel_l2 for t in impacts if np.isfinite(t.rel_l2)], dtype=np.float64)
    edited_ranks = np.array(
        [
            t.delta_stable_rank
            for t in impacts
            if t.delta_stable_rank is not None
            and np.isfinite(t.delta_stable_rank)
            and np.isfinite(t.rel_l2)
            and t.rel_l2 > band_floor
        ],
        dtype=np.float64,
    )
    stats: dict[str, Any] = {
        "median_rel_l2": float(np.median(rel)) if rel.size else math.nan,
        "median_delta_stable_rank": float(np.median(edited_ranks)) if edited_ranks.size else math.nan,
        "band_mass_share": band_mass_share,
    }
    if bool(edit.get("u1_coherence", False)):
        stats["u1_coherence"] = _u1_coherence(impacts, spec)

    return {
        "classification": classification,
        "stats": stats,
        "bands": bands,
        "hotspot_ranking_rel_l2": _hotspot_ranking_rel_l2(impacts),
    }


def run_paired(
    model_a: Path,
    model_b: Path,
    out: Path,
    spec: AtlasSpec,
    *,
    fp_a: dict[str, Any] | None = None,
    fp_b: dict[str, Any] | None = None,
    ref_side: str = "a",
    jobs: int | None = None,
    mode: str = "strict",
    preset: str = "quant",
    progress: Callable[[float, str], None] | None = None,
) -> list[Path]:
    """Run the full paired tensor-difference pipeline.

    Opens both weight snapshots, pairs tensors at the name level, computes
    chunked float64 metrics, and writes TIFFs + summary for the selected
    preset (``quant`` or ``edit``). Deterministic for any ``jobs``.

    Paired analysis is strict-only: ``mode`` must be ``"strict"`` (the metric
    pairing requires identical tensor shapes and layer indices); anything else
    raises ValueError.
    """
    if preset not in ("quant", "edit"):
        raise ValueError(f"unknown preset {preset!r}; expected 'quant' or 'edit'")
    if mode != "strict":
        raise ValueError(
            f"paired analysis is strict-only (mode={mode!r}); "
            "tensor-level pairing requires identical tensor shapes and layer "
            "indices on both sides. Use the compare subcommand for aligned "
            "(cross-architecture) comparison."
        )
    def _report(pct: float, msg: str) -> None:
        if progress is not None:
            progress(float(pct), msg)

    out.mkdir(parents=True, exist_ok=True)

    _report(0.0, "Opening models...")
    loader_a_id = detect_loader(model_a)
    loader_b_id = detect_loader(model_b)
    handles_a = list(get_loader(loader_a_id)().open(model_a))
    handles_b = list(get_loader(loader_b_id)().open(model_b))

    _report(0.05, "Pairing tensors...")
    pairs, skipped = pair_tensors(handles_a, handles_b)

    block = spec.edit if preset == "edit" else spec.qimpact
    block = block or {}
    compute_spectrum = preset == "edit" or bool(block.get("operator_impact", False))
    want_u1 = preset == "edit" and bool(block.get("u1_coherence", False))
    chunk_size = int(block.get("chunk_size", _DEFAULT_CHUNK))
    ref_side = "a" if ref_side != "b" else "b"

    impacts: list[TensorImpact] = []
    n_total = len(pairs)
    report_every = max(1, n_total // 40) if n_total else 1
    jobs_n = _resolve_jobs(jobs)

    def _one(pair: tuple[TensorHandle, TensorHandle]) -> TensorImpact:
        return _pair_metrics(
            pair[0], pair[1], spec,
            ref_side=ref_side,
            compute_spectrum=compute_spectrum,
            want_u1=want_u1,
            chunk_size=chunk_size,
        )

    if jobs_n > 1 and n_total > 1:
        from concurrent.futures import ThreadPoolExecutor

        try:
            from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]
        except ImportError:  # pragma: no cover - optional dep
            threadpool_limits = None
        if threadpool_limits is not None:
            try:
                with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=jobs_n) as ex:
                    for i, ti in enumerate(ex.map(_one, pairs)):
                        impacts.append(ti)
                        if i % report_every == 0 or i == n_total - 1:
                            _report(0.08 + 0.72 * ((i + 1) / n_total), f"Measuring impact ({i + 1}/{n_total})...")
            except RuntimeError:
                with ThreadPoolExecutor(max_workers=jobs_n) as ex:
                    for i, ti in enumerate(ex.map(_one, pairs)):
                        impacts.append(ti)
                        if i % report_every == 0 or i == n_total - 1:
                            _report(0.08 + 0.72 * ((i + 1) / n_total), f"Measuring impact ({i + 1}/{n_total})...")
        else:
            with ThreadPoolExecutor(max_workers=jobs_n) as ex:
                for i, ti in enumerate(ex.map(_one, pairs)):
                    impacts.append(ti)
                    if i % report_every == 0 or i == n_total - 1:
                        _report(0.08 + 0.72 * ((i + 1) / n_total), f"Measuring impact ({i + 1}/{n_total})...")
    else:
        for i, pair in enumerate(pairs):
            impacts.append(_one(pair))
            if i % report_every == 0 or i == n_total - 1:
                _report(0.08 + 0.72 * ((i + 1) / n_total), f"Measuring impact ({i + 1}/{n_total})...")

    # Release model payloads: handles are memoized, so clear them now that all
    # metrics are computed.
    for h in handles_a:
        h.clear()
    for h in handles_b:
        h.clear()

    warnings: list[str] = []
    if preset == "edit":
        nf = _noise_floor_policy(impacts, fp_a, fp_b)
        if nf["warning"] is not None:
            warnings.append(nf["warning"])
    else:
        sqnr_finite = sum(1 for t in impacts if np.isfinite(t.sqnr_db))
        if impacts and sqnr_finite / len(impacts) < 0.5:
            warnings.append(
                f"valid_fraction {sqnr_finite / len(impacts):.1%} < 50%: SQNR is NaN/"
                "inf for most tensor pairs (identical or silent tensors?)."
            )

    _report(0.85, "Writing artefacts...")
    artefacts = _write_artefacts(
        impacts, spec, out, ref_side, preset=preset, operator_impact=compute_spectrum
    )

    if preset == "edit":
        summary = _build_summary(impacts, skipped, fp_a, fp_b, ref_side, warnings)
        nf = _noise_floor_policy(impacts, fp_a, fp_b)
        body: dict[str, Any] = {
            "preset": preset,
            "ref_side": summary.ref_side,
            "model_a": summary.model_a,
            "model_b": summary.model_b,
            "alignment": summary.alignment,
            "edit_signature": _build_edit_signature(impacts, spec),
            "noise_floor": nf,
            "warnings": summary.warnings,
        }
        summary_path = out / "compare_summary.json"
    else:
        summary = _build_summary(impacts, skipped, fp_a, fp_b, ref_side, warnings)
        body = {
            "preset": preset,
            "ref_side": summary.ref_side,
            "model_a": summary.model_a,
            "model_b": summary.model_b,
            "alignment": summary.alignment,
            "global": summary.global_,
            "per_type": summary.per_type,
            "hotspot_ranking": summary.hotspot_ranking,
            "warnings": summary.warnings,
        }
        summary_path = out / "impact_summary.json"
    with open(summary_path, "w") as f:
        json.dump(body, f, indent=2, sort_keys=True)
        f.write("\n")
    artefacts.append(summary_path)

    _report(0.92, "Rendering sheets...")
    if preset == "edit":
        from weight_atlas.paired.render import EditSheet

        rendered = EditSheet().render(out, spec, out)
    else:
        from weight_atlas.paired.render import ImpactSheet

        rendered = ImpactSheet().render(out, spec, out)
    artefacts.extend(rendered)

    _report(0.97, "Writing manifest...")
    from weight_atlas.scan import _sha256

    manifest = {str(p.relative_to(out)): _sha256(p) for p in artefacts}
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    _report(1.0, "Done.")
    return artefacts + [manifest_path]


# Backwards-compatible alias: the quant preset is the historical M8/M9
# quantization-impact command.
run_impact = run_paired
