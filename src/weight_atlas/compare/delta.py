"""Delta computation and summary metrics for model comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from weight_atlas.compare.align import AlignResult, align
from weight_atlas.core.types import AtlasSpec
from weight_atlas.fields.scaling import apply_scale


@dataclass
class ChannelDelta:
    """Delta field and per-channel metrics for a single channel."""

    channel: str
    delta: np.ndarray  # B - A on scaled channel values
    abs_delta: np.ndarray  # |delta|
    rel_l2: float  # relative L2 norm of delta (raw stats)
    cosine_sim: float  # cosine similarity (raw stats)
    hotspot_layer: int  # layer index of max |delta|
    hotspot_slot: str  # slot name of max |delta|
    hotspot_value: float  # value at hotspot
    argmax: tuple[int, str]  # (layer, slot) of max |delta|


@dataclass
class CompareSummary:
    """Full comparison summary for two models."""

    mode: str
    spec_version: int
    model_a: dict[str, Any]
    model_b: dict[str, Any]
    channels: dict[str, ChannelDelta] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    aligned_row_labels: list[str] = field(default_factory=list)
    aligned_col_labels: list[str] = field(default_factory=list)
    # Aligned-mode metadata for the report: resampling method + layer maps.
    alignment: dict[str, Any] = field(default_factory=dict)


def _get_aligned_interp(spec: AtlasSpec) -> str:
    """Return the aligned-mode row resampling method from the spec.

    ``compare.aligned_interp`` accepts ``"linear"`` (default) or
    ``"nearest"`` (nearest-layer matching by normalized depth).
    """
    interp = spec.compare.get("aligned_interp", "linear")
    return interp if interp in ("linear", "nearest") else "linear"


def compute_delta(
    aligned: AlignResult,
    channel: str,
    spec: AtlasSpec,
) -> ChannelDelta:
    """Compute delta on scaled channel values (B - A).

    Applies the channel scale to both fields, then computes delta.
    Summary metrics (rel_l2, cosine_sim) are computed on the raw (unscaled) fields.
    """
    data_a = aligned.data_a
    data_b = aligned.data_b

    # Summary metrics on raw stats (before scaling). Both use the SAME
    # position-aligned element set (intersection of the finite masks):
    # masking independently shifts row-major order whenever the NaN
    # footprints differ (missing slots, aligned-mode NaN column padding),
    # and the cosine would then compare unrelated elements.
    both_finite = np.isfinite(data_a) & np.isfinite(data_b)

    rel_l2 = _compute_rel_l2(data_a, data_b)
    cosine_sim = _compute_cosine_sim(data_a[both_finite], data_b[both_finite])

    # Apply scale to both fields for delta computation
    scale_spec = spec.channel_scale(channel)
    scaled_a = apply_scale(data_a, scale_spec)
    scaled_b = apply_scale(data_b, scale_spec)

    # Delta on scaled values (B - A)
    delta = _safe_subtract(scaled_b, scaled_a)
    abs_delta = np.abs(delta)

    # Find hotspot (argmax of |delta|)
    hotspot_layer_idx, hotspot_slot_idx, hotspot_value = _find_hotspot(abs_delta)
    hotspot_layer = hotspot_layer_idx
    hotspot_slot = (
        aligned.col_labels[hotspot_slot_idx]
        if hotspot_slot_idx < len(aligned.col_labels)
        else str(hotspot_slot_idx)
    )

    return ChannelDelta(
        channel=channel,
        delta=delta,
        abs_delta=abs_delta,
        rel_l2=rel_l2,
        cosine_sim=cosine_sim,
        hotspot_layer=hotspot_layer,
        hotspot_slot=hotspot_slot,
        hotspot_value=hotspot_value,
        argmax=(hotspot_layer, hotspot_slot),
    )


def _compute_rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Compute ||a - b||_2 / ||a||_2 on finite elements.

    If a is all zeros or has no finite elements, returns 0.0.
    """
    # Get finite mask for both
    finite_a = np.isfinite(a)
    finite_b = np.isfinite(b)
    both_finite = finite_a & finite_b

    if not both_finite.any():
        return 0.0

    a_finite = a[both_finite]
    b_finite = b[both_finite]

    diff = a_finite - b_finite
    norm_diff = float(np.linalg.norm(diff))
    norm_a = float(np.linalg.norm(a_finite))

    if norm_a == 0.0:
        return 0.0
    return norm_diff / norm_a


def _compute_cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two position-aligned element sets.

    ``a`` and ``b`` must be extracted at identical positions (the caller
    intersects the finite masks). Mismatched sizes mean a misaligned
    extraction — raise instead of zero-padding and comparing shifted
    elements.
    """
    if a.size != b.size:
        raise ValueError(
            f"cosine_sim inputs must be position-aligned "
            f"({a.size} vs {b.size} elements)"
        )

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _safe_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Subtract two fields element-wise, preserving NaN where either is NaN."""
    result = np.full_like(a, np.nan, dtype=np.float64)
    both_finite = np.isfinite(a) & np.isfinite(b)
    result[both_finite] = a[both_finite] - b[both_finite]
    return result


def _find_hotspot(abs_delta: np.ndarray) -> tuple[int, int, float]:
    """Find the location and value of the maximum |delta|.

    Returns (row_idx, col_idx, value). If all NaN, returns (0, 0, 0.0).
    """
    if not np.isfinite(abs_delta).any():
        return 0, 0, 0.0

    # Use nanargmax on flattened
    flat = abs_delta.copy()
    flat[~np.isfinite(flat)] = -1.0
    idx = int(np.argmax(flat))
    row, col = divmod(idx, abs_delta.shape[1])
    value = float(abs_delta[row, col])
    return row, col, value


def compute_compare_summary(
    field_a: np.ndarray,
    field_b: np.ndarray,
    spec: AtlasSpec,
    *,
    mode: str = "strict",
    interp: str | None = None,
    fingerprint_a: dict[str, Any] | None = None,
    fingerprint_b: dict[str, Any] | None = None,
    row_labels_a: list[str] | None = None,
    row_labels_b: list[str] | None = None,
) -> CompareSummary:
    """Full comparison pipeline: align + delta + summary metrics.

    Args:
        field_a, field_b: raw fields (e.g., spectral_norm) for the same channel
        spec: atlas specification
        mode: "strict" or "aligned"
        interp: aligned-mode row resampling ("linear" or "nearest"). Defaults to
            the spec's ``compare.aligned_interp`` setting.
        fingerprint_a, fingerprint_b: optional fingerprints for metadata
        row_labels_a, row_labels_b: optional row labels from scan
    """
    from weight_atlas.compare.align import check_compatibility

    # Check compatibility if fingerprints provided
    warnings: list[str] = []
    if fingerprint_a is not None and fingerprint_b is not None:
        warnings.extend(check_compatibility(fingerprint_a, fingerprint_b))

        # Check for loader/quantization mismatch (warning only, no reject)
        loader_a = fingerprint_a.get("loader", "unknown")
        loader_b = fingerprint_b.get("loader", "unknown")
        if loader_a != loader_b:
            warnings.append(
                f"loader mismatch: A={loader_a}, B={loader_b}. The scanned "
                "fields are rank-normalized, so the topographic structure is "
                "comparable across formats, but absolute magnitudes (rel_l2 / "
                "cosine_sim) carry quantization/format noise. For "
                "magnitude-exact comparisons use the same loader and "
                "quantization on both sides."
            )

        # Check for quantization mismatch
        quant_a = fingerprint_a.get("quantization", {})
        quant_b = fingerprint_b.get("quantization", {})
        if quant_a and quant_b and quant_a != quant_b:
            warnings.append(
                f"quantization mismatch: A={quant_a}, B={quant_b}. "
                "Quantization noise affects signature; for abliteration studies "
                "both sides should use identical quantization."
            )

    # Align fields
    if interp is None:
        interp = _get_aligned_interp(spec)
    aligned = align(
        field_a, field_b, spec,
        mode=mode,
        row_labels_a=row_labels_a,
        row_labels_b=row_labels_b,
        interp=interp,
    )
    warnings.extend(aligned.warnings)

    # Compute delta for each channel in the spec
    channels: dict[str, ChannelDelta] = {}
    for channel in spec.channels:
        channels[channel] = compute_delta(aligned, channel, spec)

    # Alignment metadata for the report: expose how rows were matched so
    # readers never mistake aligned-mode layer indices for absolute ones.
    alignment: dict[str, Any] = {
        "mode": mode,
        "n_rows_a": int(field_a.shape[0]),
        "n_rows_b": int(field_b.shape[0]),
        "n_rows_common": int(aligned.data_a.shape[0]),
    }
    if mode == "aligned":
        alignment["interp"] = aligned.interp
        alignment["layer_map_a"] = aligned.layer_map_a
        alignment["layer_map_b"] = aligned.layer_map_b

    return CompareSummary(
        mode=mode,
        spec_version=spec.spec_version,
        model_a=_extract_model_meta(fingerprint_a),
        model_b=_extract_model_meta(fingerprint_b),
        channels=channels,
        warnings=warnings,
        aligned_row_labels=aligned.row_labels,
        aligned_col_labels=aligned.col_labels,
        alignment=alignment,
    )


def _extract_model_meta(fingerprint: dict[str, Any] | None) -> dict[str, Any]:
    """Extract model metadata from a fingerprint."""
    if fingerprint is None:
        return {}
    return {
        "tool_version": fingerprint.get("tool_version", "unknown"),
        "loader": fingerprint.get("loader", "unknown"),
        "n_tensors": fingerprint.get("model", {}).get("n_tensors", 0),
        "n_layers": fingerprint.get("model", {}).get("n_layers", 0),
        "quantization": fingerprint.get("quantization", {}),
    }


def hotspot_ranking(
    delta: ChannelDelta,
    col_labels: list[str] | None = None,
    top_k: int = 5,
) -> list[tuple[int, str, float]]:
    """Return top-k hotspot locations ranked by |delta| value.

    Returns list of (layer, slot_name, abs_delta_value) sorted descending.
    """
    abs_d = delta.abs_delta.copy()
    # Replace NaN with -1 so they sort last
    abs_d[~np.isfinite(abs_d)] = -1.0

    # Flatten and get top-k indices
    flat = abs_d.flatten()
    if flat.size == 0:
        return []

    # Filter out NaN positions (marked as -1)
    # Only consider positions with actual values (>= 0 for abs values)
    valid_mask = flat >= 0
    if not valid_mask.any():
        return []

    valid_indices = np.where(valid_mask)[0]
    valid_values = flat[valid_indices]

    # Get top-k from valid values only
    top_k = min(top_k, valid_values.size)
    if top_k < valid_values.size:
        top_local = np.argpartition(valid_values, -top_k)[-top_k:]
        top_local = top_local[np.argsort(valid_values[top_local])[::-1]]
    else:
        top_local = np.argsort(valid_values)[::-1]

    results: list[tuple[int, str, float]] = []
    n_cols = abs_d.shape[1]
    for local_idx in top_local:
        global_idx = valid_indices[local_idx]
        row, col = divmod(int(global_idx), n_cols)
        slot = col_labels[col] if col_labels and col < len(col_labels) else str(col)
        results.append((row, slot, float(valid_values[local_idx])))

    return results
