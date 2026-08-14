"""Field alignment: strict (same architecture) and aligned (cross-architecture, normalized depth)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import zoom

from weight_atlas.core.types import AtlasSpec


@dataclass
class AlignResult:
    """Result of aligning two fields for comparison."""

    data_a: np.ndarray  # aligned field A
    data_b: np.ndarray  # aligned field B
    row_labels: list[str]
    col_labels: list[str]
    mode: str  # "strict" or "aligned"
    warnings: list[str]
    interp: str = "linear"  # "linear" (zoom) or "nearest" (layer matching)
    # Original layer index of each aligned row (aligned mode only).
    layer_map_a: list[int] | None = None
    layer_map_b: list[int] | None = None


def check_compatibility(
    spec_a: dict[str, Any],
    spec_b: dict[str, Any],
) -> list[str]:
    """Check compatibility between two fingerprint specs.

    Returns a list of warnings. Raises ValueError on hard incompatibility
    (spec_version mismatch).
    """
    warnings: list[str] = []

    sv_a = spec_a.get("spec_version")
    sv_b = spec_b.get("spec_version")
    if sv_a != sv_b:
        raise ValueError(
            f"spec_version mismatch: A={sv_a}, B={sv_b}. "
            "Cannot compare models scanned with different spec versions."
        )

    tv_a = spec_a.get("tool_version", "unknown")
    tv_b = spec_b.get("tool_version", "unknown")
    if tv_a != tv_b:
        warnings.append(
            f"tool_version mismatch: A={tv_a}, B={tv_b}. "
            "Results may not be fully comparable."
        )

    loader_a = spec_a.get("loader", "unknown")
    loader_b = spec_b.get("loader", "unknown")
    if loader_a != loader_b:
        warnings.append(f"loader mismatch: A={loader_a}, B={loader_b}")

    return warnings


def align(
    field_a: np.ndarray,
    field_b: np.ndarray,
    spec: AtlasSpec,
    *,
    mode: str = "strict",
    row_labels_a: list[str] | None = None,
    row_labels_b: list[str] | None = None,
    interp: str = "linear",
) -> AlignResult:
    """Align two fields for comparison.

    Strict mode: requires same shape, identical row/col indices.
    Aligned mode: normalizes depth to t∈[0,1], resamples to common grid.

    ``interp`` controls how aligned mode resamples rows:
    - ``"linear"`` (default): bilinear interpolation via scipy zoom — smooth but
      smears layer-local structure across interpolated rows.
    - ``"nearest"``: maps each normalized depth to the nearest actual layer
      index and copies that row verbatim — preserves layer structure and NaN
      holes, at the cost of stepping behaviour.
    """
    if mode == "strict":
        return _align_strict(field_a, field_b, spec, row_labels_a, row_labels_b)
    if mode == "aligned":
        return _align_normalized(field_a, field_b, spec, row_labels_a, row_labels_b, interp)
    raise ValueError(f"unknown align mode: {mode}")


def _align_strict(
    field_a: np.ndarray,
    field_b: np.ndarray,
    spec: AtlasSpec,
    row_labels_a: list[str] | None,
    row_labels_b: list[str] | None,
) -> AlignResult:
    """Strict alignment: same architecture, same indices."""
    warnings: list[str] = []

    if field_a.shape != field_b.shape:
        raise ValueError(
            f"strict mode requires identical shapes: A={field_a.shape}, B={field_b.shape}. "
            "Use --mode aligned for cross-architecture comparison."
        )

    # Verify row labels match if provided
    if row_labels_a is not None and row_labels_b is not None and row_labels_a != row_labels_b:
        warnings.append("row_labels differ between A and B")

    return AlignResult(
        data_a=field_a.copy(),
        data_b=field_b.copy(),
        row_labels=list(row_labels_a) if row_labels_a else [str(i) for i in range(field_a.shape[0])],
        col_labels=list(spec.slots),
        mode="strict",
        warnings=warnings,
    )


def _align_normalized(
    field_a: np.ndarray,
    field_b: np.ndarray,
    spec: AtlasSpec,
    row_labels_a: list[str] | None,
    row_labels_b: list[str] | None,
    interp: str = "linear",
) -> AlignResult:
    """Aligned mode: normalize depth to t∈[0,1], resample to common grid."""
    warnings: list[str] = []

    n_rows_a = field_a.shape[0]
    n_rows_b = field_b.shape[0]

    # Use the larger grid for better resolution
    n_rows_max = max(n_rows_a, n_rows_b)
    n_rows_common = max(n_rows_max, 64)  # at least 64 depth samples

    # Determine common column count (max of both, but at least spec.slots)
    n_cols_a = field_a.shape[1]
    n_cols_b = field_b.shape[1]
    n_cols_common = max(n_cols_a, n_cols_b, len(spec.slots))

    # Resample A to common grid
    layer_map_a: list[int] | None = None
    layer_map_b: list[int] | None = None
    if interp == "nearest":
        field_a_aligned, layer_map_a = _resample_rows_nearest(field_a, n_rows_common)
        field_b_aligned, layer_map_b = _resample_rows_nearest(field_b, n_rows_common)
    elif interp == "linear":
        field_a_aligned = _resample_field(field_a, n_rows_common, n_cols_common)
        field_b_aligned = _resample_field(field_b, n_rows_common, n_cols_common)
    else:
        raise ValueError(f"unknown interp method: {interp}")

    row_labels = [f"{t:.3f}" for t in np.linspace(0, 1, n_rows_common)]

    if n_rows_a != n_rows_b:
        warnings.append(
            f"different layer counts (A={n_rows_a}, B={n_rows_b}), "
            f"resampled to {n_rows_common} common depth grid"
        )
    if interp == "linear":
        warnings.append(
            "aligned mode uses bilinear interpolation: rows are normalized "
            "depth t∈[0,1], NOT absolute layer indices — layer 15/40 is "
            "compared with the t=0.375 row of B, not B layer 15."
        )
    else:
        warnings.append(
            "aligned mode uses nearest-layer matching: each row copies the "
            "nearest real layer of A and B by normalized depth t∈[0,1], so "
            "layer 15/40 (t=0.375) is compared with B's t=0.375 layer."
        )

    return AlignResult(
        data_a=field_a_aligned,
        data_b=field_b_aligned,
        row_labels=row_labels,
        col_labels=list(spec.slots),
        mode="aligned",
        warnings=warnings,
        interp=interp,
        layer_map_a=layer_map_a,
        layer_map_b=layer_map_b,
    )


def _resample_rows_nearest(field: np.ndarray, n_rows: int) -> tuple[np.ndarray, list[int]]:
    """Resample rows by nearest-layer matching on normalized depth.

    Each target row t∈[0,1] copies the source row whose index is closest to
    ``round(t * (n_source - 1))``. NaN cells are preserved verbatim (no
    interpolation), and the layer mapping is returned for reporting.
    """
    n_src = field.shape[0]
    if n_src == n_rows:
        return field.copy(), list(range(n_src))
    t = np.linspace(0, 1, n_rows)
    src_idx = np.round(t * (n_src - 1)).astype(int)
    return field[src_idx, :], [int(i) for i in src_idx]


def _resample_field(field: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Resample a field to a target shape using bilinear interpolation.

    NaN-safe: fills NaNs with 0 for resampling, re-masks afterwards.
    """
    if field.shape == (n_rows, n_cols):
        return field.copy()

    # Calculate zoom factors
    zoom_row = n_rows / field.shape[0]
    zoom_col = n_cols / field.shape[1]

    # Handle NaNs
    mask = ~np.isfinite(field)
    clean = field.copy()
    clean[mask] = 0.0

    # Resample
    resampled = zoom(clean, (zoom_row, zoom_col), order=1)

    # Resample mask to identify NaN-affected regions
    mask_float = mask.astype(np.float64)
    resampled_mask = zoom(mask_float, (zoom_row, zoom_col), order=1) > 0.5
    resampled[resampled_mask] = np.nan

    return resampled
