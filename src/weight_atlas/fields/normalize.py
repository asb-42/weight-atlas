"""Normalized-depth projection (Ebene 2): make fields comparable by relative depth.

Level-1 fields (``Field2D``) are indexed by absolute layer index, so models with
different layer counts have incomparable row axes and absent slots leave NaN
"perforation" holes in the sheet.

``project_normalized_depth`` projects each column onto a fixed set of depth
landmarks (0 % … 100 % of the layer stack), interpolating over the measured
rows, and returns an interpolation mask marking every cell whose value came from
interpolation rather than a directly measured row.
"""

from __future__ import annotations

import numpy as np


def project_normalized_depth(
    data: np.ndarray,
    n_landmarks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project rows onto ``n_landmarks`` normalized-depth positions.

    Each column is linearly interpolated (in relative depth) over its non-NaN
    values onto ``np.linspace(0, 1, n_landmarks)``. Landmarks outside a column's
    measured range stay NaN (extreme depth regions are honestly left as holes,
    not extrapolated).

    A landmark cell is marked *measured* when a directly measured row lies within
    half a landmark spacing of it; everything else — values that bridge a data
    gap or sit between sparse rows — is marked interpolated.

    Returns:
        ``(projected, interp_mask)`` where ``projected`` has shape
        ``(n_landmarks, n_cols)`` and ``interp_mask`` is a bool array of the
        same shape (``True`` = interpolated/unmeasured).
    """
    data = np.asarray(data, dtype=np.float64)
    n_rows, n_cols = data.shape

    if n_rows <= 1 or n_landmarks <= 1:
        # A single-row field has no depth axis to interpolate across.
        return data.copy(), np.zeros_like(data, dtype=bool)

    landmarks = np.linspace(0.0, 1.0, n_landmarks)
    row_pos = np.linspace(0.0, 1.0, n_rows)
    band = 0.5 / (n_landmarks - 1)

    projected = np.full((n_landmarks, n_cols), np.nan)
    interp_mask = np.zeros((n_landmarks, n_cols), dtype=bool)

    for c in range(n_cols):
        col = data[:, c]
        valid = np.isfinite(col)
        if not valid.any():
            interp_mask[:, c] = True
            continue
        pos = row_pos[valid]
        vals = col[valid]
        projected[:, c] = np.interp(landmarks, pos, vals, left=np.nan, right=np.nan)
        # Distance from each landmark to the nearest measured row in this column.
        dist = np.abs(landmarks[:, None] - pos[None, :]).min(axis=1)
        interp_mask[:, c] = dist > band

    return projected, interp_mask


def depth_landmark_labels(n_landmarks: int) -> list[str]:
    """Percentage labels for the normalized-depth landmarks (``0%`` … ``100%``)."""
    if n_landmarks <= 1:
        return ["100%"] if n_landmarks == 1 else []
    return [f"{int(round(p * 100))}%" for p in np.linspace(0.0, 1.0, n_landmarks)]
