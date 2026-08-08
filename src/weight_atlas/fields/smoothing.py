"""Bilinear upsample + Gaussian smoothing."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


def upsample(field: np.ndarray, factor: int) -> np.ndarray:
    """Upsample by integer factor using bilinear interpolation via scipy.ndimage.zoom.

    NaN values are replaced with 0 before upsampling and re-masked to NaN
    afterwards so they do not bleed into finite neighbours.
    """
    if factor <= 1:
        return field.copy()
    mask = ~np.isfinite(field)
    clean = field.copy()
    clean[mask] = 0.0
    up: np.ndarray = zoom(clean, factor, order=1)
    # Build the upsampled mask: any source NaN dilates the NaN region.
    up_mask = zoom(mask.astype(np.float64), factor, order=1) > 0.5
    up[up_mask] = np.nan
    return up


def smooth(field: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smooth. NaN-safe: interpolate over NaNs, smooth, restore mask."""
    if sigma <= 0:
        return field.copy()
    mask = ~np.isfinite(field)
    clean = field.copy()
    # Fill NaNs with 0; track weight via a mask array so we can re-normalise.
    clean[mask] = 0.0
    weight = (~mask).astype(np.float64)
    smoothed = gaussian_filter(clean, sigma=sigma)
    w_smoothed = gaussian_filter(weight, sigma=sigma)
    out = np.full_like(field, np.nan)
    valid = w_smoothed > 1e-8
    out[valid] = smoothed[valid] / w_smoothed[valid]
    return out
