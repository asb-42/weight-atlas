"""Channel scaling: log1p, robust_scale, rank_scale (v2.1 unified pipeline)."""

from __future__ import annotations

import numpy as np


def log1p(field: np.ndarray) -> np.ndarray:
    """log(1 + x), element-wise, operating on a copy."""
    out = field.copy()
    finite = np.isfinite(out)
    out[finite] = np.log1p(np.maximum(out[finite], 0.0))
    return out


def robust_scale(field: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    """Robust percentile-based scaling to [0, 1].

    1. Compute q_lo = percentile(field, lower)
       Compute q_hi = percentile(field, upper)
    2. Clip field to [q_lo, q_hi]
    3. Min-max normalize clipped range to [0, 1]
    NaN is preserved.
    """
    out = field.copy()
    finite = np.isfinite(out)
    vals = out[finite]
    if vals.size == 0:
        return out
    qlo = float(np.quantile(vals, lower))
    qhi = float(np.quantile(vals, upper))
    np.clip(out, qlo, qhi, out=out)
    denom = qhi - qlo
    if denom > 0:
        out[finite] = (out[finite] - qlo) / denom
    else:
        out[finite] = 0.0
    return out


def rank_scale(field: np.ndarray) -> np.ndarray:
    """Rank-based normalization to [0, 1].

    Each cell gets its percentile rank within the distribution:
    u_i = rank(x_i) / N

    This guarantees full color utilization regardless of outliers.
    A single extreme value no longer saturates the colormap.

    Properties:
    - Immune to outliers of any magnitude
    - Always uses full [0, 1] range
    - Works well even with small N (though resolution suffers)
    - Makes images "shape-comparable" (pattern, structure, texture)
    - Does NOT preserve magnitude comparability between models

    NaN is preserved.
    """
    out = field.copy()
    finite = np.isfinite(out)
    vals = out[finite]
    if vals.size == 0:
        return out

    # Compute ranks (0-based) and normalize to [0, 1]
    # argsort of argsort gives the rank of each element
    ranks = np.empty_like(vals, dtype=np.float64)
    order = np.argsort(vals)
    ranks[order] = np.arange(vals.size, dtype=np.float64) / (vals.size - 1) if vals.size > 1 else 0.5
    out[finite] = ranks
    return out


def quantile_clip(field: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> np.ndarray:
    """Backward-compatible alias for robust_scale with lo/hi parameter names."""
    return robust_scale(field, lower=lo, upper=hi)


_SCALE_FNS = {"log1p": log1p, "robust_scale": robust_scale, "rank_scale": rank_scale, "quantile_clip": quantile_clip}


def apply_scale(field: np.ndarray, scale_spec: dict) -> np.ndarray:
    """Apply a channel scale specification to a field."""
    typ = scale_spec["type"]
    if typ == "log1p":
        return log1p(field)
    if typ == "robust_scale":
        return robust_scale(field, lower=float(scale_spec.get("lower", 0.01)), upper=float(scale_spec.get("upper", 0.99)))
    if typ == "rank_scale":
        return rank_scale(field)
    if typ == "quantile_clip":
        return quantile_clip(field, lo=float(scale_spec["lo"]), hi=float(scale_spec["hi"]))
    raise ValueError(f"unknown scale type: {typ}")
