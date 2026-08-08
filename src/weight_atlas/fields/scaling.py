"""Channel scaling: log1p, quantile_clip."""

from __future__ import annotations

import numpy as np


def log1p(field: np.ndarray) -> np.ndarray:
    """log(1 + x), element-wise, operating on a copy."""
    out = field.copy()
    finite = np.isfinite(out)
    out[finite] = np.log1p(np.maximum(out[finite], 0.0))
    return out


def quantile_clip(field: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> np.ndarray:
    """Clip finite values to the [lo, hi] quantile range, then min-max normalise to [0, 1].

    Operates on a copy. NaN is preserved.
    """
    out = field.copy()
    finite = np.isfinite(out)
    vals = out[finite]
    if vals.size == 0:
        return out
    qlo = float(np.quantile(vals, lo))
    qhi = float(np.quantile(vals, hi))
    np.clip(out, qlo, qhi, out=out)
    denom = qhi - qlo
    if denom > 0:
        out[finite] = (out[finite] - qlo) / denom
    else:
        out[finite] = 0.0
    return out


_SCALE_FNS = {"log1p": log1p, "quantile_clip": quantile_clip}


def apply_scale(field: np.ndarray, scale_spec: dict) -> np.ndarray:
    """Apply a channel scale specification to a field."""
    typ = scale_spec["type"]
    if typ == "log1p":
        return log1p(field)
    if typ == "quantile_clip":
        return quantile_clip(field, lo=float(scale_spec["lo"]), hi=float(scale_spec["hi"]))
    raise ValueError(f"unknown scale type: {typ}")
