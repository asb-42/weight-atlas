"""Kurtosis and sparsity statistics (chunked float64 accumulation)."""

from __future__ import annotations

import numpy as np

from weight_atlas.core.registry import register_stat
from weight_atlas.core.types import TensorHandle

_CHUNK = 2**20  # 1M elements


@register_stat("kurtosis")
class Kurtosis:
    """Excess kurtosis (Fisher) of the flattened tensor values.

    Two-pass over the float32 payload with chunked float64 accumulation: one
    pass for the mean, one for the central second and fourth moments. This
    avoids materializing full-size float64 arrays (the previous
    implementation allocated ``diff``/``diff**2``/``diff**4`` copies, which
    dominated scan time and memory for large tensors). Values agree with the
    vectorized computation to float64 rounding.
    """

    stat_id = "kurtosis"

    def compute(self, t: TensorHandle) -> float:
        x = t.load().reshape(-1)
        n = x.size
        if n < 2:
            return 0.0

        mean = 0.0
        for i in range(0, n, _CHUNK):
            mean += float(x[i : i + _CHUNK].astype(np.float64).sum())
        mean /= n

        m2 = 0.0
        m4 = 0.0
        for i in range(0, n, _CHUNK):
            d = x[i : i + _CHUNK].astype(np.float64) - mean
            dd = d * d
            m2 += float(dd.sum())
            m4 += float((dd * dd).sum())
        m2 /= n
        m4 /= n

        if m2 == 0:
            return -3.0  # constant array -> excess kurtosis of degenerate dist
        return float(m4 / (m2 * m2) - 3.0)


@register_stat("sparsity")
class Sparsity:
    """Fraction of weights with absolute value below ``eps``."""

    stat_id = "sparsity"
    _EPS = 1e-3

    def compute(self, t: TensorHandle) -> float:
        x = t.load().reshape(-1)
        if x.size == 0:
            return 0.0
        n_below = 0
        for i in range(0, x.size, _CHUNK):
            n_below += int(np.count_nonzero(np.abs(x[i : i + _CHUNK]) < self._EPS))
        return n_below / x.size
