"""Kurtosis and sparsity statistics."""

from __future__ import annotations

import numpy as np

from weight_atlas.core.registry import register_stat
from weight_atlas.core.types import TensorHandle


@register_stat("kurtosis")
class Kurtosis:
    """Excess kurtosis (Fisher) of the flattened tensor values."""

    stat_id = "kurtosis"

    def compute(self, t: TensorHandle) -> float:
        x = t.load().reshape(-1).astype(np.float64)
        if x.size < 2:
            return 0.0
        mean = x.mean()
        diff = x - mean
        m2 = np.mean(diff ** 2)
        if m2 == 0:
            return -3.0  # constant array -> excess kurtosis of degenerate dist
        m4 = np.mean(diff ** 4)
        return float(m4 / (m2 ** 2) - 3.0)


@register_stat("sparsity")
class Sparsity:
    """Fraction of weights with absolute value below ``eps``."""

    stat_id = "sparsity"
    _EPS = 1e-3

    def compute(self, t: TensorHandle) -> float:
        x = t.load().reshape(-1)
        if x.size == 0:
            return 0.0
        return float(np.sum(np.abs(x) < self._EPS) / x.size)
