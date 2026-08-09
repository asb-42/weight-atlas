"""Stable rank statistic: log1p((frobenius / spectral_norm)²).

Stable rank is a continuous, bounded measure of how "spread out" the
singular values of a matrix are. For a rank-r matrix with equal singular
values, stable_rank = r. For a rank-1 matrix, stable_rank = 0 (since
log1p(1) is not 0, but for a pure rank-1 matrix frobenius == spectral_norm
so the ratio is 1 and log1p(1) = log(2) ≈ 0.693 — the minimum).

The formula: stable_rank = log1p((||A||_F / ||A||_2)²)

Properties:
- Always >= log(2) ≈ 0.693 for any non-zero matrix
- Equal to log1p(r) for a rank-r matrix with equal singular values
- More robust than effective_rank for noisy/quantized tensors
"""

from __future__ import annotations

import numpy as np

from weight_atlas.core.registry import register_stat
from weight_atlas.core.types import TensorHandle
from weight_atlas.stats.norms import FrobeniusNorm, SpectralNorm


@register_stat("stable_rank")
class StableRank:
    """Stable rank = log1p((frobenius / spectral_norm)²).

    For 1-D tensors (vectors), stable_rank = log(2) since frobenius == spectral_norm.
    """

    stat_id = "stable_rank"

    def __init__(self, seed: int = 0) -> None:
        self._spectral = SpectralNorm(seed=seed)

    def compute(self, t: TensorHandle) -> float:
        x = t.load()
        frob = FrobeniusNorm().compute(t)
        spec = self._spectral.compute(t)
        if spec == 0.0:
            return 0.0
        ratio_sq = (frob / spec) ** 2
        return float(np.log1p(ratio_sq))
