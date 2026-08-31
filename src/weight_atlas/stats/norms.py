"""Frobenius and spectral norms, plus effective rank.

All SVD-based statistics share a single truncated spectrum per tensor (see
:mod:`weight_atlas.stats.spectrum`) — the scan pipeline computes all stats
per tensor, so sharing turns three SVDs into one. O(n) statistics are
computed with chunked float64 accumulation over the float32 payload to bound
memory and runtime on large (MoE) models.
"""

from __future__ import annotations

import numpy as np

from weight_atlas.core.registry import register_stat
from weight_atlas.core.types import TensorHandle
from weight_atlas.stats.spectrum import entropy_rank, truncated_spectrum

_CHUNK = 2**20  # 1M elements


def _sum_of_squares(x: np.ndarray) -> float:
    """Chunked sum of squares in float64 over a float32 array (bounded memory)."""
    xf = x.reshape(-1)
    acc = np.float64(0.0)
    for i in range(0, xf.size, _CHUNK):
        chunk = xf[i : i + _CHUNK].astype(np.float64)
        acc += np.dot(chunk, chunk)
    return float(acc)


@register_stat("frobenius")
class FrobeniusNorm:
    """Frobenius norm with chunked float64 accumulation for stability."""

    stat_id = "frobenius"

    def compute(self, t: TensorHandle) -> float:
        return float(np.sqrt(_sum_of_squares(t.load())))


@register_stat("spectral_norm")
class SpectralNorm:
    """Spectral norm (largest singular value) from the shared spectrum.

    - 1-D tensors: ``L2`` norm (the spec allows computing these; no norm
      exception for bias vectors).
    - Small matrices (``min(m, n) <= 512``): exact SVD via numpy.
    - Larger: randomized SVD (Halko) with parameters from
      :mod:`weight_atlas.stats.spectrum` (k=16, q=2) and an RNG seeded from
      ``spec.seeds.svd``. Computed once per tensor and shared with
      ``effective_rank`` and ``stable_rank``.
    """

    stat_id = "spectral_norm"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def compute(self, t: TensorHandle) -> float:
        return float(truncated_spectrum(t, seed=self._seed)[0])


@register_stat("kernel_norm")
class KernelNorm:
    """Mean per-output-channel L2 norm of a conv kernel; Frobenius otherwise.

    Conv kernels are 4-D ``(C_out, C_in, kh, kw)``. The spectral norm of their
    2-D flattening mixes the output-channel axis with the spatial axes, which
    does not reflect how convolution kernels are structured. Reporting the
    mean per-output-channel norm gives a magnitude signature matched to the
    convolution layout (vision towers use Conv kernels instead of attention
    projections). Non-4-D tensors fall back to the Frobenius norm.
    """

    stat_id = "kernel_norm"

    def compute(self, t: TensorHandle) -> float:
        x = t.load()
        if x.ndim == 4:
            norms = np.empty(x.shape[0], dtype=np.float64)
            for c in range(x.shape[0]):
                norms[c] = np.sqrt(_sum_of_squares(x[c]))
            return float(norms.mean())
        return float(np.sqrt(_sum_of_squares(x)))


@register_stat("effective_rank")
class EffectiveRank:
    """Effective rank = exp(-sum(p_i * log p_i)) where p = s / sum(s).

    Same 1-D / small / large dispatch as SpectralNorm, derived from the same
    shared spectrum. For 1-D tensors, effective rank is 1 (a single component
    carries all energy).
    """

    stat_id = "effective_rank"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def compute(self, t: TensorHandle) -> float:
        if t.load().ndim == 1:
            return 1.0
        return entropy_rank(truncated_spectrum(t, seed=self._seed))


@register_stat("sv_decay")
class SVDecay:
    """Spectral tail decay: smallest / largest singular value of the spectrum.

    Adopted from alesha-pro/atlas (MIT): ``σ_k / σ_1`` over the shared
    (possibly truncated) spectrum — small values mean the energy is
    concentrated in the top modes. For 1-D tensors the concept is **not
    applicable**: compute returns NaN (never a fake 0/1).
    """

    stat_id = "sv_decay"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def compute(self, t: TensorHandle) -> float:
        if t.load().ndim == 1:
            return float("nan")
        s = truncated_spectrum(t, seed=self._seed)
        if s.size == 0 or s[0] == 0:
            return float("nan")
        return float(s[-1] / s[0])
