"""Frobenius and spectral norms, plus effective rank."""

from __future__ import annotations

import numpy as np

from weight_atlas.core.registry import register_stat
from weight_atlas.core.types import TensorHandle


def _flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1).astype(np.float64, copy=False)


def _to_matrix(x: np.ndarray) -> np.ndarray:
    """Flatten a tensor to a 2D matrix (rows=first dim, cols=rest)."""
    if x.ndim == 1:
        return x.reshape(1, -1)
    return x.reshape(x.shape[0], -1)


@register_stat("frobenius")
class FrobeniusNorm:
    """Frobenius norm with chunked float64 accumulation for stability."""

    stat_id = "frobenius"

    _CHUNK = 2**20  # 1M elements

    def compute(self, t: TensorHandle) -> float:
        x = _flatten(t.load())
        acc = np.float64(0.0)
        for i in range(0, x.size, self._CHUNK):
            chunk = x[i : i + self._CHUNK]
            acc += np.dot(chunk, chunk)
        return float(np.sqrt(acc))


@register_stat("spectral_norm")
class SpectralNorm:
    """Spectral norm (largest singular value).

    - 1-D tensors: ``L2`` norm (the spec allows computing these; no norm
      exception for bias vectors).
    - Small matrices (``min(m, n) <= 512``): exact SVD via numpy.
    - Larger: randomized SVD (Halko) with parameters from the spec
      (k=16, p=4, q=2) and an RNG seeded from ``spec.seeds.svd``.
    """

    stat_id = "spectral_norm"

    _SMALL = 512
    _K = 16
    _P = 4
    _Q = 2

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def compute(self, t: TensorHandle) -> float:
        x = t.load()
        if x.ndim == 1:
            return float(np.linalg.norm(x.astype(np.float64)))
        m = _to_matrix(x.astype(np.float64, copy=False))
        if min(m.shape) <= self._SMALL:
            s = np.linalg.svd(m, compute_uv=False)
            return float(s[0])
        return float(self._randomized_svd(m))

    def _randomized_svd(self, m: np.ndarray) -> float:
        rng = np.random.default_rng(self._seed)
        k = min(self._K, min(m.shape))
        omega = rng.standard_normal((m.shape[1], k)).astype(m.dtype)
        y = m @ omega
        for _ in range(self._Q):
            y = m @ (m.T @ y)
        q, _ = np.linalg.qr(y)
        b = q.T @ m
        s = np.linalg.svd(b, compute_uv=False)
        return float(s[0])


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
            norms = np.sqrt(np.square(x, dtype=np.float64).sum(axis=(1, 2, 3)))
            return float(norms.mean())
        return float(np.sqrt(np.square(x, dtype=np.float64).sum()))


@register_stat("effective_rank")
class EffectiveRank:
    """Effective rank = exp(-sum(p_i * log p_i)) where p = s / sum(s).

    Same 1-D / small / large dispatch as SpectralNorm. For 1-D tensors,
    effective rank is 1 (a single component carries all energy).
    """

    stat_id = "effective_rank"

    _SMALL = 512
    _K = 16
    _P = 4
    _Q = 2

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._spectral = SpectralNorm(seed=seed)

    def compute(self, t: TensorHandle) -> float:
        x = t.load()
        if x.ndim == 1:
            return 1.0
        m = _to_matrix(x.astype(np.float64, copy=False))
        if min(m.shape) <= self._SMALL:
            s = np.linalg.svd(m, compute_uv=False)
        else:
            s = self._randomized_svds(m)
        return float(self._entropy_rank(s))

    def _randomized_svds(self, m: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self._seed)
        k = min(self._K, min(m.shape))
        omega = rng.standard_normal((m.shape[1], k)).astype(m.dtype)
        y = m @ omega
        for _ in range(self._Q):
            y = m @ (m.T @ y)
        q, _ = np.linalg.qr(y)
        b = q.T @ m
        return np.linalg.svd(b, compute_uv=False)

    @staticmethod
    def _entropy_rank(s: np.ndarray) -> float:
        # Clip near-zero singular values to avoid log(0). Using a small
        # floor keeps the rank estimate stable without biasing large values.
        s = np.clip(s, 1e-12, None)
        p = s / s.sum()
        p = p[p > 0]
        return float(np.exp(-np.sum(p * np.log(p))))
