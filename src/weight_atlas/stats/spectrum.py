"""Shared truncated spectrum: one (randomized) SVD per tensor.

Spectral norm, effective rank and stable rank all derive from the same
singular values. Computing them independently used to run up to three SVDs
per tensor — the dominant cost of scanning MoE models, which have tens of
thousands of expert tensors. Sharing the spectrum makes it one SVD per
tensor, with bit-identical values (same seed, same algorithm as before).

Matrices with ``min(m, n) <= SMALL`` use an exact SVD; larger ones use the
randomized SVD (Halko) with ``K`` columns and ``Q`` power iterations, run in
float32 (measured ~2x faster and half the memory of float64; the resulting
singular values differ by <1e-8 relative).

The spectrum is memoized per ``TensorHandle`` (weakly), so any number of
statistics on the same handle pay one SVD.
"""

from __future__ import annotations

import threading
import weakref

import numpy as np

from weight_atlas.core.types import TensorHandle

# Matrices with min(m, n) <= SMALL use an exact SVD.
SMALL = 512
# Randomized SVD parameters (documented in ARCHITECTURE.md).
K = 16
Q = 2

_cache: weakref.WeakKeyDictionary[TensorHandle, np.ndarray] = weakref.WeakKeyDictionary()

# OpenBLAS's LAPACK SVD/QR routines are not safe for concurrent invocation from
# multiple Python threads: they can deadlock inside OpenBLAS's internal thread
# pool. The scan pipeline computes statistics in parallel, and models with small
# expert tensors (min dim <= SMALL) hit the exact-SVD path tens of thousands of
# times, so concurrent calls deadlock reliably. Every LAPACK call is therefore
# serialized behind this lock. Only the LAPACK calls (qr/svd) need the lock —
# plain GEMMs (the rSVD power iterations) are thread-safe and run outside it,
# so the expensive O(m*n*k) matmuls stay parallel across workers.
_spectrum_lock = threading.Lock()


def to_matrix(x: np.ndarray) -> np.ndarray:
    """Flatten a tensor to a 2D matrix (rows=first dim, cols=rest)."""
    if x.ndim == 1:
        return x.reshape(1, -1)
    return x.reshape(x.shape[0], -1)


def _randomized_range(x: np.ndarray, seed: int) -> np.ndarray:
    """Halko power-iteration range finder (pure GEMMs — no LAPACK).

    Deterministic per (input, seed); safe to run without ``_spectrum_lock``.
    """
    rng = np.random.default_rng(seed)
    k = min(K, min(x.shape))
    omega = rng.standard_normal((x.shape[1], k)).astype(np.float32)
    y = x @ omega
    for _ in range(Q):
        y = x @ (x.T @ y)
    return y


def truncated_spectrum(t: TensorHandle, seed: int = 0) -> np.ndarray:
    """Return the (truncated) singular values of a tensor, computed once.

    1-D tensors return a single-element array with the L2 norm (consistent
    with the historical spectral-norm convention for vectors).

    The LAPACK part (exact/randomized SVD, QR) is serialized behind
    ``_spectrum_lock`` because concurrent ``np.linalg.svd``/``qr`` calls from
    several threads can deadlock inside OpenBLAS. The tensor payload is loaded
    *before* the lock is acquired, so dequantization still runs in parallel.
    """
    cached = _cache.get(t)
    if cached is not None:
        return cached
    x = t.load()
    if x.ndim == 1:
        s = np.array([float(np.linalg.norm(x.astype(np.float64)))])
        with _spectrum_lock:
            cached = _cache.get(t)  # re-check under the lock
            if cached is None:
                _cache[t] = s
            return cached if cached is not None else s
    m = to_matrix(x)
    if min(m.shape) <= SMALL:
        with _spectrum_lock:
            cached = _cache.get(t)  # re-check under the lock
            if cached is not None:
                return cached
            s = np.linalg.svd(m.astype(np.float64), compute_uv=False)
            _cache[t] = s
            return s
    # rSVD path: power iterations are plain GEMMs (thread-safe, run unlocked
    # inside _randomized_singular_values); only qr/svd serialize.
    s = _randomized_singular_values(m, seed=seed)
    with _spectrum_lock:
        cached = _cache.get(t)
        if cached is not None:
            return cached
        _cache[t] = s
        return s


def _randomized_singular_values(m: np.ndarray, seed: int) -> np.ndarray:
    """Randomized truncated SVD (Halko) in float32, seeded for determinism.

    Serializes its LAPACK calls on ``_spectrum_lock``; the power-iteration
    GEMMs run unlocked via :func:`_randomized_range`.
    """
    y = _randomized_range(m, seed)
    with _spectrum_lock:
        q, _ = np.linalg.qr(y)
        b = q.T @ m
        return np.linalg.svd(b.astype(np.float64), compute_uv=False)


def spectrum_of_matrix(m: np.ndarray, seed: int = 0) -> np.ndarray:
    """Truncated singular values of a raw 2-D matrix (no handle/caching).

    Array-based twin of ``truncated_spectrum`` for derived payloads such as
    the edit-delta ``B - A`` in the paired pipeline, which are never
    ``TensorHandle``s. Same dispatch and lock as ``truncated_spectrum``:
    exact SVD for ``min(m, n) <= SMALL``, else the seeded Halko rSVD.
    """
    x = to_matrix(np.asarray(m))
    if min(x.shape) <= SMALL:
        with _spectrum_lock:
            return np.linalg.svd(x.astype(np.float64), compute_uv=False)
    y = _randomized_range(x, seed)
    with _spectrum_lock:
        q, _ = np.linalg.qr(y)
        b = q.T @ x
        return np.linalg.svd(b.astype(np.float64), compute_uv=False)


def top_left_singular_vector(m: np.ndarray, seed: int = 0) -> np.ndarray:
    """Top left singular vector u1 of a raw 2-D matrix (sign-fixed).

    Same dispatch as ``spectrum_of_matrix``; the returned vector is
    normalized and sign-fixed so the largest-|component| entry is positive
    (the same convention as ``embedding/pca.py``), making pairwise
    u1-coherence comparisons meaningful. Used by the edit preset's
    ``u1_coherence`` metric (opt-in).
    """
    x = to_matrix(np.asarray(m))
    if min(x.shape) <= SMALL:
        with _spectrum_lock:
            u, _, _ = np.linalg.svd(x.astype(np.float64), full_matrices=False)
            return _fix_sign(u[:, 0].copy())
    y = _randomized_range(x, seed)
    with _spectrum_lock:
        q, _ = np.linalg.qr(y)
        b = q.T @ x
        u_small, _, _ = np.linalg.svd(b.astype(np.float64), full_matrices=False)
        u1 = (q.astype(np.float64)) @ u_small[:, 0]
        n = float(np.linalg.norm(u1))
        if n > 0:
            u1 = u1 / n
        return _fix_sign(u1)


def _fix_sign(u1: np.ndarray) -> np.ndarray:
    """Sign-fix so the largest-|component| entry is positive (pca.py convention)."""
    idx = int(np.argmax(np.abs(u1)))
    if u1[idx] < 0:
        return -u1
    return u1


def entropy_rank(s: np.ndarray) -> float:
    """Effective rank = exp(-sum(p_i * log p_i)) where p = s / sum(s).

    Clip near-zero singular values to avoid log(0). Using a small floor keeps
    the rank estimate stable without biasing large values.
    """
    s = np.clip(s, 1e-12, None)
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def clear_cache_for(t: TensorHandle) -> None:
    """Drop the cached spectrum for one handle (used by tests)."""
    _cache.pop(t, None)


def cache_size() -> int:
    """Number of handles with a cached spectrum (used by tests)."""
    return len(_cache)
