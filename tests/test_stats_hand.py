"""Hand-computed statistic tests against known matrices."""

from __future__ import annotations

import numpy as np
import pytest

from weight_atlas.core.types import TensorHandle
from weight_atlas.stats.norms import EffectiveRank, FrobeniusNorm, SpectralNorm
from weight_atlas.stats.shape_moments import Kurtosis, Sparsity
from weight_atlas.stats.stable_rank import StableRank


def _handle(arr: np.ndarray) -> TensorHandle:
    return TensorHandle(name="t", shape=arr.shape, dtype=str(arr.dtype), loader=lambda: arr)


def test_spectral_diag_3_4():
    """diag(3, 4) → spectral norm = 4 (largest singular value)."""
    h = _handle(np.diag([3.0, 4.0]).astype(np.float32))
    assert pytest.approx(SpectralNorm().compute(h), rel=1e-5) == 4.0


def test_effective_rank_identity_8():
    """I₈ → effective rank = 8 (all singular values equal)."""
    h = _handle(np.eye(8, dtype=np.float32))
    assert pytest.approx(EffectiveRank().compute(h), rel=1e-5) == 8.0


def test_frobenius_known():
    """diag(3,4) → frobenius = 5."""
    h = _handle(np.diag([3.0, 4.0]).astype(np.float32))
    assert pytest.approx(FrobeniusNorm().compute(h), rel=1e-5) == 5.0


def test_kurtosis_normal():
    """Normal-distributed values have excess kurtosis ≈ 0."""
    rng = np.random.default_rng(0)
    arr = rng.normal(0, 1, 10000).astype(np.float32)
    h = _handle(arr)
    assert pytest.approx(Kurtosis().compute(h), abs=0.3) == 0.0


def test_sparsity_known():
    """3 of 6 values below eps (0.0, 0.0005, 0.0001)."""
    arr = np.array([0.0, 0.0005, 1.0, 2.0, 0.0001, 3.0], dtype=np.float32)
    h = _handle(arr)
    assert pytest.approx(Sparsity().compute(h), rel=1e-6) == 3 / 6


def test_1d_spectral_is_l2():
    arr = np.array([3.0, 4.0], dtype=np.float32)
    h = _handle(arr)
    assert pytest.approx(SpectralNorm().compute(h), rel=1e-5) == 5.0


def test_1d_effective_rank_is_one():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    h = _handle(arr)
    assert pytest.approx(EffectiveRank().compute(h), rel=1e-5) == 1.0


def test_frobenius_chunked_matches_naive():
    """Chunked accumulation must match a naive float64 computation."""
    rng = np.random.default_rng(123)
    arr = rng.normal(0, 1, 5_000_000).astype(np.float32)
    h = _handle(arr)
    got = FrobeniusNorm().compute(h)
    expected = float(np.sqrt(np.astype(arr, np.float64).dot(np.astype(arr, np.float64))))
    assert pytest.approx(got, rel=1e-10) == expected


def test_1d_stable_rank_is_one():
    """A vector is rank-1: raw stable rank is exactly 1.0 — NOT the
    log1p degenerate ln(2) (flagged by Quinn 2026-09-02 from the
    Flash-Next scan: every 1-D norm vector reported sr = 0.6931…)."""
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    h = _handle(arr)
    assert StableRank().compute(h) == 1.0


def test_stable_rank_matrix_matches_formula():
    """2-D tensors keep the log1p((frob/spec)^2) definition."""
    rng = np.random.default_rng(7)
    arr = rng.normal(0, 1, (16, 8)).astype(np.float32)
    h = _handle(arr)
    sigma = np.linalg.svd(arr, compute_uv=False)
    expect = float(np.log1p((np.linalg.norm(arr) / sigma[0]) ** 2))
    assert pytest.approx(StableRank().compute(h), rel=1e-5) == expect


def test_scalar_tensor_stats_do_not_crash():
    """0-D tensors (real models store scalars: global scales, temperature)
    must scan — to_matrix used to raise IndexError on shape[0] of ()."""
    import numpy as np

    from weight_atlas.stats.norms import EffectiveRank, FrobeniusNorm, SpectralNorm
    from weight_atlas.stats.stable_rank import StableRank

    arr = np.array(3.0, dtype=np.float32)  # shape ()
    h = _handle(arr)
    assert SpectralNorm().compute(h) == pytest.approx(3.0)
    assert FrobeniusNorm().compute(h) == pytest.approx(3.0)
    assert StableRank().compute(h) == 1.0  # rank-1 object, same as 1-D
    assert EffectiveRank().compute(h) == pytest.approx(1.0)


def test_scalar_zero_tensor():
    """The all-zero scalar degenerates to 0 spectral norm without crashing."""
    arr = np.array(0.0, dtype=np.float32)
    h = _handle(arr)
    assert SpectralNorm().compute(h) == 0.0
    assert StableRank().compute(h) == 0.0
