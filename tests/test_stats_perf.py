"""Shared-spectrum, chunked-stat and parallel-scan regression tests."""

from __future__ import annotations

import numpy as np
import pytest

from weight_atlas.core.types import TensorHandle, load_default_spec
from weight_atlas.stats import spectrum
from weight_atlas.stats.norms import EffectiveRank, FrobeniusNorm, SpectralNorm
from weight_atlas.stats.shape_moments import Kurtosis, Sparsity
from weight_atlas.stats.stable_rank import StableRank


def _handle(arr: np.ndarray) -> TensorHandle:
    return TensorHandle(name="t", shape=arr.shape, dtype="F32", loader=lambda: arr)


class TestSharedSpectrum:
    def test_svd_computed_once_per_tensor(self, monkeypatch):
        """Spectral + effective + stable rank share ONE randomized SVD."""
        calls = {"n": 0}

        def counting(m: np.ndarray, seed: int) -> np.ndarray:
            calls["n"] += 1
            return np.linalg.svd(m.astype(np.float64), compute_uv=False)

        monkeypatch.setattr(spectrum, "_randomized_singular_values", counting)
        arr = np.random.default_rng(0).normal(0, 1, (600, 600)).astype(np.float32)
        h = _handle(arr)
        SpectralNorm(seed=0).compute(h)
        EffectiveRank(seed=0).compute(h)
        StableRank(seed=0).compute(h)
        assert calls["n"] == 1

    def test_spectral_is_s0(self):
        arr = np.random.default_rng(1).normal(0, 1, (600, 700)).astype(np.float32)
        h = _handle(arr)
        assert SpectralNorm(seed=0).compute(h) == pytest.approx(spectrum.truncated_spectrum(h)[0])

    def test_effective_rank_from_shared_spectrum(self):
        arr = np.random.default_rng(2).normal(0, 1, (600, 700)).astype(np.float32)
        h = _handle(arr)
        s = spectrum.truncated_spectrum(h)
        assert EffectiveRank(seed=0).compute(h) == pytest.approx(spectrum.entropy_rank(s))

    def test_small_matrix_exact_svd(self):
        """Small matrices still use the exact SVD path."""
        arr = np.random.default_rng(3).normal(0, 1, (128, 128)).astype(np.float32)
        h = _handle(arr)
        expected = float(np.linalg.svd(arr.astype(np.float64), compute_uv=False)[0])
        assert SpectralNorm(seed=0).compute(h) == pytest.approx(expected, rel=1e-10)


class TestChunkedStats:
    def test_kurtosis_matches_vectorized(self):
        rng = np.random.default_rng(4)
        arr = rng.normal(0, 1, 3_000_000).astype(np.float32)
        h = _handle(arr)
        x = arr.astype(np.float64)
        mean = x.mean()
        m2 = np.mean((x - mean) ** 2)
        m4 = np.mean((x - mean) ** 4)
        expected = m4 / (m2**2) - 3.0
        assert Kurtosis().compute(h) == pytest.approx(expected, rel=1e-6)

    def test_kurtosis_constant(self):
        h = _handle(np.full(100, 5.0, dtype=np.float32))
        assert Kurtosis().compute(h) == -3.0

    def test_sparsity_matches_vectorized(self):
        rng = np.random.default_rng(5)
        arr = rng.normal(0, 0.0005, 2_000_000).astype(np.float32)
        h = _handle(arr)
        expected = float(np.mean(np.abs(arr) < 1e-3))
        assert Sparsity().compute(h) == pytest.approx(expected, rel=1e-9)

    def test_frobenius_matches_naive(self):
        rng = np.random.default_rng(6)
        arr = rng.normal(0, 1, 3_000_000).astype(np.float32)
        h = _handle(arr)
        expected = float(np.sqrt(np.sum(arr.astype(np.float64) ** 2)))
        assert FrobeniusNorm().compute(h) == pytest.approx(expected, rel=1e-9)


class TestHandleClear:
    def test_clear_releases_and_reloads(self):
        arr = np.random.default_rng(7).normal(0, 1, (64, 64)).astype(np.float32)
        loads = {"n": 0}

        def loader() -> np.ndarray:
            loads["n"] += 1
            return arr

        h = TensorHandle(name="t", shape=arr.shape, dtype="F32", loader=loader)
        v1 = FrobeniusNorm().compute(h)
        h.load()  # memoized — no extra loader call
        assert loads["n"] == 1
        h.clear()
        v2 = FrobeniusNorm().compute(h)  # re-reads from the loader
        assert loads["n"] == 2
        assert v1 == v2

    def test_spectrum_cache_cleared_with_handle(self):
        h = _handle(np.random.default_rng(8).normal(0, 1, (600, 600)).astype(np.float32))
        spectrum.truncated_spectrum(h)
        assert spectrum.cache_size() == 1
        del h
        import gc
        gc.collect()
        assert spectrum.cache_size() == 0


class TestParallelScan:
    def test_jobs_deterministic_byte_identical(self, tmp_path):
        """jobs=4 must produce byte-identical artefacts to jobs=1."""
        from tests.fixtures import make_fake_model
        from weight_atlas.scan import scan

        model = tmp_path / "m.safetensors"
        make_fake_model(model, n_layers=6)
        spec = load_default_spec()

        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        scan(model, out1, spec, jobs=1)
        scan(model, out2, spec, jobs=4)

        for name in ("fingerprint.json", "manifest.json"):
            assert (out1 / name).read_bytes() == (out2 / name).read_bytes(), name
        tifs1 = sorted(out1.glob("field_*.tif"))
        tifs2 = sorted(out2.glob("field_*.tif"))
        assert [p.name for p in tifs1] == [p.name for p in tifs2]
        for p1, p2 in zip(tifs1, tifs2, strict=True):
            assert p1.read_bytes() == p2.read_bytes()

    def test_resolve_jobs_default_bounds(self):
        from weight_atlas.scan import _resolve_jobs

        assert _resolve_jobs(None) <= 8
        assert _resolve_jobs(None) >= 1
        assert _resolve_jobs(2) == 2
        assert _resolve_jobs(0) >= 1

    def test_parallel_scan_many_experts_completes(self, tmp_path):
        """Parallel scan over many small expert tensors must complete.

        Deadlock regression: small expert tensors hit the exact-SVD path
        (min dim <= SMALL), and concurrent ``np.linalg.svd`` calls from several
        worker threads used to deadlock inside OpenBLAS. The spectrum lock +
        single global BLAS cap must keep jobs>1 scans both safe and byte
        identical to a serial scan.
        """
        from tests.test_moe import make_gguf_moe_file
        from weight_atlas.scan import scan

        model = tmp_path / "moe.gguf"
        make_gguf_moe_file(model, n_layers=6, n_experts=8, shared=True)
        spec = load_default_spec()

        out_par = tmp_path / "out_par"
        scan(model, out_par, spec, jobs=4)  # must not deadlock
        assert (out_par / "fingerprint.json").exists()
        assert list(out_par.glob("field_expert_*_raw.tif"))

        # Parallel results are identical to serial (determinism guarantee).
        out_ser = tmp_path / "out_ser"
        scan(model, out_ser, spec, jobs=1)
        assert (out_par / "fingerprint.json").read_bytes() == (
            out_ser / "fingerprint.json"
        ).read_bytes()
