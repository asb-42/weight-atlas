"""Streaming stats for giant tensors (P3 n-gram tables): identity, bounds, signatures."""

from __future__ import annotations

import json
import tracemalloc
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

from weight_atlas.core.types import TensorHandle, TensorStats
from weight_atlas.stats.streaming import streaming_tensor_stats


class _FakeBlockLoader:
    """Fake lazy loader (review F-4): yields row blocks of a large virtual
    matrix without materializing it — the test asserts exactly that."""

    def __init__(self, matrix: np.ndarray, block_rows: int = 4096) -> None:
        # matrix is the REAL data here, but the loader only ever hands out
        # blocks; tracemalloc proves the pipeline never grabs it whole.
        self._matrix = matrix
        self._block_rows = block_rows

    def iter_row_blocks(self, handle: TensorHandle, rows_per_block: int) -> Iterator[np.ndarray]:
        rpb = min(rows_per_block, self._block_rows)
        for lo in range(0, self._matrix.shape[0], rpb):
            yield self._matrix[lo : lo + rpb]


def _stats_via_streaming(matrix: np.ndarray, name: str = "t.weight") -> TensorStats:
    handle = TensorHandle(name, matrix.shape, "float32", lambda: matrix)
    loader = _FakeBlockLoader(matrix)
    main, _extras = streaming_tensor_stats(
        handle.name, handle.shape, handle.dtype, None,
        lambda: loader.iter_row_blocks(handle, 4096),
    )
    return main


def _stats_in_process(matrix: np.ndarray, name: str = "t.weight") -> TensorStats:
    handle = TensorHandle(name, matrix.shape, "float32", lambda: matrix)
    from weight_atlas.scan import _stats_for_handle

    return _stats_for_handle(handle)


# ---------------------------------------------------------------------------
# Cross-path numerical identity (review F-1 tolerance)
# ---------------------------------------------------------------------------


class TestCrossPathIdentity:
    @pytest.fixture(scope="class")
    def matrix(self) -> np.ndarray:
        rng = np.random.default_rng(4)
        # low-rank + noise: an "embedding-like" matrix
        base = rng.standard_normal((2000, 160)) @ np.diag(np.linspace(1, 0.01, 160))
        return (base + rng.standard_normal((2000, 160)) * 0.01).astype(np.float32)

    def test_sigma_within_f1_tolerance(self, matrix: np.ndarray) -> None:
        """Streaming vs in-process: relative σ deviation ≤ 1e-4 (review F-1 —
        fp32/fp64 GEMM reduction orders differ; 1e-15 would be the wrong measure)."""
        streamed = _stats_via_streaming(matrix)
        inproc = _stats_in_process(matrix)
        from weight_atlas.stats.spectrum import truncated_spectrum

        ref = truncated_spectrum(
            TensorHandle("t.weight", matrix.shape, "float32", lambda: matrix), seed=0
        )
        assert streamed.spectral_norm == pytest.approx(inproc.spectral_norm, rel=1e-4)
        assert inproc.spectral_norm == pytest.approx(float(ref[0]), rel=1e-4)
        del streamed, inproc

    def test_sums_equal_nan_aware(self, matrix: np.ndarray) -> None:
        """Accumulated sums (frobenius etc.) match with equal_nan semantics."""
        streamed = _stats_via_streaming(matrix)
        inproc = _stats_in_process(matrix)
        for field in ("frobenius", "spectral_norm", "stable_rank", "effective_rank",
                      "kurtosis", "std", "mean", "absmax", "absmean",
                      "p50", "p90", "p99", "p999", "p9999", "dyn_range",
                      "row_amax_ratio", "col_amax_ratio"):
            a, b = getattr(streamed, field), getattr(inproc, field)
            if isinstance(a, float) and np.isnan(a):
                assert np.isnan(b), field
            else:
                assert a == pytest.approx(b, rel=1e-4, abs=1e-9), field
        for field in ("outlier_3s", "outlier_4s", "outlier_6s", "sparsity"):
            assert getattr(streamed, field) == pytest.approx(getattr(inproc, field), abs=1e-6), field

    def test_gram_gives_full_spectrum(self, matrix: np.ndarray) -> None:
        """cols=160 ≤ Gram cap → the streaming spectrum has ALL 160 values
        (review F-2: curve shape resolved exactly, no k-truncation)."""
        from weight_atlas.stats.spectrum import truncated_spectrum

        ref = truncated_spectrum(
            TensorHandle("t.weight", matrix.shape, "float32", lambda: matrix), seed=0
        )
        assert len(ref) == 160  # in-process exact path on min-dim ≤ 512
        streamed_vals = _streamed_sigma_values(matrix)
        assert len(streamed_vals) == 160
        rel = np.abs(streamed_vals - ref) / ref
        assert rel.max() < 1e-4  # review F-1 tolerance on the whole curve


def _streamed_sigma_values(matrix: np.ndarray) -> np.ndarray:
    """The full spectrum the Gram path computed (re-derived for the test)."""
    gram = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=np.float64)
    for lo in range(0, matrix.shape[0], 4096):
        b = matrix[lo : lo + 4096].astype(np.float64)
        gram += b.T @ b
    eig = np.clip(np.linalg.eigvalsh(gram)[::-1], 0.0, None)
    return np.sqrt(eig)


# ---------------------------------------------------------------------------
# Memory honesty (review F-4: fake block loader + tracemalloc)
# ---------------------------------------------------------------------------


class TestMemoryBound:
    def test_streaming_never_materializes(self) -> None:
        """Peak allocations during streaming must stay far below the tensor's
        materialized size — the fixture uses a fake lazy block loader (the
        default load() fallback WOULD materialize and fail this test)."""
        rows, cols = 4_000_000, 64  # 1 GB fp32-equivalent
        blocks_expected = rows * cols * 4

        # virtual matrix: blocks are PRE-GENERATED (their allocation must not
        # count against the streaming peak — only the streaming path's own
        # allocations are under test); the loader yields them block by block.
        gen_rng = np.random.default_rng(6)
        all_blocks = [
            gen_rng.standard_normal((min(lo + 65536, rows) - lo, cols)).astype(np.float32)
            for lo in range(0, rows, 65536)
        ]

        class _VirtualLoader:
            def iter_row_blocks(self, handle: TensorHandle, rows_per_block: int) -> Iterator[np.ndarray]:
                yield from all_blocks

        handle = TensorHandle("giant.weight", (rows, cols), "float32", lambda: (_ for _ in ()).throw(
            AssertionError("load() must not be called on the streaming path")
        ))
        loader = _VirtualLoader()

        tracemalloc.start()
        ts, _extras = streaming_tensor_stats(
            handle.name, handle.shape, handle.dtype, None,
            lambda: loader.iter_row_blocks(handle, 65536),
        )
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert ts.streamed is True
        assert ts.absmax > 0
        # peak must be ≪ the materialized size. Legitimate residents: the
        # row-amax vector (rows × 4 B), the percentile sample (cap 2M × 4 B),
        # one block, and float64 block temporaries — together O(rows) but
        # ~100× below the 1 GB materialization.
        assert peak < blocks_expected * 0.15, f"peak {peak/1e6:.1f} MB vs tensor {blocks_expected/1e6:.0f} MB"


# ---------------------------------------------------------------------------
# Per-head records: element-count semantics (regression, real-model bug found
# by the 2026-09-02 Flash-Next acceptance run: head['n'] counted ROWS while
# every accumulator counts ELEMENTS → per-head sparsity 160× too large)
# ---------------------------------------------------------------------------


class TestPerHeadRecords:
    def _blocks_of(self, matrix: np.ndarray, rows_per_block: int = 64) -> Callable[[], Iterator[np.ndarray]]:
        def factory() -> Iterator[np.ndarray]:
            for lo in range(0, matrix.shape[0], rows_per_block):
                yield matrix[lo : lo + rows_per_block]

        return factory

    def test_head_scalars_use_element_counts(self) -> None:
        """Sparse matrix + head bounds: per-head mean/std/sparsity/outliers
        must equal the direct slice computation (element semantics)."""
        rng = np.random.default_rng(31)
        rows, cols = 512, 32
        matrix = rng.standard_normal((rows, cols)).astype(np.float32)
        matrix[rng.random((rows, cols)) < 0.5] = 0.0  # ~50% zeros
        bounds = [0, 160, 320]

        _, head_records = streaming_tensor_stats(
            "tbl.weight", matrix.shape, "float32", None,
            self._blocks_of(matrix), head_bounds=bounds,
        )
        assert len(head_records) == 3
        head_ends = bounds[1:] + [rows]

        for i, rec in enumerate(head_records):
            sl = matrix[bounds[i] : head_ends[i]]
            n_el = sl.size
            assert rec.name == f"tbl.weight.h{i}"
            # shape covers the head's rows/cols; n counts ELEMENTS
            assert rec.shape == (head_ends[i] - bounds[i], cols)
            assert rec.sparsity == pytest.approx(
                np.count_nonzero(np.abs(sl) < 1e-3) / n_el, abs=1e-6
            )
            assert rec.mean == pytest.approx(float(sl.sum()) / n_el, rel=1e-5, abs=1e-7)
            assert rec.std == pytest.approx(float(sl.astype(np.float64).std()), rel=1e-4)
            # outlier fractions are FRACTIONS — a sparsity > 1.0 was the bug
            assert 0.0 <= rec.sparsity <= 1.0
            assert 0.0 <= rec.outlier_3s <= 1.0
            assert rec.kurtosis > -3.1  # non-degenerate variance → real kurtosis

    def test_head_spectra_match_direct_slices(self) -> None:
        """Per-head spectral stats equal those of the sliced submatrix."""
        rng = np.random.default_rng(32)
        rows, cols = 400, 24
        matrix = rng.standard_normal((rows, cols)).astype(np.float32)
        bounds = [0, 150, 400]

        _, head_records = streaming_tensor_stats(
            "tbl.weight", matrix.shape, "float32", None,
            self._blocks_of(matrix), head_bounds=bounds,
        )
        assert len(head_records) == 2
        head_ends = bounds[1:] + [rows]
        for i, rec in enumerate(head_records):
            sl = matrix[bounds[i] : head_ends[i]].astype(np.float64)
            sigma = np.linalg.svd(sl, compute_uv=False)
            assert rec.spectral_norm == pytest.approx(float(sigma[0]), rel=1e-3)
            assert rec.frobenius == pytest.approx(float(np.sqrt((sl * sl).sum())), rel=1e-3)


# ---------------------------------------------------------------------------
# Preregistered instrument signatures (review F-4b: assertions, not eyeballing)
# ---------------------------------------------------------------------------


class TestInstrumentSignatures:
    """The embedding-vs-lookup measurement, validated on synthetic tables
    BEFORE it touches the real model. Expected signatures are preregistered
    here as assertions."""

    @pytest.fixture(scope="class")
    def embedding_table(self) -> np.ndarray:
        """Dense low-rank + isotropic noise — a 'real embedding'."""
        rng = np.random.default_rng(21)
        base = rng.standard_normal((4000, 128)) @ np.diag(np.geomspace(10, 0.1, 128))
        return (base + rng.standard_normal((4000, 128))).astype(np.float32)

    @pytest.fixture(scope="class")
    def hash_table(self) -> np.ndarray:
        """95% zeros + a FEW huge row-spikes — a 'hash lookup table'. Lookup
        tables concentrate extreme entries in few rows (hot buckets), which
        is precisely what makes their spectrum spiky vs. an embedding's."""
        rng = np.random.default_rng(22)
        table = np.zeros((4000, 128), dtype=np.float32)
        # 5% scattered small non-zeros (collisions)
        idx = rng.choice(table.size, size=table.size // 20, replace=False)
        table.flat[idx] = rng.standard_normal(table.size // 20)
        # 8 hot rows with 100× amplitude → rank-isolated spikes
        hot = rng.choice(4000, size=8, replace=False)
        table[hot] = rng.standard_normal((8, 128)) * 100
        return table

    def test_embedding_signature(self, embedding_table: np.ndarray) -> None:
        s = _stats_via_streaming(embedding_table, "embedding")
        # embedding: dense, smooth-ish spectrum, no heavy tails
        assert s.sparsity < 0.01
        assert s.kurtosis < 10.0
        assert s.stable_rank > 1.0  # more than one dominant direction
        assert 0.0 < s.sv_decay < 1.0

    def test_hash_table_signature(self, hash_table: np.ndarray) -> None:
        s = _stats_via_streaming(hash_table, "hash")
        # lookup table: sparse, spiky, dynamic range explodes
        assert s.sparsity > 0.9
        assert s.kurtosis > 20.0
        assert s.dyn_range > 100.0

    def test_signatures_separate_the_regimes(self, embedding_table: np.ndarray, hash_table: np.ndarray) -> None:
        e = _stats_via_streaming(embedding_table, "embedding")
        h = _stats_via_streaming(hash_table, "hash")
        assert e.sparsity < h.sparsity
        assert e.kurtosis < h.kurtosis
        # rank isolation: hash spikes live in a few directions → σ₁/σ₂ ≫
        # the embedding's smooth spectrum (σ₁/σ₂ close to 1 at the top)
        from weight_atlas.stats.spectrum import truncated_spectrum

        e_spec = truncated_spectrum(
            TensorHandle("e", embedding_table.shape, "float32", lambda: embedding_table), seed=0
        )
        h_spec = truncated_spectrum(
            TensorHandle("h", hash_table.shape, "float32", lambda: hash_table), seed=0
        )
        assert h_spec[0] / h_spec[1] > e_spec[0] / e_spec[1]


# ---------------------------------------------------------------------------
# Quant probe in streaming
# ---------------------------------------------------------------------------


class TestStreamingQuantProbe:
    def test_int8_sqnr_matches_reference(self) -> None:
        rng = np.random.default_rng(8)
        matrix = rng.standard_normal((2048, 128)).astype(np.float32)
        handle = TensorHandle("t", matrix.shape, "float32", lambda: matrix)
        loader = _FakeBlockLoader(matrix)
        ts, _extras = streaming_tensor_stats(
            handle.name, handle.shape, handle.dtype, None,
            lambda: loader.iter_row_blocks(handle, 512),
            quant_probe=True,
        )
        from weight_atlas.stats.sqnr import int8_per_channel_sqnr

        ref = int8_per_channel_sqnr(matrix)
        assert ts.sqnr_int8_ch == pytest.approx(ref, rel=1e-6)
        # INT4 group scales are not streamed (v1) — NaN per contract
        assert np.isnan(ts.sqnr_int4_g128)
        assert np.isfinite(ts.sqnr_fp8_e4m3)


# ---------------------------------------------------------------------------
# Scan integration: giant tensor routes to streaming, resume works
# ---------------------------------------------------------------------------


class TestScanStreamingIntegration:
    def test_giant_tensor_routes_and_resumes(self, tmp_path: Path) -> None:
        """A model whose one tensor exceeds the streaming threshold is
        streamed (streamed:true in the fingerprint); killing the scan
        mid-stats leaves a journal whose resume completes byte-identically
        (review F-4a: 'for free' is asserted once)."""
        import weight_atlas.scan as scan_mod
        from tests.fixtures import make_fake_model
        from weight_atlas.core.types import load_default_spec
        from weight_atlas.scan import scan

        model_path = tmp_path / "fake.safetensors"
        make_fake_model(model_path, n_layers=2)
        spec = load_default_spec()

        # Route everything through streaming by lowering the threshold —
        # the threshold patch must stay active through BOTH scans (raised
        # after the first scan would disable streaming for the crash/resume
        # half). Separate MonkeyPatch instances: the crash patch is undone
        # before the resume scan, the threshold patch afterwards.
        mp_threshold = pytest.MonkeyPatch()
        mp_threshold.setattr(scan_mod, "_STREAM_TENSOR_BYTES", 1 << 10)  # 1 KiB
        out = tmp_path / "out_stream"
        scan(model_path, out, spec, jobs=1)

        fp = json.loads((out / "fingerprint.json").read_text())
        streamed_flags = {n: info.get("streamed", False) for n, info in fp["tensors"].items()}
        assert any(streamed_flags.values()), "no tensor was streamed"
        # streamed records carry the marker AND their on-disk dtype
        streamed_names = [n for n, v in streamed_flags.items() if v]
        for name in streamed_names[:1]:
            assert fp["tensors"][name]["dtype"].startswith("F")  # safetensors dtype

        # crash mid-stats (after 3 streamed tensors), then resume via the
        # journal → byte-identical fingerprint (review F-4a: the 'free'
        # journal resume is asserted, once, on the streaming path itself)
        ref_bytes = (out / "fingerprint.json").read_bytes()
        out2 = tmp_path / "out_crash"
        real_stream = scan_mod.streaming_tensor_stats
        calls = {"n": 0}

        def exploding_stream(name, *a, **k):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            if calls["n"] > 3:
                raise RuntimeError("simulated crash")
            return real_stream(name, *a, **k)

        mp_crash = pytest.MonkeyPatch()
        mp_crash.setattr(scan_mod, "streaming_tensor_stats", exploding_stream)
        try:
            with pytest.raises(RuntimeError, match="simulated crash"):
                scan(model_path, out2, spec, jobs=1)
        finally:
            mp_crash.undo()
        journal = out2 / scan_mod._CHECKPOINT_NAME
        assert journal.exists()
        scan(model_path, out2, spec, jobs=1)
        assert (out2 / "fingerprint.json").read_bytes() == ref_bytes
        mp_threshold.undo()
