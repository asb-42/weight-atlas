"""Process-pool stats phase (P2 perf): determinism across paths + fallbacks."""

from __future__ import annotations

import json
from concurrent.futures import Future

import numpy as np
import pytest

from weight_atlas.core.types import TensorHandle
from weight_atlas.scan import (
    _PROCESS_POOL_MIN_TENSORS,
    _resolve_jobs,
    _run_stats_processes,
    _worker_init,
)


def _rng() -> np.random.Generator:
    return np.random.default_rng(11)


def _fake_handles(n: int, seed: int = 11) -> list[TensorHandle]:
    rng = np.random.default_rng(seed)
    handles = []
    for i in range(n):
        rows = int(rng.integers(8, 64))
        cols = int(rng.integers(8, 64))
        x = rng.standard_normal((rows, cols)).astype(np.float32)
        handles.append(TensorHandle(f"t{i:04d}.weight", x.shape, "float32", lambda x=x: x))
    return handles


def _serial_stats(handles: list[TensorHandle]) -> list:
    from weight_atlas.scan import _stats_for_handle

    return [_stats_for_handle(h) for h in handles]


class TestResolveJobs:
    def test_leaves_two_core_reserve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import weight_atlas.scan as scan_mod

        monkeypatch.setattr(scan_mod.os, "cpu_count", lambda: 20)
        assert scan_mod._resolve_jobs(None) == 18

    def test_small_machine_floors_at_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import weight_atlas.scan as scan_mod

        monkeypatch.setattr(scan_mod.os, "cpu_count", lambda: 3)
        assert scan_mod._resolve_jobs(None) == 1

    def test_explicit_jobs_wins(self) -> None:
        assert _resolve_jobs(7) == 7


class TestProcessPoolDeterminism:
    def test_pool_results_match_serial(self) -> None:
        """The process pool must produce byte-identical stats to the serial
        path (same BLAS=1 numeric path in workers via _worker_init)."""
        handles = _fake_handles(96)
        expected = _serial_stats(handles)

        stats: list = [None] * len(handles)
        _run_stats_processes(
            list(range(len(handles))), 4, handles, stats,
            svd_seed=0, distribution_seed=0, quant_probe=False,
            report_stats=lambda i: None, record=lambda ts: None,
        )
        assert all(s is not None for s in stats)
        for got, exp in zip(stats, expected, strict=True):
            for field in ("frobenius", "spectral_norm", "effective_rank", "stable_rank",
                          "kurtosis", "sparsity", "kernel_norm", "sv_decay",
                          "row_amax_ratio", "col_amax_ratio", "p99", "outlier_3s",
                          "dyn_range"):
                g, e = getattr(got, field), getattr(exp, field)
                if isinstance(e, float) and np.isnan(e):
                    assert np.isnan(g)
                else:
                    assert g == e, f"{field}: {g} != {e}"


class _RaisingFor:
    """Inline ProcessPoolExecutor stub: runs tasks in the calling thread,
    raising for the poisoned indices (simulates worker failures). Accepts
    the real executor's constructor signature."""

    def __init__(self, *a: object, **k: object) -> None:
        self._poisoned = getattr(self, "_poisoned", set())

    def __enter__(self) -> _RaisingFor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def submit(self, fn, task):  # type: ignore[no-untyped-def]
        fut: Future = Future()
        try:
            fut.set_result(fn(task))
        except Exception as exc:  # noqa: BLE001
            fut.set_exception(exc)
        return fut


class TestPerTensorFallback:
    def test_failed_task_recomputed_serially(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A worker-side failure must heal via the serial recompute — the scan
        never loses a tensor (infra errors), while real data errors surface
        through the same recompute."""
        import weight_atlas.scan as scan_mod

        handles = _fake_handles(12)
        expected = _serial_stats(handles)
        poisoned = {3, 7}

        real_worker = scan_mod._worker_stats

        def flaky_worker(task):
            if task[0] in poisoned:
                raise RuntimeError("simulated worker crash")
            return real_worker(task)

        monkeypatch.setattr(scan_mod, "_worker_stats", flaky_worker)
        monkeypatch.setattr(scan_mod, "ProcessPoolExecutor", _RaisingFor)

        stats: list = [None] * len(handles)
        _run_stats_processes(
            list(range(len(handles))), 2, handles, stats,
            svd_seed=0, distribution_seed=0, quant_probe=False,
            report_stats=lambda i: None, record=lambda ts: None,
        )
        assert all(s is not None for s in stats)
        for i, (got, exp) in enumerate(zip(stats, expected, strict=True)):
            assert got.frobenius == exp.frobenius, f"tensor {i} diverged"

    def test_cleared_handles_reload_on_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The main process clears handles after submission; the fallback
        recompute must reload the tensor (lazy loader, not the cached array)."""
        import weight_atlas.scan as scan_mod

        loads = {"count": 0}
        x = np.ones((8, 8), dtype=np.float32)

        def loader() -> np.ndarray:
            loads["count"] += 1
            return x

        handles = [TensorHandle("t.weight", (8, 8), "float32", loader)]

        def flaky_worker(task):
            raise RuntimeError("worker died before computing")

        monkeypatch.setattr(scan_mod, "_worker_stats", flaky_worker)
        monkeypatch.setattr(scan_mod, "ProcessPoolExecutor", _RaisingFor)

        stats: list = [None]
        _run_stats_processes(
            [0], 1, handles, stats,
            svd_seed=0, distribution_seed=0, quant_probe=False,
            report_stats=lambda i: None, record=lambda ts: None,
        )
        assert stats[0] is not None
        assert loads["count"] >= 2  # submitted once, recomputed once
        assert stats[0].frobenius == pytest.approx(float(np.sqrt(64.0)))


class TestWorkerInit:
    def test_init_runs_without_threadpoolctl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_worker_init must not explode when threadpoolctl is missing."""
        import builtins

        real_import = builtins.__import__

        def no_threadpoolctl(name: str, *a, **k):  # type: ignore[no-untyped-def]
            if name == "threadpoolctl":
                raise ImportError("blocked for test")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_threadpoolctl)
        _worker_init()  # must not raise


@pytest.mark.slow
class TestScanIntegration:
    def test_cross_path_fingerprint_identical(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Serial (jobs=1) vs process-pool (jobs=4) scans produce byte-identical
        fingerprints — the determinism contract across execution paths."""
        from tests.fixtures import make_fake_model
        from weight_atlas.core.types import load_default_spec
        from weight_atlas.scan import scan

        model_path = tmp_path / "fake.safetensors"
        tensors = make_fake_model(model_path, n_layers=12)
        n_tensors = len(tensors)
        assert n_tensors >= _PROCESS_POOL_MIN_TENSORS

        spec = load_default_spec()
        out_serial = tmp_path / "serial"
        out_pool = tmp_path / "pool"
        scan(model_path, out_serial, spec, jobs=1)
        scan(model_path, out_pool, spec, jobs=4)
        fp_serial = (out_serial / "fingerprint.json").read_bytes()
        fp_pool = (out_pool / "fingerprint.json").read_bytes()
        assert fp_serial == fp_pool
        assert len(json.loads(fp_serial)["tensors"]) == n_tensors
