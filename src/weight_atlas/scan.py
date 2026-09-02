"""Scan pipeline: load → stats → fields → artefacts + manifest."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import multiprocessing
import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np

from weight_atlas.core.name_map import is_expert_tensor, is_shared_expert, map_name
from weight_atlas.core.registry import get_loader
from weight_atlas.core.types import AtlasSpec, TensorHandle, TensorStats, detect_loader
from weight_atlas.fields.rasterizer import (
    detect_moe,
    detect_vision,
    rasterize,
    rasterize_bdh_lattice,
    rasterize_expert_panels,
    rasterize_flat,
    rasterize_vision,
)
from weight_atlas.fields.scaling import apply_scale, log1p
from weight_atlas.fields.tif_io import write_tif
from weight_atlas.loaders import (
    gguf_loader,  # noqa: F401 — triggers registration
    pytorch_loader,  # noqa: F401 — triggers registration
    safetensors_loader,  # noqa: F401 — triggers registration
)
from weight_atlas.loaders.blocking import _ROWS_PER_BLOCK_DEFAULT, iter_row_blocks
from weight_atlas.stats.distribution import amax_ratios, distribution_summary
from weight_atlas.stats.norms import (
    EffectiveRank,
    FrobeniusNorm,
    KernelNorm,
    SpectralNorm,
    SVDecay,
)
from weight_atlas.stats.shape_moments import Kurtosis, Sparsity
from weight_atlas.stats.sqnr import SQNRFp8E4M3, SQNRInt4Group128, SQNRInt8PerChannel
from weight_atlas.stats.stable_rank import StableRank
from weight_atlas.stats.streaming import streaming_tensor_stats

logger = logging.getLogger(__name__)

# Tensors whose float32 materialization is >= this many bytes are computed
# serially (one at a time) during the stats phase to keep peak RAM bounded on
# models with very large tensors (see ``scan``).
_BIG_TENSOR_THRESHOLD_BYTES = 1 << 30  # 1 GiB
# Tensors whose float32 materialization would exceed this are computed via
# the block-streaming path (stats.streaming) — never fully materialized.
# 8 GiB fp32 ≈ 2G elements.
_STREAM_TENSOR_BYTES = 8 << 30


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_handles(
    tensor: TensorHandle,
    svd_seed: int = 0,
    distribution_seed: int = 0,
    quant_probe: bool = False,
) -> TensorStats:
    """Compute all registered statistics for one tensor.

    ``svd_seed`` comes from the spec's ``seeds.svd`` so scan spectra and
    paired-pipeline Δ-spectra share the same seeded rSVD contract;
    ``distribution_seed`` drives the deterministic percentile subsample above
    the 16M-element cap (stats.distribution). ``quant_probe`` enables the
    measured RTN-SQNR stats (stats.sqnr) — opt-in, ~6 extra weight passes.
    """
    row_ratio, col_ratio = amax_ratios(tensor)
    summary = distribution_summary(tensor, seed=distribution_seed)
    nans = float("nan")
    sqnr_int8 = SQNRInt8PerChannel().compute(tensor) if quant_probe else nans
    sqnr_int4 = SQNRInt4Group128().compute(tensor) if quant_probe else nans
    sqnr_fp8 = SQNRFp8E4M3().compute(tensor) if quant_probe else nans
    return TensorStats(
        name=tensor.name,
        shape=tensor.shape,
        frobenius=FrobeniusNorm().compute(tensor),
        spectral_norm=SpectralNorm(seed=svd_seed).compute(tensor),
        effective_rank=EffectiveRank(seed=svd_seed).compute(tensor),
        stable_rank=StableRank(seed=svd_seed).compute(tensor),
        kurtosis=Kurtosis().compute(tensor),
        sparsity=Sparsity().compute(tensor),
        kernel_norm=KernelNorm().compute(tensor),
        sv_decay=SVDecay(seed=svd_seed).compute(tensor),
        row_amax_ratio=row_ratio,
        col_amax_ratio=col_ratio,
        mean=summary["mean"],
        std=summary["std"],
        absmax=summary["absmax"],
        absmean=summary["absmean"],
        p50=summary["p50"],
        p90=summary["p90"],
        p99=summary["p99"],
        p999=summary["p999"],
        p9999=summary["p9999"],
        outlier_3s=summary["outlier_3s"],
        outlier_4s=summary["outlier_4s"],
        outlier_6s=summary["outlier_6s"],
        dyn_range=summary["dyn_range"],
        sqnr_int8_ch=sqnr_int8,
        sqnr_int4_g128=sqnr_int4,
        sqnr_fp8_e4m3=sqnr_fp8,
        expert_id=tensor.expert_id,
    )


def _resolve_jobs(jobs: int | None) -> int:
    """Resolve the stats worker count: explicit value, else ``cores - 2``.

    The scan machine is dedicated while scanning, so every core except a
    small reserve (2 for the OS / main process / BLAS overhead) should join
    the stats pool.
    """
    if jobs is not None and jobs > 0:
        return jobs
    return max(1, (os.cpu_count() or 8) - 2)


def _constant_blocks(arr: np.ndarray) -> Callable[[], Iterator[np.ndarray]]:
    """Restartable factory over one materialized array (streaming fallback)."""

    def factory() -> Iterator[np.ndarray]:
        return iter([arr])

    return factory


def _loader_blocks(
    handle: TensorHandle, loader: Any, rows_per_block: int
) -> Callable[[], Iterator[np.ndarray]]:
    """Restartable factory re-opening the loader's row-block iterator."""

    def factory() -> Iterator[np.ndarray]:
        it = iter_row_blocks(handle, loader, rows_per_block)
        assert it is not None
        return it

    return factory


def _worker_init() -> None:
    """Process-pool initializer: pin BLAS to one thread per worker.

    Same numeric path as the in-process stats phase (which runs under
    ``threadpool_limits(1)``), so results are byte-identical between the
    process-pool and serial paths. Also avoids 18 workers × N BLAS threads
    oversubscription.

    Heap containment (2026-09-02 OOM post-mortem): worker RSS must stay
    proportional to live tensors. Two mechanisms — see ``_worker_stats``
    for the per-task trim, and ``_run_stats_processes`` for the payload cap
    that keeps multi-hundred-MB tensors out of the pool entirely.
    """
    try:
        from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

        threadpool_limits(limits=1)
    except Exception:  # pragma: no cover - no supported BLAS in the worker
        pass


def _malloc_trim() -> None:
    """Return freed glibc heap to the OS (no-op on non-glibc)."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # pragma: no cover - non-glibc platforms
        pass


# Pool tasks above this size run serially in the main process instead of
# being pickled into a worker: the inflight bound (2 x workers) multiplied
# by multi-hundred-MB payloads can approach the RAM budget on huge scans.
_POOL_TASK_MAX_BYTES = 256 << 20  # 256 MiB


def _worker_stats(task: tuple) -> tuple[int, TensorStats]:
    """Compute all statistics for one tensor payload in a pool worker.

    The payload carries the dequantized float32 array (loaded by the main
    process — loader closures are not picklable and GGUF's shared 3D expert
    parents live in main-process RAM). ``stats.spectrum``'s lock is present
    in the worker too but uncontended (one stats thread per process).

    ``_malloc_trim`` after every task bounds glibc heap growth: numpy
    temporaries of varying sizes fragment the arena free lists and glibc
    rarely returns that memory to the OS on its own. Untrimmed workers on
    the 74k-tensor Flash-Next scan reached 10-14 GiB anon RSS for ≤120 MiB
    live tasks (kernel OOM, 2026-09-02); trimming keeps RSS flat.
    """
    i, name, shape, dtype, expert_id, arr, svd_seed, distribution_seed, quant_probe = task
    try:
        handle = TensorHandle(
            name=name,
            shape=tuple(shape),
            dtype=dtype,
            loader=_payload_loader(arr),
            expert_id=expert_id,
        )
        return i, _stats_for_handle(handle, svd_seed, distribution_seed, quant_probe)
    finally:
        del arr
        _malloc_trim()


def _payload_loader(arr: np.ndarray) -> Callable[[], np.ndarray]:
    """Loader returning a pickled-over payload array (mypy-friendly, no lambda)."""

    def _load() -> np.ndarray:
        return arr

    return _load


# Below this tensor count the process-pool spawn overhead (~1 s per worker)
# outweighs the parallelism gain; small scans stay on the thread path.
_PROCESS_POOL_MIN_TENSORS = 64


def _run_stats_processes(
    idxs: list[int],
    jobs_n: int,
    handles: list[TensorHandle],
    stats: list[TensorStats | None],
    svd_seed: int,
    distribution_seed: int,
    quant_probe: bool,
    report_stats: Callable[[int], None],
    record: Callable[[TensorStats], None],
) -> None:
    """Compute stats for ``idxs`` in a process pool, draining in place.

    - spawn context: no inherited locks (the API server runs scans from a
      worker thread while other threads exist — fork would be hazardous)
    - bounded outstanding submissions: the main process loads each tensor
      (mmap-backed), pickles it into the task, and releases the handle
      immediately — peak RAM stays ≈ 2 × workers × tensor size
    - per-tensor fallback: a failed worker task is recomputed serially in
      the main process (infra errors heal; genuine data errors still raise)

    Heap containment (2026-09-02 OOM post-mortem, Flash-Next 74k-tensor
    scan on a 125 GB host): payloads above ``_POOL_TASK_MAX_BYTES`` are NOT
    pickled into the pool — they are deferred to a serial tail pass after
    the pool drains, so inflight bytes stay bounded by small tensors × 2 x
    workers. Worker RSS is additionally trimmed per task
    (``_worker_stats``'s ``_malloc_trim``).
    """
    ctx = multiprocessing.get_context("spawn")
    inflight_limit = max(2, 2 * jobs_n)
    serial_tail: list[int] = []  # oversized for the pool — serial after drain
    with ProcessPoolExecutor(
        max_workers=jobs_n, mp_context=ctx, initializer=_worker_init
    ) as ex:
        futures: dict[Future, int] = {}
        pending = iter(idxs)

        def _fill() -> None:
            while len(futures) < inflight_limit:
                try:
                    i = next(pending)
                except StopIteration:
                    return
                h = handles[i]
                if int(np.prod(h.shape)) * 4 > _POOL_TASK_MAX_BYTES:
                    serial_tail.append(i)  # too big to pickle safely
                    continue
                arr = h.load()
                task = (
                    i, h.name, h.shape, h.dtype, h.expert_id, arr,
                    svd_seed, distribution_seed, quant_probe,
                )
                futures[ex.submit(_worker_stats, task)] = i
                h.clear()

        _fill()
        while futures:
            done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for fut in done:
                i = futures.pop(fut)
                try:
                    _, ts = fut.result()
                except Exception:
                    logger.warning(
                        "stats worker failed for tensor index %d — recomputing serially",
                        i,
                        exc_info=True,
                    )
                    h = handles[i]
                    ts = _stats_for_handle(h, svd_seed, distribution_seed, quant_probe)
                    h.clear()
                stats[i] = ts
                record(ts)
                report_stats(i)
            _fill()

    # Oversized payloads: serial, one at a time, in the main process.
    for i in serial_tail:
        h = handles[i]
        ts = _stats_for_handle(h, svd_seed, distribution_seed, quant_probe)
        h.clear()
        stats[i] = ts
        record(ts)
        report_stats(i)


# ── Stats checkpoint journal (crash resume) ────────────────────────────────
# Append-only JSONL, one line per completed tensor, flushed per line. On
# resume, a torn trailing line (crash mid-write) is skipped and the affected
# tensor recomputed — values are identical because every stat is a pure
# function of (tensor, seeds, probe flag), which the header identity pins.


def _model_identity(
    handles: list[TensorHandle],
    svd_seed: int,
    distribution_seed: int,
    quant_probe: bool,
) -> str:
    """Cheap identity of everything that changes per-tensor stat values."""
    import hashlib

    parts = sorted(
        f"{h.name}|{h.shape}|{h.dtype}|{h.expert_id}" for h in handles
    )
    payload = "\n".join(parts) + (
        f"|svd={svd_seed}|dist={distribution_seed}|probe={int(quant_probe)}|tool={_tool_version()}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _tool_version() -> str:
    try:
        return importlib.metadata.version("weight-atlas")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "0"


_CHECKPOINT_NAME = "stats_checkpoint.jsonl"


def _journal_load(path: Path, identity: str) -> dict[str, TensorStats]:
    """Read a checkpoint journal; empty dict when absent/invalid/foreign.

    Tolerates a torn trailing line (crash mid-write): unparseable lines are
    skipped, later duplicate entries win (identical values — stats are pure).
    """
    if not path.exists():
        return {}
    identity_ok = False
    entries: dict[str, TensorStats] = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue  # torn line
                if not isinstance(obj, dict):
                    continue
                if "_checkpoint_header" in obj:
                    identity_ok = obj["_checkpoint_header"].get("identity") == identity
                    continue
                if not identity_ok:
                    continue  # foreign journal — do not trust any entry
                name = obj.pop("name", None)
                if name is None:
                    continue
                try:
                    shape = tuple(int(s) for s in obj.pop("shape", []))
                    entries[str(name)] = TensorStats(name=str(name), shape=shape, **obj)
                except (TypeError, ValueError):
                    continue
    except OSError:
        return {}
    return entries if identity_ok else {}


def _journal_open(path: Path, identity: str, has_entries: bool) -> Any:
    """Open the journal for appending; write the header when starting fresh.

    Returns an open handle — the caller owns closing it (SIM115 is fine here:
    the handle's lifetime spans the whole stats phase, not a lexical block).
    """
    if not has_entries:
        fh = open(path, "w")  # noqa: SIM115
        fh.write(json.dumps({"_checkpoint_header": {"identity": identity}}) + "\n")
        fh.flush()
        return fh
    return open(path, "a")  # noqa: SIM115


def _journal_append(fh: Any, ts: TensorStats) -> None:
    from dataclasses import asdict

    # asdict carries name + shape; NaN floats are Python-JSON (consistent
    # with fingerprint.json).
    fh.write(json.dumps(asdict(ts)) + "\n")
    fh.flush()


def _journal_discard(path: Path) -> None:
    import contextlib

    with contextlib.suppress(OSError):
        path.unlink()


def _stats_for_handle(
    h: TensorHandle, svd_seed: int = 0, distribution_seed: int = 0, quant_probe: bool = False
) -> TensorStats:
    """Compute all statistics for one tensor.

    BLAS threading is capped once by ``scan()`` around the whole parallel
    section (see ``scan``), never per tensor: resizing OpenBLAS's internal
    thread pool from several worker threads at once can deadlock inside
    OpenBLAS (observed on large MoE scans), so the limit must be applied a
    single time before the pool starts, not re-entered for every tensor.
    """
    return _make_handles(h, svd_seed, quant_probe=quant_probe)


def scan(
    model_path: Path,
    out: Path,
    spec: AtlasSpec,
    *,
    loader_id: str | None = None,
    progress: Callable[[float, str], None] | None = None,
    jobs: int | None = None,
    quant_probe: bool = False,
    fresh: bool = False,
) -> list[Path]:
    """Run the full scan pipeline.

    Produces:
    - fingerprint.json (sorted keys, indent 2)
    - field_<channel>_raw.tif
    - field_<channel>_smooth.tif
    - field_expert_<slot>_{raw,smooth}.tif (for MoE models)
    - manifest.json (sha256 per artefact)

    Args:
        model_path: path to model file or directory
        out: output directory
        spec: atlas specification
        loader_id: override loader (default: auto-detect)
        progress: optional ``(fraction, message)`` callback reported as each
            phase of the pipeline completes (loading, statistics, rasterizing,
            smoothing, expert panels, embedding, manifest).
        jobs: number of parallel statistics workers (default: cores - 2).
        quant_probe: also compute the measured RTN-SQNR stats
            (``sqnr_int8_ch``/``sqnr_int4_g128``/``sqnr_fp8_e4m3``); opt-in
            because it adds ~6 chunked passes over every weight tensor.
            Each tensor's statistics are computed independently and
            deterministically, so results are identical for any ``jobs``.
        fresh: ignore and overwrite an existing stats checkpoint journal
            (default: resume a crashed scan when the journal identity
            matches — same tensor set, seeds, probe flag and tool version).
    """
    def _report(pct: float, msg: str) -> None:
        if progress is not None:
            progress(float(pct), msg)

    out.mkdir(parents=True, exist_ok=True)

    # Auto-detect loader if not specified
    if loader_id is None:
        loader_id = detect_loader(model_path)

    _report(0.0, "Opening model...")
    loader = get_loader(loader_id)()
    _report(0.02, "Reading tensor metadata...")
    handles = list(loader.open(model_path))
    loader_metadata = getattr(loader, "metadata", {})

    # Compute per-tensor statistics (the expensive SVD steps), optionally in
    # parallel across tensors. Every handle's memoized payload is released
    # right after its statistics are computed so the whole model is never held
    # in RAM (~4 bytes/parameter for a 35B MoE would be ~140 GB).
    #
    # Memory bound: computing statistics materializes each tensor as float32
    # and several stats build temporary copies (kurtosis, sparsity, SVD), so a
    # single worker can transiently use ~3-4x the tensor's float32 size. With 8
    # parallel workers on a model whose largest tensors are multi-GB (e.g. Kimi
    # K3's 4.7 GB lm_head/embed_tokens), peak RSS reached ~120 GB and the
    # process was OOM-killed. The few multi-GB tensors are therefore computed
    # serially (one at a time) while the many small tensors still run in
    # parallel for throughput. Per-tensor stats are deterministic regardless of
    # processing order, so output is byte-identical to a fully serial scan.
    n_total = len(handles)
    report_every = max(1, n_total // 40) if n_total else 1
    jobs_n = _resolve_jobs(jobs)
    # Shared rSVD seed: the paired pipeline reads the same spec key, so scan
    # spectra and Δ-spectra stay comparable when it changes.
    svd_seed = int(spec.seeds.get("svd", 0))
    distribution_seed = int(spec.seeds.get("distribution", 0))

    # Tensors that materialize to >= 1 GiB float32 are handled serially to keep
    # peak RAM bounded even on models with very large tensors.
    big_threshold = _BIG_TENSOR_THRESHOLD_BYTES
    # Giant tensors: block-streamed (never materialized). These take
    # precedence over the big/small split.
    stream_idxs = [
        i for i, h in enumerate(handles)
        if int(np.prod(h.shape)) * 4 >= _STREAM_TENSOR_BYTES
    ]
    stream_set = set(stream_idxs)
    big_idxs = [
        i for i, h in enumerate(handles)
        if big_threshold <= int(np.prod(h.shape)) * 4 < _STREAM_TENSOR_BYTES
    ]
    small_idxs = [
        i for i, h in enumerate(handles)
        if int(np.prod(h.shape)) * 4 < big_threshold and i not in stream_set
    ]

    stats: list[TensorStats | None] = [None] * n_total

    # ── Checkpoint journal: resume a crashed scan from per-tensor stats ────
    checkpoint_path = out / _CHECKPOINT_NAME
    identity = _model_identity(handles, svd_seed, distribution_seed, quant_probe)
    resumed: dict[str, TensorStats] = {} if fresh else _journal_load(checkpoint_path, identity)
    pending_small = [i for i in small_idxs if handles[i].name not in resumed]
    pending_big = [i for i in big_idxs if handles[i].name not in resumed]
    pending_stream = [i for i in stream_idxs if handles[i].name not in resumed]
    for i in small_idxs + big_idxs + stream_idxs:
        ts = resumed.get(handles[i].name)
        if ts is not None:
            stats[i] = ts
    n_resumed = n_total - len(pending_small) - len(pending_big) - len(pending_stream)
    if n_resumed:
        _report(
            0.04 + 0.36 * (n_resumed / n_total),
            f"Resuming from checkpoint ({n_resumed}/{n_total} tensors)...",
        )
    journal_fh = _journal_open(checkpoint_path, identity, bool(resumed))

    # Records with no handle slot (streaming per-head n-gram segments): they
    # ride the journal for crash resume but live in this list for the
    # fingerprint — the resume path re-reads them by name.
    extra_stats: list[TensorStats] = []
    extra_names = {h.name for h in handles}
    for ts in resumed.values():
        if ts.name not in extra_names:
            extra_stats.append(ts)

    def record(ts: TensorStats) -> None:
        _journal_append(journal_fh, ts)
        if ts.name not in extra_names:
            extra_stats.append(ts)

    def _report_stats(i: int) -> None:
        if i % report_every == 0 or i == n_total - 1:
            _report(
                0.04 + 0.36 * ((i + 1) / n_total),
                f"Computing statistics ({i + 1}/{n_total})...",
            )

    def _run_stats(idx_iter: Iterable[int], parallel: bool) -> None:
        items = list(idx_iter)
        if parallel and jobs_n > 1 and len(items) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=jobs_n) as ex:
                for i, ts in ex.map(
                    lambda i: (i, _stats_for_handle(handles[i], svd_seed, distribution_seed, quant_probe)), items
                ):
                    handles[i].clear()
                    stats[i] = ts
                    record(ts)
                    _report_stats(i)
        else:
            for i in items:
                ts = _stats_for_handle(handles[i], svd_seed, distribution_seed, quant_probe)
                handles[i].clear()
                stats[i] = ts
                record(ts)
                _report_stats(i)

    # Small tensors first (process pool → thread pool → serial), then the few
    # multi-GB tensors serially.
    use_pool = jobs_n > 1 and n_total > 1
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - optional dep
        threadpool_limits = None

    remaining_small = list(pending_small)
    try:
        if (
            use_pool
            and len(remaining_small) >= _PROCESS_POOL_MIN_TENSORS
        ):
            # Primary path: spawn-process pool. Workers have their own BLAS
            # context (pinned to 1 thread by _worker_init), so the OpenBLAS
            # concurrent-LAPACK hazard that forces _spectrum_lock in-process does
            # not apply — the SVD phase parallelizes across cores. All process
            # failures degrade to the in-process paths below.
            try:
                _run_stats_processes(
                    remaining_small, jobs_n, handles, stats,
                    svd_seed, distribution_seed, quant_probe, _report_stats, record,
                )
                remaining_small = []
            except Exception:
                # Pool-level failure (broken /dev/shm, spawn issues) → thread path.
                logger.warning(
                    "process stats pool failed — falling back to threads", exc_info=True
                )
        if remaining_small and use_pool:
            if threadpool_limits is not None:
                # The stats thread pool runs numpy/BLAS concurrently. Cap the BLAS
                # thread pool ONCE around the whole section: entering/exiting
                # ``threadpool_limits`` from several worker threads at once resizes
                # OpenBLAS's internal pool concurrently and can deadlock inside
                # OpenBLAS (observed on large MoE scans). A single cap keeps every
                # BLAS call single-threaded with no pool resizing mid-flight.
                try:
                    with threadpool_limits(limits=1):
                        _run_stats(remaining_small, parallel=True)
                    remaining_small = []
                except RuntimeError:
                    # threadpoolctl present but no supported BLAS loaded → there is no
                    # shared thread pool to resize, so parallel numpy is safe.
                    _run_stats(remaining_small, parallel=True)
                    remaining_small = []
            else:
                # jobs=1, or no threadpoolctl → cannot cap BLAS. Concurrent
                # multithreaded OpenBLAS calls from several Python threads risk the
                # same deadlock, so stats run serially (safe, deterministic).
                _run_stats(remaining_small, parallel=False)
                remaining_small = []
        if remaining_small:
            _run_stats(remaining_small, parallel=False)

        # Giant tensors: block-streamed, serial (I/O bound — never
        # materialized, hence never pooled). Each pass re-opens the block
        # iterator (blocks_factory) — the mmap data stays on the loader.
        for i in pending_stream:
            h = handles[i]
            probe = iter_row_blocks(h, loader, _ROWS_PER_BLOCK_DEFAULT)
            if probe is None:
                logger.warning(
                    "tensor %s needs streaming but the loader has no block "
                    "support — falling back to full load (may OOM)",
                    h.name,
                )
                blocks_factory = _constant_blocks(h.load())
            else:
                del probe  # cheap to re-open; streaming_tensor_stats walks 3-4x
                blocks_factory = _loader_blocks(h, loader, _ROWS_PER_BLOCK_DEFAULT)

            head_bounds = None
            ngram_info = getattr(loader, "ngram_head_bounds", None)
            if ngram_info is not None:
                head_name, bounds = ngram_info
                if h.name == head_name:
                    head_bounds = bounds
            ts, head_records = streaming_tensor_stats(
                h.name, h.shape, h.dtype, h.expert_id, blocks_factory,
                svd_seed=svd_seed, distribution_seed=distribution_seed,
                quant_probe=quant_probe, head_bounds=head_bounds,
            )
            h.clear()
            stats[i] = ts
            record(ts)
            for head_rec in head_records:
                record(head_rec)
            _report_stats(i)

        # Few multi-GB tensors: serial, BLAS capped, peak RAM bounded.
        if threadpool_limits is not None:
            try:
                with threadpool_limits(limits=1):
                    _run_stats(pending_big, parallel=False)
            except RuntimeError:
                _run_stats(pending_big, parallel=False)
        else:
            _run_stats(pending_big, parallel=False)
    finally:
        # Close (data is flushed per line). The journal FILE is kept on
        # failure — that is the resume payload — and discarded on success
        # below; in all cases the handle is closed deterministically.
        journal_fh.close()

    _journal_discard(checkpoint_path)  # success → fingerprint is authoritative

    # Rebind: drop the None placeholders (all tensors are computed now) so
    # downstream consumers see list[TensorStats]. Per-head extras (no handle
    # slot) are NOT appended to stats_narrow: scaling metadata and field
    # rasterisation must see exactly the handle records (byte-identical
    # with scans that have no head segments); the fingerprint gets
    # handle records + extras.
    stats_narrow: list[TensorStats] = [s for s in stats if s is not None]
    assert len(stats_narrow) == n_total
    fp_stats = stats_narrow + extra_stats if extra_stats else stats_narrow

    _report(0.42, "Building fingerprint...")
    fingerprint = _build_fingerprint(fp_stats, spec, loader_id, handles, loader_metadata)

    # Compute scaling metadata for fingerprint (v2.1)
    scaling_meta = _compute_scaling_metadata(stats_narrow, spec)
    if scaling_meta:
        fingerprint["scaling"] = scaling_meta

    fp_path = out / "fingerprint.json"
    with open(fp_path, "w") as f:
        json.dump(fingerprint, f, indent=2, sort_keys=True)
        f.write("\n")

    artefacts: list[Path] = [fp_path]
    fields_for_diag: dict[str, np.ndarray] = {}
    n_channels = len(spec.channels)
    for ci, (channel, ch_spec) in enumerate(spec.channels.items()):
        chan_lo = 0.44 + 0.20 * (ci / n_channels)
        chan_hi = 0.44 + 0.20 * ((ci + 1) / n_channels)
        stat_key = ch_spec["stat"]
        _report(chan_lo, f"Rasterizing {channel} field ({stat_key})...")
        field_raw = rasterize(stats_narrow, spec, stat_key)
        raw_path = out / f"field_{channel}_raw.tif"
        write_tif(raw_path, field_raw.data)
        artefacts.append(raw_path)

        # v2.1 pipeline: apply pre-transform (e.g. log1p) then robust_scale
        pre = ch_spec.get("pre")
        data = field_raw.data
        if pre == "log1p":
            data = log1p(data)
        scaled = apply_scale(data, ch_spec["scale"])
        from weight_atlas.fields.degenerations import diagnose_fields
        from weight_atlas.fields.smoothing import smooth, upsample

        _report(chan_lo + 0.55 * (chan_hi - chan_lo), f"Smoothing {channel} field...")
        up = upsample(scaled, int(spec.grid["upsample"]))
        smoothed = smooth(up, float(spec.grid["smooth_sigma"]))
        smooth_path = out / f"field_{channel}_smooth.tif"
        write_tif(smooth_path, smoothed)
        artefacts.append(smooth_path)

    # Flat visualization for non-layered models (e.g. BDH): a single-row
    # grid with one column per mapped slot.  Only generated when the main
    # raster produced no per-layer fields (i.e. all tensors mapped to
    # layer=None).
    has_flat = any(
        map_name(ts.name)[0] is None
        for ts in stats_narrow
        if not is_expert_tensor(ts.name) and not is_shared_expert(ts.name)
    )
    has_per_layer = any(
        map_name(ts.name)[0] is not None
        for ts in stats_narrow
        if not is_expert_tensor(ts.name) and not is_shared_expert(ts.name)
    )
    if has_flat and not has_per_layer:
        from weight_atlas.fields.smoothing import smooth, upsample

        for _ci, (channel, ch_spec) in enumerate(spec.channels.items()):
            stat_key = ch_spec["stat"]
            _report(0.65, f"Rasterizing flat field ({channel})...")
            flat_field = rasterize_flat(stats_narrow, spec, stat_key)
            if flat_field is None:
                continue
            flat_raw_path = out / f"field_flat_{channel}_raw.tif"
            write_tif(flat_raw_path, flat_field.data)
            artefacts.append(flat_raw_path)

            pre = ch_spec.get("pre")
            data = flat_field.data
            if pre == "log1p":
                data = log1p(data)
            scaled = apply_scale(data, ch_spec["scale"])
            up = upsample(scaled, int(spec.grid["upsample"]))
            smoothed = smooth(up, float(spec.grid["smooth_sigma"]))
            flat_smooth_path = out / f"field_flat_{channel}_smooth.tif"
            write_tif(flat_smooth_path, smoothed)
            artefacts.append(flat_smooth_path)

    # MoE expert panels. Expert tensors are the vast majority of a MoE model's
    # tensors, so the panels use the spec's ``expert_channels`` (cheap O(n)
    # statistics: frobenius/kurtosis/sparsity) instead of the SVD-based main
    # channels — the shared spectrum is reserved for the (few) dense tensors.
    panel_channels = spec.expert_channels if spec.expert_channels else spec.channels
    n_panel_channels = max(1, len(panel_channels))
    for pi, (channel, ch_spec) in enumerate(panel_channels.items()):
        stat_key = ch_spec["stat"]
        # Panels run right after the main channels — report in that order
        # instead of jumping ahead to 0.93 and back.
        _report(
            0.66 + 0.06 * (pi / n_panel_channels),
            f"Generating expert panels ({stat_key})...",
        )
        expert_panels = rasterize_expert_panels(stats_narrow, spec, stat_key)
        for panel in expert_panels:
            panel_raw_path = out / f"field_expert_{panel.slot}_{channel}_raw.tif"
            write_tif(panel_raw_path, panel.data)
            artefacts.append(panel_raw_path)

            # v2.1 pipeline: apply pre-transform then robust_scale
            pre = ch_spec.get("pre")
            panel_data = panel.data
            if pre == "log1p":
                panel_data = log1p(panel_data)
            scaled_panel = apply_scale(panel_data, ch_spec["scale"])
            up_panel = upsample(scaled_panel, int(spec.grid["upsample"]))
            smoothed_panel = smooth(up_panel, float(spec.grid["smooth_sigma"]))
            panel_smooth_path = out / f"field_expert_{panel.slot}_{channel}_smooth.tif"
            write_tif(panel_smooth_path, smoothed_panel)
            artefacts.append(panel_smooth_path)

    # BDH route-lattice panels: per-(head, unit) grids for the core tensors,
    # written with the expert-panel naming so the sheet renderer and compare
    # panel machinery pick them up unchanged. Uses the main spec channels
    # (spectral/stable SVD stats are cheap at [D, unit] block size).
    lattice_panels: list = []
    for _channel, ch_spec in spec.channels.items():
        lattice_panels.extend(rasterize_bdh_lattice(stats_narrow, spec, ch_spec["stat"]))
    if lattice_panels:
        _report(0.72, "Generating BDH route-lattice panels...")
        from weight_atlas.fields.smoothing import smooth, upsample

        for panel in lattice_panels:
            for channel, ch_spec in spec.channels.items():
                if panel.channel != ch_spec["stat"]:
                    continue
                panel_raw_path = out / f"field_expert_{panel.slot}_{channel}_raw.tif"
                write_tif(panel_raw_path, panel.data)
                artefacts.append(panel_raw_path)

                pre = ch_spec.get("pre")
                panel_data = panel.data
                if pre == "log1p":
                    panel_data = log1p(panel_data)
                scaled_panel = apply_scale(panel_data, ch_spec["scale"])
                up_panel = upsample(scaled_panel, int(spec.grid["upsample"]))
                smoothed_panel = smooth(up_panel, float(spec.grid["smooth_sigma"]))
                panel_smooth_path = out / f"field_expert_{panel.slot}_{channel}_smooth.tif"
                write_tif(panel_smooth_path, smoothed_panel)
                artefacts.append(panel_smooth_path)

    # Vision tower fields (VLM models): a separate sheet with its own slot
    # taxonomy and statistics, so multimodal models show a distinct fingerprint
    # instead of having their vision tensors silently dropped.
    if spec.vision_slots and spec.vision_channels:
        from weight_atlas.fields.smoothing import smooth, upsample

        n_vis = len(spec.vision_channels)
        for vi, (channel, ch_spec) in enumerate(spec.vision_channels.items()):
            vis_lo = 0.74 + 0.04 * (vi / n_vis)
            stat_key = ch_spec["stat"]
            _report(vis_lo, f"Rasterizing vision {channel} field ({stat_key})...")
            vision_field = rasterize_vision(stats_narrow, spec, stat_key)
            if vision_field is None:
                continue  # text-only model — no vision tensors
            field_name = f"vision_{channel}"
            raw_path = out / f"field_{field_name}_raw.tif"
            write_tif(raw_path, vision_field.data)
            artefacts.append(raw_path)

            pre = ch_spec.get("pre")
            data = vision_field.data
            if pre == "log1p":
                data = log1p(data)
            scaled = apply_scale(data, ch_spec["scale"])
            up = upsample(scaled, int(spec.grid["upsample"]))
            smoothed = smooth(up, float(spec.grid["smooth_sigma"]))
            smooth_path = out / f"field_{field_name}_smooth.tif"
            write_tif(smooth_path, smoothed)
            artefacts.append(smooth_path)

            fields_for_diag[field_name] = vision_field.data

    # Embedding projection (PCA or UMAP)
    embedding_spec = getattr(spec, 'embedding', {})
    if embedding_spec:
        _report(0.80, "Projecting embeddings...")
        method = embedding_spec.get('method', 'pca')
        grid_size = embedding_spec.get('grid', 256)
        n_components = embedding_spec.get('components', 3)
        subsample = embedding_spec.get('subsample_scatter', 5000)
        seeds = embedding_spec.get('seeds', {'pca': 0, 'umap': 0})

        # Find embedding tensor (handle HF, GGUF, and prefixed VLM naming e.g.
        # Kimi K3's ``language_model.model.embed_tokens.weight``).
        embed_tensor = None
        for h in handles:
            if h.name.endswith(('model.embed_tokens.weight', 'token_embd.weight')):
                embed_tensor = h
                break

        if embed_tensor is not None:
            embeddings = embed_tensor.load()  # (V, D)
            _report(0.84, f"Projecting embeddings ({method})...")

            if method == 'umap':
                from weight_atlas.embedding.umap import compute_umap
                projected, umap_meta = compute_umap(
                    embeddings,
                    n_components=2,
                    seed=seeds.get('umap', 0),
                )
                # Save UMAP result
                np.save(out / 'embedding_umap.npy', projected.astype(np.float32))
                artefacts.append(out / 'embedding_umap.npy')
                embedding_meta = umap_meta
            else:
                # PCA (default)
                from weight_atlas.embedding.pca import (
                    compute_pca,
                    embedding_to_density,
                    project_with_pca,
                )
                components, explained_variance, mean = compute_pca(
                    embeddings,
                    n_components=n_components,
                    seed=seeds.get('pca', 0),
                )
                projected = project_with_pca(embeddings, components, mean)

                # Save PCA result
                np.save(out / 'embedding_pca.npy', projected.astype(np.float32))
                artefacts.append(out / 'embedding_pca.npy')

                # Create density field
                density = embedding_to_density(
                    projected[:, :2],
                    grid_size=grid_size,
                    subsample=subsample,
                    seed=seeds.get('pca', 0),
                )

                # Write density TIFFs
                raw_path = out / 'field_embed_density_raw.tif'
                write_tif(raw_path, density)
                artefacts.append(raw_path)

                from weight_atlas.fields.degenerations import diagnose_fields
                from weight_atlas.fields.smoothing import smooth, upsample

                scaled = apply_scale(density, {'type': 'log1p'})
                up = upsample(scaled, int(spec.grid['upsample']))
                smoothed = smooth(up, float(spec.grid['smooth_sigma']))
                smooth_path = out / 'field_embed_density_smooth.tif'
                write_tif(smooth_path, smoothed)
                artefacts.append(smooth_path)

                # Save scatter coordinates (subsampled for visualization)
                subsample_scatter = embedding_spec.get('subsample_scatter', 5000)
                scatter_seed = seeds.get('pca', 0)
                rng = np.random.default_rng(scatter_seed)
                n_points = projected.shape[0]
                if n_points > subsample_scatter:
                    scatter_indices = rng.choice(n_points, size=subsample_scatter, replace=False)
                    scatter_coords = projected[scatter_indices, :2]
                else:
                    scatter_coords = projected[:, :2]

                np.save(out / 'embedding_scatter.npy', scatter_coords.astype(np.float32))
                artefacts.append(out / 'embedding_scatter.npy')

                embedding_meta = {
                    'method': 'pca',
                    'explained_variance': explained_variance.tolist(),
                    'n_components': n_components,
                    'sign_convention': 'max_abs_positive',
                    'scatter_subsample': subsample_scatter,
                    'scatter_seed': scatter_seed,
                }

            # Save embedding metadata
            with open(out / 'embedding_meta.json', 'w') as f:
                json.dump(embedding_meta, f, indent=2)
            artefacts.append(out / 'embedding_meta.json')

    # Degeneration checks on raw fields
    _report(0.93, "Checking field degenerations...")
    for channel in spec.channels:
        raw_path = out / f"field_{channel}_raw.tif"
        if raw_path.exists():
            from weight_atlas.fields.tif_io import read_tif
            fields_for_diag[channel] = read_tif(raw_path)
    if fields_for_diag:
        degen_report = diagnose_fields(fields_for_diag)
        if degen_report.warnings:
            fingerprint["warnings"] = fingerprint.get("warnings", []) + degen_report.warnings
            # fingerprint.json was written before the field TIFFs existed, so
            # the merge above happened after serialization — rewrite the file
            # or the warnings are computed and silently dropped. The manifest
            # below hashes this final content.
            with open(fp_path, "w") as f:
                json.dump(fingerprint, f, indent=2, sort_keys=True)
                f.write("\n")

    _report(0.97, "Writing manifest...")
    manifest = {str(p.relative_to(out)): _sha256(p) for p in artefacts}
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    return artefacts + [manifest_path]


def _build_fingerprint(
    stats: Iterable[TensorStats],
    spec: AtlasSpec,
    loader_id: str,
    handles: list[TensorHandle] | None = None,
    loader_metadata: dict[str, str] | None = None,
) -> dict:
    """Build the fingerprint dict from computed tensor statistics."""
    # Get tool version from package metadata
    try:
        tool_version = importlib.metadata.version("weight-atlas")
    except importlib.metadata.PackageNotFoundError:
        tool_version = "0.2.0"

    out: dict = {
        "spec_version": spec.spec_version,
        "tool_version": tool_version,
        "loader": loader_id,
        "loader_metadata": loader_metadata or {},
        "model": {"n_tensors": 0, "n_layers": 0},
        "tensors": {},
    }
    layers: set[int] = set()

    # Build ggml_type/dtype mapping from handles if available.
    ggml_types: dict[str, str] = {}
    dtypes: dict[str, str] = {}
    if handles:
        for h in handles:
            if h.dtype.startswith("ggml_"):
                ggml_types[h.name] = h.dtype
            dtypes[h.name] = h.dtype

    for ts in stats:
        layer, slot = map_name(ts.name)
        if layer is not None:
            layers.add(layer)
        tensor_info = {
            "shape": list(ts.shape),
            "frobenius": ts.frobenius,
            "spectral_norm": ts.spectral_norm,
            "effective_rank": ts.effective_rank,
            "stable_rank": ts.stable_rank,
            "kurtosis": ts.kurtosis,
            "sparsity": ts.sparsity,
            "kernel_norm": ts.kernel_norm,
            "sv_decay": ts.sv_decay,
            "row_amax_ratio": ts.row_amax_ratio,
            "col_amax_ratio": ts.col_amax_ratio,
            "mean": ts.mean,
            "std": ts.std,
            "absmax": ts.absmax,
            "absmean": ts.absmean,
            "p50": ts.p50,
            "p90": ts.p90,
            "p99": ts.p99,
            "p999": ts.p999,
            "p9999": ts.p9999,
            "outlier_3s": ts.outlier_3s,
            "outlier_4s": ts.outlier_4s,
            "outlier_6s": ts.outlier_6s,
            "dyn_range": ts.dyn_range,
            "sqnr_int8_ch": ts.sqnr_int8_ch,
            "sqnr_int4_g128": ts.sqnr_int4_g128,
            "sqnr_fp8_e4m3": ts.sqnr_fp8_e4m3,
            "streamed": ts.streamed,
        }
        # Add ggml_type if present
        if ts.name in ggml_types:
            tensor_info["ggml_type"] = ggml_types[ts.name]
        # Add on-disk dtype for type_map (GGUF ggml_type, safetensors dtype)
        if ts.name in dtypes:
            tensor_info["dtype"] = dtypes[ts.name]
        out["tensors"][ts.name] = tensor_info

    out["model"]["n_tensors"] = len(out["tensors"])
    out["model"]["n_layers"] = len(layers)

    # Add mapping coverage (name audit)
    n_mapped = sum(1 for name in out["tensors"] if map_name(name)[1] != "other")
    n_total = len(out["tensors"])
    n_unmapped = n_total - n_mapped
    out["mapping_coverage"] = {
        "in_slots": round(n_mapped / n_total, 4) if n_total > 0 else 0.0,
        "in_other": round(n_unmapped / n_total, 4) if n_total > 0 else 0.0,
        "unmapped": n_unmapped,
        "unmapped_tensors": [name for name in out["tensors"] if map_name(name)[1] == "other"][:20],
    }

    # Add quantization summary for GGUF
    if loader_id == "gguf" and ggml_types:
        quant_summary: dict[str, int] = {}
        for ggml_type in ggml_types.values():
            quant_summary[ggml_type] = quant_summary.get(ggml_type, 0) + 1
        out["quantization"] = quant_summary

    # Add MoE info
    moe_info = detect_moe(stats)
    if moe_info:
        out["model"]["moe"] = moe_info

    # Add vision-tower info (VLM models) — mapped tensors, block count, and the
    # number of global tensors (patch_embed / pos_embed / projector). Text-only
    # models get no ``model.vision`` block.
    vision_info = detect_vision(stats)
    if vision_info:
        out["model"]["vision"] = vision_info
        out["mapping_coverage"]["vision_tensors"] = vision_info["n_tensors"]

    return out


def _compute_scaling_metadata(stats: Iterable[TensorStats], spec: AtlasSpec) -> dict | None:
    """Compute scaling metadata for fingerprint.json (v2.1).

    For each channel, records the robust scale parameters and the raw/clip bounds.
    Only applies when spec uses robust_scale.
    """
    # Check if any channel uses robust_scale
    has_robust = any(
        ch_spec["scale"]["type"] in ("robust_scale", "quantile_clip")
        for ch_spec in spec.channels.values()
    )
    if not has_robust:
        return None

    # Build per-channel stat arrays from stats
    channels_meta: dict[str, dict] = {}
    params: dict[str, float] | None = None
    for channel, ch_spec in spec.channels.items():
        stat_key = ch_spec["stat"]
        scale_type = ch_spec["scale"]["type"]
        if scale_type not in ("robust_scale", "quantile_clip"):
            continue
        # Record the actual quantile bounds from the spec (not hardcoded).
        lower = float(ch_spec["scale"].get("lower", ch_spec["scale"].get("lo", 0.01)))
        upper = float(ch_spec["scale"].get("upper", ch_spec["scale"].get("hi", 0.99)))
        if params is None:
            params = {"lower": lower, "upper": upper}

        # Collect all values for this stat across tensors
        vals_list: list[float] = []
        for ts in stats:
            v = getattr(ts, stat_key, None)
            if v is not None and np.isfinite(v):
                vals_list.append(float(v))

        if not vals_list:
            continue

        arr = np.array(vals_list, dtype=np.float64)
        raw_min = float(np.min(arr))
        raw_max = float(np.max(arr))
        # Apply pre-transform (e.g. log1p) before computing clip bounds (v2.1)
        pre = ch_spec.get("pre")
        if pre == "log1p":
            arr = np.log1p(np.maximum(arr, 0.0))

        q_lo = float(np.quantile(arr, lower))
        q_hi = float(np.quantile(arr, upper))

        channels_meta[channel] = {
            "q_lo": round(q_lo, 4),
            "q_hi": round(q_hi, 4),
            "raw_min": round(raw_min, 4),
            "raw_max": round(raw_max, 4),
        }

    if not channels_meta:
        return None

    return {
        "method": "robust_scale",
        "params": params or {"lower": 0.01, "upper": 0.99},
        "channels": channels_meta,
    }
