"""Streaming statistics for giant tensors (n-gram embedding tables).

Computes the full TensorStats record without ever materializing the tensor:
the caller supplies a *restartable* row-block factory (see
:func:`weight_atlas.loaders.blocking.iter_row_blocks`), and every statistic
accumulates block-wise in float64.

Spectral path — the review-F-2 concern (curve shape between σ₃ and σ₆₄) is
resolved by the real Flash-Next data: the n-gram table is
[320.003.536 × 160], so the small dimension is tiny.

- **Gram path** (``min(rows, cols) <= _GRAM_MAX_DIM``): ``G = Σ blockᵀ·block``
  accumulated in float64 over row blocks; ``eigvalsh(G)`` yields the **exact
  full spectrum** — every singular value of the small dimension, no k-
  truncation. This is *the* measurement: smooth decay vs. spiky tail over the
  whole curve. Caveat (documented): the Gram squares the condition number —
  relative σ error grows as ε·(σ₁/σᵢ)². With fp64 accumulation that stays far
  inside the review-F-1 tolerance (rel 1e-4 on σ) unless the matrix is
  pathologically conditioned at the tiny-σ tail.
- **Halko path** (both dimensions huge): block-wise sketch with
  ``STREAM_RSVD_K = 32`` (review F-2: more than the in-process k=16 because
  the curve shape matters here; top-k semantics as the in-process rSVD).

O(n) stats accumulate block-wise (chunked float64, the established
discipline). Percentile ladder and row-amax sample deterministically above a
cap — seeded global stride, never unseeded.

Per-head segmentation (Flash-Next PLE table, 16 contiguous head segments):
when ``head_bounds`` partitions the row axis, each block is routed (sliced at
head boundaries) into per-head accumulators across all passes, yielding one
exact-spectrum record per head — the per-head embedding-vs-lookup answer at
~10 MB per 16k-bucket Gram block.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from weight_atlas.core.types import TensorStats
from weight_atlas.stats.spectrum import entropy_rank

_GRAM_MAX_DIM = 8192  # Gram accumulator 8192² × 8 B = 537 MiB — fine
STREAM_RSVD_K = 32
_SAMPLE_CAP = 2_000_000
_ROW_SAMPLE_CAP = 64_000_000
_CHUNK = 2**20


def _chunks(block: np.ndarray) -> Iterator[np.ndarray]:
    flat = block.reshape(-1)
    for i in range(0, flat.size, _CHUNK):
        yield flat[i : i + _CHUNK].astype(np.float64)


def _sqnr_from(sig: float, err: float) -> float:
    if sig == 0.0:
        return float("nan")
    if err == 0.0:
        return 300.0  # lossless ceiling, JSON-safe (stats.sqnr contract)
    return float(10.0 * np.log10(sig / err))


def _spectrum_gram(blocks: Callable[[], Iterator[np.ndarray]], cols: int) -> np.ndarray:
    gram = np.zeros((cols, cols), dtype=np.float64)
    for block in blocks():
        b = block.astype(np.float64)
        gram += b.T @ b
    eig = np.linalg.eigvalsh(gram)
    eig = np.clip(eig[::-1], 0.0, None)
    spectrum: np.ndarray = np.sqrt(eig)  # annotated local: ufunc returns Any under numpy 2.4 stubs
    return spectrum


def _spectrum_halko(
    blocks: Callable[[], Iterator[np.ndarray]], rows: int, cols: int, k: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    kk = min(k, rows, cols)
    omega = rng.standard_normal((cols, kk)).astype(np.float32)
    y = np.zeros((rows, kk), dtype=np.float32)
    lo = 0
    for block in blocks():
        r = block.shape[0]
        y[lo : lo + r] = block @ omega
        lo += r
    q, _ = np.linalg.qr(y)
    bt = np.zeros((kk, cols), dtype=np.float32)
    lo = 0
    for block in blocks():
        r = block.shape[0]
        bt += q[lo : lo + r].T @ block
        lo += r
    return np.linalg.svd(bt.astype(np.float64), compute_uv=False)


def _head_stats_from_gram(
    gram: np.ndarray, head: dict[str, Any]
) -> TensorStats:
    """Per-head record: exact full spectrum from the head's Gram + scalars."""
    eig = np.clip(np.linalg.eigvalsh(gram)[::-1], 0.0, None)
    spectrum = np.sqrt(eig)
    spectral_norm = float(spectrum[0]) if spectrum.size else float("nan")
    frobenius = float(np.sqrt((spectrum**2).sum())) if spectrum.size else float("nan")
    n = float(head["n"])
    mean = head["s1"] / n if n else float("nan")
    hm2 = head["s2"] / n - mean * mean if n else 0.0
    std = float(np.sqrt(max(hm2, 0.0)))
    absmean = head["sabs"] / n if n else float("nan")
    return TensorStats(
        name=head["record_name"],
        shape=(head["end"] - head["start"], head["cols"]),
        frobenius=frobenius,
        spectral_norm=spectral_norm,
        effective_rank=float(entropy_rank(spectrum)) if spectrum.size else float("nan"),
        stable_rank=(
            1.0
            if spectrum.size == 1
            else float(np.log1p((frobenius / spectral_norm) ** 2))
            if spectrum.size and spectral_norm > 0
            else float("nan")
        ),
        sv_decay=float(spectrum[-1] / spectrum[0]) if spectrum.size and spectrum[0] > 0 else float("nan"),
        kurtosis=head.get("kurtosis", float("nan")),
        sparsity=head["n_zeros"] / n if n else float("nan"),
        mean=mean,
        std=std,
        absmean=absmean,
        absmax=head["amax"],
        streamed=True,
    )


def streaming_tensor_stats(
    name: str,
    shape: tuple[int, ...],
    dtype: str,
    expert_id: int | None,
    blocks_factory: Callable[[], Iterator[np.ndarray]],
    *,
    svd_seed: int = 0,
    distribution_seed: int = 0,
    quant_probe: bool = False,
    head_bounds: list[int] | None = None,
) -> tuple[TensorStats, list[TensorStats]]:
    """Full TensorStats record for a tensor supplied as restartable row blocks.

    Returns ``(main, extras)`` — ``extras`` holds one per-head record when
    ``head_bounds`` partitions the row axis (Flash-Next n-gram table: 16
    contiguous head segments; each head gets an exact full spectrum from its
    own Gram accumulator — one pass routes every block to exactly one head).
    All outputs are deterministic given (tensor, seeds, head bounds).
    """
    # Stream orientation comes from the BLOCKS, not the handle shape: GGUF
    # stores dims reversed for bucket-packed tables, so a handle shape of
    # (dims, buckets) means blocks arrive as (buckets, dims).  Derive cols
    # from the first block, rows from the element count.
    # Stream orientation comes from the BLOCKS, not the handle shape: GGUF
    # stores dims reversed for bucket-packed tables, so a handle shape of
    # (dims, buckets) means blocks arrive as (buckets, dims).  Derive cols
    # from the first block, rows from the element count.  The factory is
    # restartable by contract (fresh iterator per call), so the peek call
    # is simply discarded — every pass below starts its own fresh walk.
    # The first block is copied: a loader that reuses one block buffer
    # across yields would otherwise corrupt it during the real walks.
    first_block = next(blocks_factory())
    cols = int(first_block.shape[1])
    rows = (int(shape[0]) * int(shape[1])) // cols
    total = float(rows * cols)
    if rows * cols != int(shape[0]) * int(shape[1]):
        raise ValueError(
            f"streaming shape mismatch for {name!r}: "
            f"blocks {first_block.shape} vs shape {shape}"
        )

    bounds: list[int] | None = list(head_bounds) if head_bounds else None
    use_heads = bounds is not None and cols <= _GRAM_MAX_DIM
    heads: list[dict[str, Any]] = []
    if use_heads and bounds is not None:
        ends = bounds[1:] + [rows]
        for idx, start in enumerate(bounds):
            if start >= rows:
                break
            heads.append({
                "start": start,
                "end": min(ends[idx], rows),
                "cols": cols,
                "record_name": f"{name}.h{idx}",
                "gram": np.zeros((cols, cols), dtype=np.float64),
                "s1": 0.0, "s2": 0.0, "sabs": 0.0, "amax": 0.0, "m2": 0.0, "m4": 0.0,
                "n_rows": 0, "n": 0, "n3": 0, "n4": 0, "n6": 0, "n_zeros": 0,
            })

    def head_slices(block: np.ndarray, block_lo: int) -> Iterator[tuple[dict[str, Any], np.ndarray]]:
        """Yield (head, sub_block) splitting a block at head boundaries."""
        if not heads:
            return
        block_hi = block_lo + block.shape[0]
        for head in heads:
            hs, he = head["start"], head["end"]
            if block_hi <= hs or block_lo >= he:
                continue
            lo_c = max(block_lo, hs) - block_lo
            hi_c = min(block_hi, he) - block_lo
            yield head, block[lo_c:hi_c]

    # ── Pass 1: mean, E[x²], absmax/absmean, Gram/head-Gram, row-amax, sample
    # (s1 + s2 give mean/std/m2 exactly; pass 2 then only computes the 4th
    # moment + outliers — 2 tensor walks instead of 3 for the scalar stats)
    s1 = 0.0
    s2 = 0.0
    sabs = 0.0
    amax = 0.0
    gram = np.zeros((cols, cols), dtype=np.float64) if cols <= _GRAM_MAX_DIM else None
    row_amax_exact = rows <= _ROW_SAMPLE_CAP
    row_stride = max(1, -(-rows // _ROW_SAMPLE_CAP))
    row_amax_parts: list[np.ndarray] = []
    elem_stride = max(1, -(-int(total) // _SAMPLE_CAP))
    sample_parts: list[np.ndarray] = []

    lo = 0
    for block in blocks_factory():
        r = block.shape[0]
        for chunk in _chunks(block):
            s1 += float(chunk.sum())
            s2 += float((chunk * chunk).sum())
            a = np.abs(chunk)
            sabs += float(a.sum())
            if a.size:
                amax = max(amax, float(a.max()))
        if gram is not None:
            b64 = block.astype(np.float64)
            gram += b64.T @ b64
        for head, sub in head_slices(block, lo):
            sub64 = sub.astype(np.float64)
            head["gram"] += sub64.T @ sub64
            head["n_rows"] += sub.shape[0]  # buckets (for the record shape)
            for chunk in _chunks(sub):
                head["s1"] += float(chunk.sum())
                head["s2"] += float((chunk * chunk).sum())
                head["n"] += chunk.size  # ELEMENTS: denominators need these
                a = np.abs(chunk)
                head["sabs"] += float(a.sum())
                head["amax"] = max(head["amax"], float(a.max()))
        rmax = np.abs(block).max(axis=1).astype(np.float32)
        row_amax_parts.append(rmax if row_amax_exact else rmax[::row_stride])
        # strided element sample — MUST copy: the view would pin the whole
        # block's |x| buffer (hundreds of MB retained across a long walk)
        flat = np.abs(block.reshape(-1))
        sample_parts.append(flat[::elem_stride].copy())
        lo += r

    mean = s1 / total
    m2 = s2 / total - mean * mean
    m2 = max(m2, 0.0)  # fp64 cancellation guard
    absmean = sabs / total
    sample = np.concatenate(sample_parts) if sample_parts else np.zeros(0, dtype=np.float32)
    del sample_parts  # free the parts before the quantile's internal copies
    pcts = np.quantile(sample, (0.5, 0.9, 0.99, 0.999, 0.9999)).tolist()
    p50 = pcts[0]
    std = float(np.sqrt(m2))

    # ── Spectral ───────────────────────────────────────────────────────────
    if gram is not None:
        eig = np.linalg.eigvalsh(gram)
        eig = np.clip(eig[::-1], 0.0, None)
        spectrum = np.sqrt(eig)
    else:
        spectrum = _spectrum_halko(blocks_factory, rows, cols, STREAM_RSVD_K, svd_seed)

    # ── Pass 2: m4 (needs pass-1 mean), col-amax, outliers, sparsity,
    # per-head m4/outliers — ONE walk (everything here needs pass-1 scalars
    # only, so merging the old passes 2+3 halves the I/O on giant tensors)
    m4 = 0.0
    col_acc = np.zeros(cols, dtype=np.float64)
    row_offset = 0
    out3 = out4 = out6 = 0.0
    n_sparsity = 0
    n3 = n4 = n6 = 0
    for block in blocks_factory():
        col_acc = np.maximum(col_acc, np.abs(block).max(axis=0).astype(np.float64))
        for chunk in _chunks(block):
            d = chunk - mean
            dd = d * d
            m4 += float((dd * dd).sum())
        if std > 0:
            d = np.abs(block - mean)
            n3 += int(np.count_nonzero(d > 3 * std))
            n4 += int(np.count_nonzero(d > 4 * std))
            n6 += int(np.count_nonzero(d > 6 * std))
        n_sparsity += int(np.count_nonzero(np.abs(block.reshape(-1)) < 1e-3))
        for head, sub in head_slices(block, row_offset):
            hmean = head["s1"] / head["n"] if head["n"] else 0.0
            hm2 = max(head["s2"] / head["n"] - hmean * hmean, 0.0) if head["n"] else 0.0
            for chunk in _chunks(sub):
                d = chunk - hmean
                dd = d * d
                head["m4"] += float((dd * dd).sum())
            if hm2 > 0:
                hstd = float(np.sqrt(hm2))
                d = np.abs(sub - hmean)
                head["n3"] += int(np.count_nonzero(d > 3 * hstd))
                head["n4"] += int(np.count_nonzero(d > 4 * hstd))
                head["n6"] += int(np.count_nonzero(d > 6 * hstd))
            head["n_zeros"] += int(np.count_nonzero(np.abs(sub.reshape(-1)) < 1e-3))
            head["_hm2"] = hm2  # stash for kurtosis below
        row_offset += block.shape[0]

    if std > 0:
        out3, out4, out6 = n3 / total, n4 / total, n6 / total
    sparsity = n_sparsity / total
    m4 /= total
    kurtosis = float(m4 / (m2 * m2) - 3.0) if m2 > 0 else -3.0

    # ── Channel ratios ─────────────────────────────────────────────────────
    row_amax = np.concatenate(row_amax_parts) if row_amax_parts else np.zeros(0)
    row_med = float(np.median(row_amax)) if row_amax.size else 0.0
    row_ratio = float(row_amax.max()) / row_med if row_amax.size and row_med > 0 else float("nan")
    col_med = float(np.median(col_acc))
    col_ratio = float(col_acc.max()) / col_med if col_med > 0 else float("inf")

    # ── Spectral derivatives ───────────────────────────────────────────────
    spectral_norm = float(spectrum[0]) if spectrum.size else float("nan")
    effective_rank = float(entropy_rank(spectrum)) if spectrum.size else float("nan")
    sv_decay = float(spectrum[-1] / spectrum[0]) if spectrum.size and spectrum[0] > 0 else float("nan")
    frobenius = float(np.sqrt((spectrum**2).sum())) if spectrum.size else float("nan")
    if spectrum.size == 1:
        stable_rank = 1.0  # rank-1 by construction (same 1-D fix as StableRank)
    else:
        stable_rank = (
            float(np.log1p((frobenius / spectral_norm) ** 2))
            if spectrum.size and spectral_norm > 0
            else float("nan")
        )
    kernel_norm = frobenius  # non-4-D fallback, same as KernelNorm

    # ── Per-head records ───────────────────────────────────────────────────
    for head in heads:
        hm2 = head.pop("_hm2", 0.0)
        n_el = float(head["n"])
        head["kurtosis"] = (
            float(head["m4"] / n_el / (hm2 * hm2) - 3.0)
            if n_el and hm2 > 0
            else -3.0
        )
    head_records = [_head_stats_from_gram(head["gram"], head) for head in heads]
    for head, rec in zip(heads, head_records, strict=True):
        n_el = float(head["n"])
        rec.outlier_3s = head["n3"] / n_el if n_el else float("nan")
        rec.outlier_4s = head["n4"] / n_el if n_el else float("nan")
        rec.outlier_6s = head["n6"] / n_el if n_el else float("nan")

    # ── Pass 4 (opt-in): measured RTN-SQNR (INT8 per-row + FP8 global) ─────
    # Requires the exact row-amax vector from pass 1; sqnr_int4_g128 is NaN
    # in the streaming path (group scales need a (rows × groups) accumulator
    # that does not fit the sampling caps for genuinely giant tensors).
    sqnr_int8 = sqnr_fp8 = float("nan")
    if quant_probe and row_amax_exact and amax > 0 and not heads:
        import ml_dtypes

        scale_row = np.maximum(row_amax.astype(np.float64), 1e-12) / 127.0
        scale_fp8 = max(amax, 1e-12) / 448.0
        off = 0
        sig = err8 = errf8 = 0.0
        for block in blocks_factory():
            w = block.astype(np.float64)
            r = w.shape[0]
            sl = scale_row[off : off + r, None]
            wq8 = np.round(w / sl).clip(-127, 127) * sl
            v = np.clip(w / scale_fp8, -448.0, 448.0)
            wqf = v.astype(ml_dtypes.float8_e4m3fn).astype(np.float64) * scale_fp8
            sig += float((w * w).sum())
            err8 += float(((w - wq8) ** 2).sum())
            errf8 += float(((w - wqf) ** 2).sum())
            off += r
        sqnr_int8 = _sqnr_from(sig, err8)
        sqnr_fp8 = _sqnr_from(sig, errf8)

    main = TensorStats(
        name=name,
        shape=shape,
        frobenius=frobenius,
        spectral_norm=spectral_norm,
        effective_rank=effective_rank,
        stable_rank=stable_rank,
        kurtosis=kurtosis,
        sparsity=sparsity,
        kernel_norm=kernel_norm,
        sv_decay=sv_decay,
        row_amax_ratio=row_ratio,
        col_amax_ratio=col_ratio,
        mean=mean,
        std=std,
        absmax=amax,
        absmean=absmean,
        p50=pcts[0],
        p90=pcts[1],
        p99=pcts[2],
        p999=pcts[3],
        p9999=pcts[4],
        outlier_3s=out3,
        outlier_4s=out4,
        outlier_6s=out6,
        dyn_range=amax / p50 if p50 > 0 else float("inf"),
        sqnr_int8_ch=sqnr_int8,
        sqnr_int4_g128=float("nan"),
        sqnr_fp8_e4m3=sqnr_fp8,
        streamed=True,
        expert_id=expert_id,
    )
    return main, head_records
