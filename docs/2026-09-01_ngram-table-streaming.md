# Streaming Statistics for Giant Tensors (n-gram Embedding Tables)

> Status: Proposal | Date: 2026-09-01 | Target: Qwen3.8-Flash-Next scan (125B main + 51B n-gram embeddings)
> Depends on: IQ-quant family support (implemented, `fdd1115`); deterministic distribution stats (implemented, `07ac3a7`)

---

## 1. Problem

Flash-Next carries ~51B parameters of n-gram embedding tables. Depending on
how the tables are sharded, one or more single tensors will hold tens of
millions of rows × thousands of columns — tens of GiB of float32-equivalent
per tensor. The current scan pipeline **materializes every tensor fully**:

- `TensorHandle.load()` dequantizes the entire payload in one call
  (`gguf_loader._ensure` → `dequantize(self._data, ...)` allocates the full
  float32 array). A 51B-element table = **204 GiB fp32 → OOM at load**, before
  any statistic runs.
- Even materialized, the spectral path casts to float64 (`+2× memory`) and an
  exact SVD of a (V, D) giant matrix is `O(min²·max)` — astronomically out.
- `_BIG_TENSOR_THRESHOLD_BYTES` (1 GiB fp32) only steers *scheduling* (serial
  instead of pooled) — it does not change memory behavior.
- The seeded 16M-element percentile subsample (P1.3) runs **after** full
  materialization, so it does not help here either.

The measurement we actually want — *do n-gram tables behave spectroscopically
like embeddings (low stable rank, smooth decay) or like hash lookup tables
(high sparsity, spiky kurtosis, wide dyn range)?* — is exactly the one that
crashes today.

## 2. Why streaming + rSVD, not subsampling

**Row subsampling for the spectrum is wrong.** The singular values of a
row-subsampled matrix are not the singular values of the full matrix — for
structured tables (hash collisions, frequency-ordered buckets) the subsample
spectrum can be arbitrarily misleading. The question "embedding or lookup
table" lives precisely in the full-table spectral structure.

**Block-wise Halko rSVD gives the true top-k spectrum with bounded memory.**
The pipeline we already run for `min(m, n) > 512` (`spectrum.py:
_randomized_singular_values`) needs only three operations on the matrix:

1. `y = x @ ω` — (rows × k) auxiliary, k=16 → for a [25M, 2048] table:
   25M × 16 × 4 B = **1.6 GiB** (acceptable; the only large auxiliary)
2. `q, _ = qr(y)` — (25M × 16), small LAPACK call on the auxiliary
3. `b = qᵀ x`, then `svd(b)` — b is (k × cols) = tiny

All touch points on `x` are **matrix-vector/GEMM products over row blocks** —
if `x` is provided as a row-block iterator instead of an ndarray, no step ever
materializes the table. Deterministic (seeded ω, fixed block boundaries).

## 3. Design

### 3.1 Loader-level block streaming

GGUF K-quant/IQ rows dequantize **independently** (`data` is
`(rows, block_bytes)`; each row's blocks are self-contained: scale bytes +
index nibbles). New loader surface:

```python
def iter_row_blocks(handle: TensorHandle, rows_per_block: int) -> Iterator[np.ndarray]:
    """Yield float32 row blocks (rows_per_block, cols) without materializing
    the full tensor. Implementable for GGUF (per-row dequantize over the
    mmap'd payload) and safetensors (byte offsets); PyTorch/BFH follow."""
```

- GGUF: per-row-block dequantize over the mmap-backed uint8 payload — the
  block decoders already accept `(rows, block_bytes)` arrays; rows_per_block
  bounded (e.g. 16k rows ≈ 128 MiB fp32 at D=2048).
- safetensors: row-major fp16/bf16 → per-block `np.frombuffer` slices.
- Handles gain a lazy fast-path flag; `load()` stays as-is (full
  materialization) for every existing consumer — **no behavior change** for
  tensors that fit in RAM.

### 3.2 Streaming stats path

A dedicated runner for tensors above a *materialization* threshold
(`_STREAM_TENSOR_BYTES`, e.g. 8 GiB fp32 ≈ 2G elements — distinct from the
BIG/scheduling threshold), computing the full TensorStats record block-wise:

| Stat | Streaming strategy |
|---|---|
| frobenius / mean / std | chunked float64 accumulation (existing pattern) |
| percentile ladder (p50…p9999) | seeded strided sample across blocks (cap 16M) |
| outlier_3s/4s/6s | pass 2 after mean/std (existing two-pass shape) |
| row_amax_ratio / col_amax_ratio | per-row amax accumulated block-wise (rows×1 float64 vector is small); col-amax needs a cols-float64 max-vector update per block |
| sparsity / kurtosis / skew | chunked accumulation (existing) |
| **spectral_norm, effective_rank, stable_rank, sv_decay** | **block-wise Halko rSVD** (§2) — same k=16, q=2, seeded; same truncation semantics as the existing large-matrix path |
| kernel_norm | n/a for 1–2-D; block-wise per-channel for 4-D |

Numerical identity contract: streaming results for a tensor that *would*
materialize must equal the in-process path's results to float64-rounding
(accumulation order is block-major instead of whole-array — the existing
chunked accumulators already accept this; pinned by a cross-path test like
the process-pool one).

### 3.3 Integration

- `scan()` routes tensors ≥ `_STREAM_TENSOR_BYTES` to the streaming runner
  (serial, in-process, like today's big path — payload never enters the
  process pool).
- Checkpoint journal (3b73a34) covers streaming tensors for free: one
  TensorStats line per tensor, appended on completion.
- `--quant-probe`: RTN-SQNR works block-wise for INT8/INT4 (per-row/group
  scales need one amax pass first — two extra streaming passes). FP8 needs a
  global amax → also two passes. All bounded.
- Fingerprint: add `streamed: true` per giant tensor (honesty marker; also
  flags the top-k-approximation caveat for `sv_decay`).

## 4. Memory & time budget (example [25M, 2048] table, 51B elements)

| Quantity | Value |
|---|---|
| fp32 materialization (today) | 204 GiB → OOM |
| row block (16k rows) | 128 MiB |
| Halko auxiliary y (rows × 16) | 1.6 GiB |
| b (16 × 2048) | 128 KiB |
| streaming passes | 3–6 over ~68 GiB quantized payload (mmap, page-cached) |
| wall time (est.) | minutes, I/O-bound |

The auxiliary `y` (1.6 GiB) is the only full-column-dim allocation; for wider
tables (D=4096+) it scales linearly but stays ≪ materialization.

## 5. Open questions (need the real model)

1. **Actual tensor shapes/sharding**: one [25M, 2048] table or hundreds of
   smaller per-layer n-gram tables? The design handles both, but the
   threshold tuning and expected runtime differ by an order of magnitude.
2. **Quant type of the tables**: F16/BF16 (typical for embeddings) or
   quantized like the rest? (IQ family now supported either way.)
3. **Semantic axis order**: are table rows frequency-ordered (buckets)? If
   yes, row blocks are *not* i.i.d. samples for the percentile ladder — the
   strided sample must span blocks accordingly (it does: strided across the
   whole row space, not per-block).
4. **`iter_row_blocks` surface**: loader-plugin method vs. standalone util
   keyed on format. Loader-plugin method (registry-consistent), with a
   default implementation via `load()` for formats that lack block support.

## 6. Non-goals

- Exact full SVD of giant matrices — unnecessary for the question and
  physically impossible at this scale.
- Byte-layout statistics (§ measured on dequantized values; see analysis doc
  §3) — byte-level spectra would measure the serialization scheme.
- Parallelizing streaming across processes for a single giant tensor — the
  row-block passes are I/O-bound on one mmap; splitting adds coordination
  without wall-clock benefit.

## 7. Verification plan

1. Synthetic giant fixture: memory-mapped fake table (e.g. [2M, 2048] =
   16 GiB fp32-equivalent via a lazy loader) — streaming path completes
   without materialization (peak-RAM assertion via tracemalloc).
2. Cross-path numerical identity: streaming rSVD stats ≡ in-process rSVD
   stats on a [50k, 2048] matrix that *does* materialize (equal_nan compare,
   float64-rounding tolerance) — same discipline as the process-pool test.
3. Determinism: two streaming scans byte-identical.
4. Embedding-vs-lookup sanity: synthetic embedding matrix (low-rank + noise)
   vs. synthetic hash table (sparse spikes) produce the expected stable_rank
   / sparsity / kurtosis signatures — the measurement the whole feature
   exists for.
