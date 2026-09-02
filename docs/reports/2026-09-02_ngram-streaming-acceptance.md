# n-gram Table Streaming — Acceptance Report

**Date:** 2026-09-02
**Feature:** streaming statistics for giant tensors
(`docs/2026-09-01_ngram-table-streaming.md`, proposal)
**Target:** Unsloth Qwen3.8-Flash-Next UD-IQ4_XS GGUF (3 shards, 1224 file
tensors) — `per_layer_token_embd.weight`, the 51.2B-parameter n-gram table
(320,001,536 buckets × 160 dims, IQ4_NL, 28.8 GB quantized payload).

## 1. Verdict

**Accepted.** The streaming path scans the giant tensor end-to-end on this
machine (125 GiB RAM) with bounded heap, exact per-head spectra, and sane
per-head scalars. Two real-model bugs were found and fixed during acceptance
(§4); both are now pinned by tests.

## 2. Measured results (final run, after fixes)

- Handles: 74,808 (1224 file tensors; MoE 3-D expert tensors split per-expert,
  expert axis = 512 — the handle count is expected to exceed the file count)
- Head bounds: 16 segments from KV `qwen4exp.ple.head_offsets`
  `[0, 20000003, …, 300001275]`, last head extends to the 320M bucket end
- Walks: **3** (scalars+Gram+sample / m4+outliers+sparsity / quant probe off)
- Wall: **66.4 min** for the giant tensor alone (I/O-bound on mmap;
  first pass reads the 28.8 GB cold, later passes are page-cache-warm)
- Heap: bounded — the 29.3 GiB process RSS is the **file-backed page cache**
  of the 28.8 GB mmap; the acceptance memory criterion is RssAnon ≪ 16 GiB,
  which holds (per-block float32 workset ≈ tens of MiB)

### Table-level statistics (deterministic across runs)

| stat | value |
|---|---|
| spectral_norm | 164.74 |
| stable_rank | 4.71 |
| effective_rank | 159.9 |
| sv_decay (σ_min/σ_max) | 0.802 |
| sparsity | 0.0883 |
| kurtosis | 0.09 |
| dyn_range | 17.6 |
| outlier_3s / 4s | 0.0032 / 0.0001 |
| p50 / p99 / p9999 | 0.0051 / 0.0199 / 0.0306 |
| absmean / std | 0.00605 / 0.00763 |

### Per-head records (16)

- spectral_norm spread **63.5 .. 91.4** (head 15 is the outlier — 40% above
  the pack; its sv_decay 0.342 vs. ~0.48 for heads 0–2)
- stable_rank spread **3.19 .. 3.81**
- sparsity ≈ 0.086–0.090 per head (fraction, element semantics)
- kurtosis ≈ 0.0–0.2 per head (non-degenerate — see §4 bug 1)

### Reading (embedding vs. lookup)

The table sits in a **mixed regime, leaning embedding-like**: near-full
spectrum utilization (effective_rank 159.9/160, sv_decay 0.80 — no
rank-isolated spikes), low stable_rank 4.7 (energy concentrated in few
directions), only 8.8% zeros, no kurtosis blowup. The 16 heads are
individually measurable and differ meaningfully in spectral_norm — per-head
records are load-bearing, not decoration.

## 3. Cost model (verified vs. proposal §4)

- Per-16k-bucket block dequant (IQ4_NL): **~0.072 s** → ~19.5k blocks ≈
  23 min per cold walk; warm walks ≈ 0.09 s/block
- Gram GEMM per block (16384×160, fp64): ~3 ms — negligible vs. dequant
- 3 walks → 66 min; the proposal's "minutes" estimate was optimistic by ~10×
  for cold storage, right for page-cached re-walks. Pass 2+3 of the original
  design were merged into one walk (both depend only on pass-1 scalars),
  cutting a walk for free.

## 4. Bugs found by acceptance (fixed + pinned)

1. **Per-head element semantics** — `head["n"]` counted rows (buckets) while
   every accumulator (s1/s2/n_zeros/n3..n6) counted elements: per-head
   sparsity came out at 14.46 (> 1!) and kurtosis collapsed to the −3.0
   degenerate fallback. Fixed: element counts for denominators, `n_rows`
   kept for the record shape. Pinned by
   `TestPerHeadRecords::test_head_scalars_use_element_counts`.
2. **KV decode / shard placement** — `ple.head_offsets` lives only in the
   metadata shard (shard 1); reading it after the shard loop saw shard 3's
   reader (no field) → `ngram_head_bounds = None`. Also `field.data` indexes
   ALL array elements (no type part to skip): the `[1:]` slice dropped the
   leading 0. Fixed: per-shard read + full `data` iteration. Verified against
   the real file before/after.
3. **Stream orientation** — handle shape is GGUF file order (dims, buckets)
   = (160, 320M); the streaming code derived rows/cols from the handle shape
   and crashed in Halko. Fixed: cols from the first block, rows from the
   element count (`streaming.py` — blocks are (buckets, dims)).
4. **Restartable-factory contract** — an earlier repair introduced a one-shot
   iterator shared between the scalar pass and the Halko fallback (Halko
   would have silently seen one block). Fixed: peek-and-discard orientation
   probe; every pass calls `blocks_factory()` fresh.

## 5. Determinism

The main-table stats were **bit-identical between the two acceptance runs**
(spectral_norm 164.738, all scalars equal), and the merged pass-2 walk
changed no numbers — cross-path identity is pinned by
`tests/test_streaming_stats.py` (F-1 tolerance rel 1e-4 on σ, exact-sum
equal_nan).

## 6. Full-model scan acceptance (supplement, same day)

The whole 74,808-handle scan completed end-to-end (fingerprint 85.6 MB,
main + expert-panel fields, manifest; journal discarded on success).

- Wall: ~2.5 h total — pooled phase ~35 min (18 spawn workers, ~30
  tensors/s), big-serial phase ~45 min (two 2.4 GiB tensors, exact SVD,
  BLAS-capped serial), giant streaming ~66 min, fields/manifest minutes.
- Giant-table stats in `fingerprint.json` are **bit-identical** to the
  standalone §2 runs (third independent confirmation of determinism).
- Two incidents on the way, both fixed and committed:
  1. **Kernel OOM 02:16** — pool workers accumulated 10–14 GiB anon RSS
     each for ≤120 MiB live tasks (glibc malloc arena fragmentation over
     ~75k varying-size numpy tasks; glibc never returns it). Post-mortem
     fix (commit `d9bd366`): `malloc_trim` after every pool task (measured
     avg worker RSS after fix: **81 MiB**), payloads >256 MiB excluded
     from the pool (serial tail) so inflight bytes stay bounded.
  2. **Lost head records** — the completed scan's fingerprint had zero
     `.h{n}` records: `record()` journaled them but the success path built
     the fingerprint from the handle-indexed stats list alone. Fixed in
     the same commit (`extra_stats` list; scaling/rasterize still see
     handle records only, so headless scans stay byte-identical); pinned
     by `TestScanStreamingIntegration::test_head_records_reach_fingerprint_and_survive_resume`.
     The scan output at `output/flash-next-ud-iq4xs-accept/` predates the
     fix — a re-run adds the 16 head records (~2.5 h, resumable).

## 7. Follow-ups

- Re-run the full scan to capture the 16 head records in the artefacts
  (then import into the server DB).
- Records tab / scatter views pick head records up as ordinary tensors
  with `.h{n}` names.
- `docs/2026-09-01_ngram-table-streaming.md` §5 open questions 1–3 are
  answered by this report (shapes/KV: §2; quant type: IQ4_NL; axis order:
  bucket-major, handled by the orientation probe).
