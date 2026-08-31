# alesha-pro/atlas — Analysis and Comparison with weight-atlas

> Status: Analysis | Date: 2026-08-31 | Reviewed at commit: github.com/alesha-pro/atlas (MIT)
> Scope: what the tool does, feature inventory, concept-by-concept comparison with
> weight-atlas, and a concrete assessment of what is worth adopting.

---

## 1. Executive summary

`alesha-pro/atlas` ("Weight Atlas", hosted at atlas.alesha.pro) is a
**single-model, interactive, quantization-focused** weight explorer: one
pan/zoom HTML canvas where every tensor of a checkpoint appears as a cell,
colored by a switchable metric, with the core question *"what in this model can
be compressed losslessly, what will fall apart, and for what reason."* Its
crown jewel is **measured (simulated, reference-free) quantization SQNR** per
tensor — INT8 per-channel, INT4 group-128, FP8 e4m3 — not estimates.

It is complementary to weight-atlas rather than competing:

- weight-atlas: multi-model **comparison** pipeline (scan → deterministic
  topographic fields → render → compare → paired analysis → query API), broad
  format support (GGUF k-quants, MXFP4, safetensors, PyTorch/BDH),
  spec-driven name mapping, byte-identical outputs.
- atlas (alesha-pro): single-model **deep-dive** browser, richer per-tensor
  distribution metrics, measured RTN quantizability, hybrid-attention live
  probing (Gated DeltaNet state, write gates, half-life), interactive
  metric×metric scatter — but only bf16 safetensors input, GPU-only scanner,
  unseeded (non-deterministic) subsampling, and no comparison or pipeline
  story.

Adoption recommendation (§7): take the **metrics** (sv_decay, channel-outlier
ratios, percentile/outlier ladder, and especially the RTN-SQNR simulation —
re-implemented deterministically in NumPy, including a software FP8-e4m3
encoder) and two **UI ideas** for our query API (metric×metric scatter,
extremes leaderboard). Do not take their scanner architecture, classification,
or data format.

The showcased model, "Qwen3.8-27B", is from the **same family** as our pending
Qwen3.8-Flash-Next target — their linear-attention (GDN) scan patterns
(`in_proj_a/b/z`, `conv1d`, `dt_bias`/`A_log`) and live-probing design are
directly relevant to that upcoming work.

---

## 2. What the tool is

Architecture (≈5,500 LOC total, MIT):

| Component | Language | Role |
|---|---|---|
| `scripts/weight_atlas.py` (203 LOC) | Python + torch | Streams safetensors shards, one GPU pass per tensor → JSONL record per tensor |
| `src/` (≈4,600 LOC TypeScript) | Vite, zero runtime deps | Static single-page canvas: regions (intro, architecture, wall, links/scatter, depth, herbarium/treemap, records, dossier*, live*) — *optional, gated on extra JSON files |
| `scripts/atlas_live.py` (770 LOC) | Python + torch | "Living model": forward-pass hooks over a calibration mix (en/code/agent) — residual flow, activation SQNR, attention stats, linear-attention state, per-layer fragility |
| `scripts/reduce_live.py` (237 LOC) | Python | Folds live artifacts (+ external "carve" parquet files for FFN-neuron fingerprints) into `live.json` |

Data model: `public/models/<slug>/atlas.jsonl` (one JSON object per tensor) +
`manifest.json`; optional `dossier.json` (hand-written passport facts, block
diagrams as generic node/edge JSON) and `live.json`. Layers, groups, wall rows
and metric ranges are **derived from the data at load time** — adding a model
is a scan + manifest entry, no code changes (same philosophy as our
spec-driven slots).

Reference result: Qwen3.8-27B (27.78B params, 1199 tensors) scanned in 116 s
on one GPU. Interface is bilingual EN/RU.

---

## 3. Scanner metrics (their §"What the data is")

Per tensor (float32 on GPU):

**Distribution shape** — mean, std, absmax, absmean; percentiles
p50/p90/p99/p999/p9999 of |w| (subsampled to 16M elements above that);
kurtosis, skew; sparsity (`|w| < 1e-6`); outlier fractions beyond 3σ/4σ/6σ of
std; dynamic range (`absmax / p50`).

**Histogram** — 29 bins of log₂|w| over `[2⁻²⁴, 2⁴]`, real shares (used by the
inspector to draw the actual distribution).

**Channel structure** (2-D only) — `row_amax_ratio` = max/median of per-row
amax, `col_amax_ratio` likewise, `row_amax_p99`. This is the **outlier-channel
problem** made into a number: it predicts whether per-channel quantization
will be healthy or whether one channel destroys the scale.

**Measured quantizability** (2-D only) — the signature feature. Real
round-to-nearest simulation, SQNR = 10·log₁₀(‖W‖²/‖W−Ŵ‖²):

| Format | Scheme |
|---|---|
| `sqnr_int8_ch` | INT8 symmetric, per-row (last dim) amax/127 |
| `sqnr_int4_g128` | INT4 symmetric, per 128-group amax/7 (k % 128 ≠ 0 → N/A) |
| `sqnr_fp8_e4m3` | FP8 e4m3, global amax/448 scale, hardware dtype cast |

Note honestly: this is a **canonical RTN baseline**, no zero-point, no
GPTQ/AWQ/optimal-scaling — it measures the *floor* damage of the standard
formats, which is exactly the right thing for a "will it survive" atlas.

**Spectrum** (2-D, min dim ≥ 16) — `svd_lowrank(q=32, niter=2)`: top-16
singular values, `stable_rank = ‖W‖²_F / σ₁²`, `sv_decay = σ₃₂/σ₁` (tail of the
truncated spectrum, not the true tail).

**Classification** — flat regex list: dense (embed/lm_head/norms/q/k/v/o),
MoE (router/shared/experts), hybrid linear attention (`in_proj_a/b/z`,
`conv1d`, `dt_bias|A_log` — Gated DeltaNet / Qwen3-Next aware), vision
(`visual.`), MTP handled by name prefix. Unknown → `other`. Layer index from
`layers.(\d+)`, expert index from `experts.(\d+)`.

**Discipline worth copying:** metrics that do not apply (1-D tensors vs.
2-D-only metrics) render as *not applicable*, never as zero. This matches our
NaN discipline exactly.

---

## 4. Live capture (their "living model")

`atlas_live.py` runs the bf16 checkpoint over a domain-balanced calibration
mix (english / code / agent traces) with forward hooks, one JSON per probe:

| Probe | What is measured |
|---|---|
| `flow` | Residual-stream RMS per layer, per-layer contribution Δh, **activation outlier-channel ratio** (max/median of mean-|act| per channel, count > 5×median), and **activation quantizability**: INT8 per-token dynamic + FP8 e4m3 SQNR at the input of every large Linear |
| `attn` | Attention entropy (normalized by ln L), sink mass (token 0), attention-mass decay profile over distance (edges 1…2048), output-gate openness; showcase attention maps on a fixed prompt, image-token attention share |
| `linattn` | GDN specifics: write-gate β (sigmoid mean), decay g from A_log/dt → **half-life of the recurrent memory**, state RMS after the window — per layer, per domain |
| `frag` | **Per-layer fragility**: KL(base ‖ INT4-g128 of all Linears in that one layer) + logit cosine, one forward per layer (weights temporarily fake-quantized) |

Plus a showcase where the model attends to a screenshot of the atlas itself.
The FFN-neuron fingerprint ("a million FFN neurons by domain") comes from
external carve artifacts (parquet), not from this repository.

This is the strongest part of the project conceptually: it connects **weights
→ activations → behavior** for exactly the hybrid-attention architectures
(Qwen3-Next lineage) where classical attention atlases are blind.

---

## 5. UI regions (their canvas)

| Region | Mechanic | weight-atlas counterpart |
|---|---|---|
| start | intro card with model params, full/linear rhythm | — (we have no narrative intro) |
| architecture | group map (embed → layers → attn/MLP → head, vision, MTP) with live param shares | fingerprint `model` + mapping coverage; no dedicated view |
| **wall** | every tensor: column = layer, row = role, rows aligned across layers; ◆ marks full-attention layers in hybrid stacks; metric switch recolors everything live; click → inspector | our raster = same (layer × slot) grid, but as static topographic TIFF/sheet with hillshade; rows=slots too; we have no interactive recoloring, they have no topology render |
| **links (scatter)** | any metric × any metric, one dot per 2-D tensor, presets (kurt→INT4, hot channels→INT4, rank→INT4, size→INT4) | nothing — our query API serves the per-tensor records but no scatter view exists |
| depth | per-layer mean of the current metric down the stack + quarters | implicit in our raster rows; no line-chart view |
| herbarium | treemap, area = parameter count | — |
| **records** | leaderboard of extremes: worst/best INT4, worst INT8, most kurtotic, hottest channel, widest dyn range, lowest/highest rank, sparsest, biggest | nothing dedicated; our query API could derive these trivially |
| inspector | plain-language verdict from real numbers, histogram, spectrum, channel ratios, percentile ladder | our model detail tabs (statistics table, LLM query API); theirs is rule-based templated text, ours can be an actual LLM |
| live / dossier | optional regions (see §4) | our activity capture (M8) + job-rendered sheets |

Engineering notes: no framework, hand-rolled pan/zoom canvas, i18n table,
SVG charts. Determinism is explicitly not a goal on the scan side — the
percentile subsampling uses `torch.randint` **without a generator** and the
SVD is `svd_lowrank(niter=2)` unseeded: two scans of the same checkpoint do
not produce identical JSONL. (Contrast: our determinism contract.)

---

## 6. Concept-by-concept comparison

| Axis | alesha-pro/atlas | weight-atlas |
|---|---|---|
| Core question | "What survives quantization, tensor by tensor?" | "What is this model's weight topology, and how do two models differ?" |
| Models | one at a time | many, plus compare/paired deltas |
| Input formats | bf16 safetensors only | safetensors (incl. MXFP4 pairs), GGUF (K-quants, Q8_K, I8–I64/F64, MXFP4), PyTorch `.pt` (BDH layout) |
| Quantized checkpoints | cannot scan them (simulates instead) | dequantizes and *maps the actual recipe* (fingerprint `quantization`) |
| Quantization analysis | **measured RTN SQNR** per tensor, reference-free, 3 formats | qimpact (M9): paired impact of an *actual* recipe vs. reference scan; edit signatures (M9) |
| Metrics per tensor | ~20 incl. percentiles, skew, outlier fractions, histogram, channel ratios, SQNR×3, sv_top/decay | 7 core stats (frobenius, spectral, stable_rank, effective_rank, kurtosis, sparsity, kernel_norm) + MoE/vision channels |
| stable_rank definition | ‖F‖²/σ₁² (= our pre-log1p argument) | log1p((‖F‖/σ₁)²) — monotone transform, same ordering |
| Name mapping | one flat regex list (dense/MoE/GDN/vision) | spec-driven, per-convention (HF, GGUF, MoE, Kimi-MLA, Qwen3-Next hybrid, VLM, BDH lattice), canonical + fallback |
| Visualization | interactive canvas, live metric switching, click-through inspector | deterministic topographic sheets (hillshade/tint/contours), 3-D terrain (Blender), fractal render, static by design |
| Behavior/live | hooks: flow/actq/attn/linattn/frag per domain | activity capture (M8): per-layer reductions, torch-state save/restore, attention-mask-aware; no activation-SQNR/fragility/GDN-state probes yet |
| Comparison | none | compare (M4) + paired (M9) pipelines, Δ-sheets, hotspot ranking |
| Serving | static files | FastAPI UI, job queue, LLM query API (v0.2) |
| Determinism | not pursued (unseeded RNG, GPU) | contract: byte-identical artefacts |
| Determinism of adoption interest | — | any adopted metric must be re-derived deterministically (seeded) |
| Extensibility | derive-everything-from-data UI; scan = one script | plugin registry (loaders/stats/renderers), spec-driven everything |

Where they are clearly ahead of us:

1. **Quantization layer** — measured weight-side RTN-SQNR per tensor and
   (live) activation SQNR and per-layer KL fragility. Our qimpact answers a
   different (paired) question; we have no reference-free "canonical damage
   floor" per tensor.
2. **Distribution-shape richness** — percentiles, outlier fractions, channel
   amax ratios, histograms. We compress the distribution into kurtosis/sparsity
   and lose the outlier-channel story entirely.
3. **Hybrid-attention live probing** — GDN write gates, memory half-life,
   state RMS. Directly relevant to Qwen3.8-Flash-Next.
4. **Interactive exploration** — metric×metric scatter and extremes are cheap
   wins on top of our existing query API.

Where we are ahead (and should not regress): format breadth, multi-model
comparison, determinism, name-mapping breadth (their scanner maps our vision
slots as `visual.`-only and would misfile DeepSeek MLA/Kimi tensors), the
render pipeline, the query API, and job orchestration.

---

## 7. Adoption proposal (prioritized)

### P1 — scan-side metrics (pure NumPy, deterministic, cheap)

1. **`sv_decay`** — σ_k/σ₁ from the shared per-tensor spectrum we already
   compute once. ~3 lines in `stats/norms.py`; free for every future scan.
2. **Channel-outlier ratios** — `row_amax_ratio`, `col_amax_ratio` (max/median
   of per-row/per-column amax) as 2-D stats. One-liners in NumPy; the
   outlier-channel story is currently invisible in our fingerprints.
3. **Outlier fractions + percentile ladder** — `outlier_3s/4s/6s`,
   `p50/p90/p99/p999/p9999`, `dyn_range`. Trivially seeded: `np.quantile` on
   the full array when numel ≤ 16M, else a *seeded* subsample
   (`np.random.default_rng(spec.seeds.<key>)`) — fix their determinism gap.
4. **RTN-SQNR stats** (`sqnr_int8_ch`, `sqnr_int4_g128`, `sqnr_fp8_e4m3`) —
   the flagship. All three are O(numel) NumPy:
   - INT8/INT4: per-row/per-group amax scaling + `np.round` + clip — direct
     ports of their scheme.
   - FP8 e4m3: NumPy has no fp8 dtype; simulate by bit-twiddling (exponent
     clamp to e4m3 range + 3-bit mantissa rounding) — ~25 lines, testable
     against known e4m3 tables.
   - Cost: 3 passes over weights. For the 51B Qwen3.8 target that is ~200 GB
     of streaming I/O per format — make the block opt-in per scan
     (`--quant-probe`) or gate on tensor size, and record the probe set in
     the fingerprint.
   - Once present: the query API gets the scatter axes, the raster can color
     by `sqnr_int4_g128` via a spec channel, and **compare** gains a free
     delta dimension (fragility migration between models).

### P2 — UI over the existing query API

5. **Scatter view** (`/models/{id}` new tab): x/y = any two fingerprint
   fields, one dot per tensor — pure frontend over the existing paginated
   records; presets map to their kurt→INT4 style once P1.4 lands.
6. **Extremes leaderboard**: `query.py` already caches per-tensor records;
   add top-N queries (max/min per field) and a small "records" card grid.

### P3 — activity extensions (GPU, opt-in)

7. **Activation SQNR probe** in `capture_activity`: INT8-per-token + FP8 at
   Linear inputs, aggregated per (layer, site) — their `flow` collector is a
   good blueprint; our torch-state save/restore and mask handling already
   exist.
8. **Per-layer fragility (KL of INT4-ing one layer)** — expensive (one
   forward per layer) but the single most decision-relevant number for
   mixed-precision recipes; same module, same determinism discipline
   (fixed calibration text, seeded sampling).
9. **GDN/linear-attention state probe** (write gate β, half-life, state RMS)
   — schedule for the Qwen3.8-Flash-Next work specifically; our hybrid rules
   already map `ssm_*`/linear-attention slots, so weight-side naming is done.

### Not adopted, deliberately

- Their scanner architecture (monolithic script, GPU-only, streaming JSONL
  with inline flush) — our loader registry + handles + parallel stats is
  strictly more capable and deterministic.
- Their classification table — subsumed by our name_map; their GDN patterns
  (`in_proj_a/b/z`, `dt_bias|A_log`) are already covered by our
  `_HF_HYBRID_RULES` / `ssm_*` slots.
- Their data format — our fingerprint JSON is the established contract with
  query API + compare; JSONL adds nothing we need.
- Their unseeded subsampling — violates the determinism contract; any port
  must seed.

### License note

MIT — code can be ported verbatim with attribution. The RTN-SQNR schemes
themselves are textbook (amax-symmetric quantization); an independent NumPy
implementation only needs a courtesy credit, e.g. in the module docstring and
here: metrics in §3/§4 of this analysis document the source of the schemes.

---

## 8. Verdict

Worth studying, and partly worth adopting: the project validates several of
our design choices independently (NaN ≠ zero, derive-structure-from-data,
slot-aligned grids) and contributes one genuinely missing capability —
reference-free, measured quantizability per tensor — plus a set of cheap
distribution metrics that make the outlier-channel problem visible. The
hybrid-attention live probes are a targeted, high-value extension for the
Qwen3.8-Flash-Next work. The single-model, non-deterministic,
bf16-only scanner architecture is not worth porting; ours answers broader
questions at higher engineering standards.
