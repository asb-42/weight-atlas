# weight-atlas — BDH (Dragon Hatchling) Continual-Learning Support

> Status: Implemented (preliminary) | Merged: 2026-08-31 (`c97340a`, `294a0c7`) | Breaking: No (additive)
> Reviewed 2026-08-31 (A0/Quinn): F-1 (§9.2 rewrite), F-3, F-4, F-5 applied; F-2 quantified in §7.3. Review record: `reports/2026-08-31_bdh-implementation-review-A0.md`.

---

## 1. Scope

This document describes the BDH (Dragon Hatchling, arXiv:2509.26507) support in
weight-atlas: why it does not fit the layer×slot raster model, the design
decisions behind the implementation, the route-lattice integration, how
comparability with other architectures is preserved, and the current
limitations.

BDH is a continual-learning (CL) research architecture: byte-level LMs trained
in sequential domain phases where later phases grow the model by appending new
neuron blocks (the *route lattice*) and train only those, while earlier
capacity stays frozen by construction. weight-atlas support targets the
**route-aware phase-growth training path** (Mechanism B/F lineage:
`grow_mult`, `route_aware`, `route_alpha=0.9` in the BDH reference code), with
the goal of making the frozen/trained split **visible weight-side** — a view
independent of behavioural routing measurements (P-Route/P-Det from
`eval_router`).

Implemented:

- PyTorch `.pt` checkpoint loader (pure-python, no torch dependency)
- BDH layout detection + three-granularity tensor expansion
- Route-lattice panels (heads × units) rendered through the existing expert-
  panel machinery
- Flat (1 × n_slots) visualization for non-layered models
- Compare/panel compatibility for phase checkpoints (same architecture, grown
  capacity)

Not implemented (out of scope for now, see §9): Blender/fractal renders for
lattice panels, paired/qimpact BDH recipes, activity capture, MLX-adjacent
BDH variants (`bdh_prime.py` checkpoint with extra attention weights).

---

## 2. Why BDH breaks the layer×slot model

The weight-atlas raster places per-tensor statistics into a
`(layer, slot)` grid: rows are transformer block indices, columns are spec
slots (`attn_q`, `mlp_up`, ...). BDH has **no layers** — a single
`(encoder, encoder_v, decoder)` weight triple is *shared across depth levels*
(the depth structure lives in state, not parameters), plus embedding, LM head
and RoPE frequencies. Mapped naively, every tensor lands on `layer=None`, which
the raster skips: an empty atlas.

Two consequences drive the design:

1. **The raster dimension must be replaced, not removed.** BDH's interesting
   axis is not depth but the *neuron lattice*: `N = mlp_internal_dim_multiplier
   × n_embd / n_head` neurons per head, where phase growth appends units along
   that axis. The natural tiling is one lattice unit = `n_embd // n_head`
   neurons per head (64 for the ~70M config — labeled "100M" in the BDH
   README, actual parameter count 69,473,792: `D=512, nh=8`), because growth
   adds exactly one multiplier unit = 64 neurons/head at a time and phase
   boundaries always land on that grid (train.py:130 of the reference code:
   per-head suffix `keep[n_old:] = 1.0`).
2. **Head-major vs neuron-major memory layout.** `encoder`/`encoder_v` are
   `(nh, D, N)` — head block is contiguous, the neuron axis is *last* — while
   `decoder` is `(nh·N, D)` with rows ordered `h·N + n` (head-major, contiguous
   neuron runs). Any per-(head, unit) slicing must treat these differently;
   getting it wrong silently computes statistics over mixed heads.

---

## 3. Data model: three granularities

For a BDH-layout checkpoint, the PyTorch loader expands each core tensor into
three granularities. The ~70M config (`nh=8, D=512, mult=88`) produces 2142
handles from 6 tensors:

| Granularity | Name pattern | Shape | expert_id | Feeds |
|---|---|---|---|---|
| Monolithic | `encoder`, `decoder`, ... | full tensor | — | flat field, fingerprint, whole-tensor stats |
| Per-head | `blk.{h}.{name}` | encoder: `(D, N)`; decoder: `(N, D)` | — | **main raster** as heads × slots |
| Per-unit | `{name}.u{u}.h{h}` | encoder: `(D, 64)`; decoder: `(64, D)` | `u` | route-lattice panels |

Non-core tensors (`embed.weight`, `lm_head`, `attn.freqs`, and any additional
attention weights from variant checkpoints) pass through monolithically and map
to the shared slots `embed`, `lm_head`, `rope_freqs` — deliberately reused, not
BDH-prefixed, so cross-architecture comparisons of these universal components
remain possible.

### 3.1 Layout detection

Detection runs per checkpoint from the unpickled `cfg` dict:

```
n_head, n_embd, mlp_internal_dim_multiplier present and positive, D % nh == 0,
encoder/encoder_v shape == (nh, D, N), decoder shape == (nh·N, D)
→ BDH layout; unit = D // nh, N = mult · unit
```

Anything else → plain passthrough (monolithic only). Detection is strict: no
heuristics on tensor names. A non-BDH `.pt` (e.g. the baseline `transformer.py`
checkpoints) never triggers expansion. When expansion triggers, the loader
records `bdh_n_heads`, `bdh_unit`, `bdh_mult`, `bdh_decoder_layout` in
`fingerprint.json` → `loader_metadata`.

### 3.2 Slice geometry (correctness-critical)

All unit statistics must slice the same neurons the trainer treats as a unit.
With element order:

- `encoder`/`encoder_v` `(nh, D, N)`: element `(h, d, i)` at flat index
  `h·D·N + d·N + i`. A **head** is the contiguous range `[h·D·N, (h+1)·D·N)`;
  a **unit** is a *column slice* `head[:, u·64:(u+1)·64]` — strided, materialized
  with `np.ascontiguousarray` (a copy; the storage cache stays shared).
- `decoder` `(nh·N, D)` head-major: head `h` is rows `[h·N, (h+1)·N)`, unit
  `(h, u)` is the **contiguous range** `[(h·N + u·64)·D, ...+64·D)`.

Both verified bit-exact (`np.array_equal`) against a real phase-3 checkpoint
for corner units (`u0/h0`, `u23/h3`, `u87/h7`) on both tensors.

### 3.3 Storage cache and determinism

Per-unit and per-head handles share an instance-level cache of each storage
entry as float32: the ZIP payload is read once per storage (`data/0..5`), the
full array is memoized, slices are views/copies of it. Peak RAM is the full
model in float32 (~450 MB for the ~70M checkpoint), not one materialization
per handle. Output is deterministic: same file → same slices → same stats, no
parallel-order-dependent aggregation. The monolithic handles deliberately do
*not* use the cache (single full-tensor read, `clear()` releases immediately).

---

## 4. Unpickler: no torch, no arbitrary imports

`data.pkl` is parsed with a minimal `pickle.Unpickler` subclass that
substitutes the handful of globals PyTorch checkpoints reference:

| Pickle global | Replacement |
|---|---|
| `torch.FloatStorage` (+ all `*Storage`) | stub class carrying `dtype` as class attribute |
| `torch._utils._rebuild_tensor_v2` | capture function: records `(zip_path, dtype, shape, offset, numel)` and returns `None` |
| `torch._utils._rebuild_parameter*` | passthrough (returns first arg) |
| `collections.OrderedDict` | stdlib |

`persistent_load` passes the storage tuple through unchanged
(`('storage', <stub>, key, device, numel)`) — the capture function unpacks it.

**Security:** `find_class` is a hard whitelist. Unknown globals raise
`pickle.UnpicklingError` instead of falling through to pickle's default import
machinery — the earlier no-op fallback for unknown `torch.*` imports (and the
stdlib fallthrough before that) would have allowed GLOBAL+REDUCE on arbitrary
callables (e.g. `os.system`) when scanning an untrusted file. Checkpoints are
model weights, but they arrive from the internet; the loader treats them as
input, not code. Pinned by `test_unpickler_rejects_non_whitelisted_globals`.

Names are resolved by matching `model_state` keys to the first n captured
tensors in pickle opcode order — model_state is fully processed before
optimizer_state in the stream, so the mapping is exact; optimizer state and
step counters are discarded.

This makes BDH/PyTorch support dependency-free for the *scan* path (the venv
has no torch; the `activity` extra remains the only torch consumer).

---

## 5. Name mapping

Slot taxonomy additions (v2.4 spec): `bdh_encoder`, `bdh_encoder_v`,
`bdh_decoder`.

Mapping rules (spec `name_map` block, canonical; in-code fallback synced):

- **Monolithic + per-unit names** → non-layer rules, anchored:
  `^encoder($|\.)` → `bdh_encoder`, `^encoder_v($|\.)` → `bdh_encoder_v`,
  `^decoder($|\.)` → `bdh_decoder`, `^attn\.freqs$` → `rope_freqs`,
  `^embed\.weight$` → `embed`, `^lm_head$` → `lm_head`. All map to
  `layer=None`.
- **Per-head names** (`blk.0.encoder`) ride the existing GGUF layer pattern
  (`blk\.(\d+)` → layer = head index) via three new gguf/base rules
  (`blk\.\d+\.encoder_v$`, `blk\.\d+\.encoder$`, `blk\.\d+\.decoder$`). No real
  GGUF tensor ends in `.encoder`/`.decoder`, so this is collision-free; `$`
  anchors prevent e.g. `blk.2.encoder_layer` from matching.

Anchoring matters beyond tidiness: the first cut used bare `encoder` /
`decoder` patterns with `re.search`, which would have re-mapped any
architecture with "encoder" in a tensor name (T5-style stacks land on
`other`/`layer.0` — verified unchanged by test).

`map_name` returns `(layer=None, slot)` for BDH core names; the raster skips
`layer=None`, which is what routes BDH away from the main raster and into the
dedicated paths below.

---

## 6. Visualization

### 6.1 Route-lattice panels (the main event)

`rasterize_bdh_lattice()` consumes the per-unit stats and emits one ExpertPanel
per BDH slot: **rows = heads (8), columns = lattice units (mult)**, written as
`field_expert_bdh_{slot}_{channel}_{raw,smooth}.tif` — the expert-panel naming,
so the matplotlib sheet renderer (title parsing, expert channel scales), the
compare panel machinery, and paired sheets pick them up **unchanged**. This was
the decisive reuse decision: zero renderer code for a new visualization class.

Panels are generated from the **main spec channels** (height=spectral_norm,
tint=stable_rank, rough=kurtosis). At `(D, 64)` / `(64, D)` block size the
SVD-based stats are cheap (64 singular values per unit); the expert_channels
shortcut used for MoE is unnecessary.

Read the panels as: column = 64-neuron block, column groups = training phases.
Phase growth (frozen prefix vs trained suffix) should appear as column
structure; concentration across heads shows up as row structure.

### 6.2 Flat field

`rasterize_flat()` builds a 1 × n_slots row for the non-layered model as a
whole (monolithic tensors only — handles with `expert_id` are skipped so the
2112 unit handles cannot collapse into one column). It fires only when the
model has no per-layer tensors, so transformer models are unaffected. Sheets
label the single row `model`.

### 6.3 Main raster as heads × slots

Per-head handles give BDH a conventional 8-row raster (`blk.0.encoder` →
layer 0, slot `bdh_encoder`, ...). This is the coarse view; the lattice panels
are the fine one.

---

## 7. What the panels show on a real CL run

Scan of `bdh_europarl_poc-ra2-a09-p3_last.pt` (route-aware α=0.9, phases EN→DE
→ES, mult 24→56→88; units 0–23 = p1-frozen, 24–55 = p2-frozen, 56–87 =
p3-trained). Table values are **encoder-slot** panels, arithmetic means over
per-unit cell values (heads × units) in each window:

| Unit window | stable_rank (mean) | spectral_norm (mean) |
|---|---|---|
| p1 units 0–23 (frozen) | 2,36 | 1,04 |
| p2 units 24–55 (frozen) | 2,13 | 1,93 |
| p3 units 56–87 (trained) | **1,18** | **7,43** |

**Panel reproducibility.** Cell values are bit-exact reproducible: per unit,
`log1p((‖A‖_F/σ₁)²)` over the exact float32 SVD of the `(D, 64)` / `(64, D)`
block (verified cell-by-cell against independent SVD and the fingerprint).
All BDH matrix blocks have `min(m, n) ≤ 512` (`SMALL` in
`stats/spectrum.py`), so the exact-SVD path always applies — the randomized
truncated spectrum (k=16, q=2, seeded by `spec.seeds.svd`) is never used for
BDH and panel numbers carry no seed sensitivity. Note the aggregation
semantics when recomputing: a window **mean over per-unit stats** is a
different quantity than the stable_rank of the aggregated window block
(e.g. p1 window: 2,36 per-unit mean vs 2,17 block-level) — do not mix them.

Findings weight-side, complementary to behavioural P-Route/P-Det:

1. **Routing is visible in weights.** Trained units collapse toward rank-1
   (stable_rank → 1 = one dominant singular direction), frozen units keep
   their original structure. stable_rank is scale-invariant, so this signal is
   immune to the amplitude shrink below.
2. **Growth is uniform across heads** (p3 window per-head means 1,08–1,43):
   no head-concentrated specialization at this scale.
3. **Weight-decay leak on "frozen" neurons.** The grad-masked prefix shrinks
   by a uniform factor per phase — structural change preserved (cos ≥
   0,9999), amplitude decays. Cause: `p.grad.mul_(mask)` leaves a zero tensor
   where the trainer needed `p.grad = None`, so AdamW's decoupled weight
   decay (wd=0.1) applies to all core weights. ("Frozen" embed/lm_head use
   real `requires_grad=False` and are bit-identical across phases.)
   Independent review (A0/Quinn) quantified and closed the mechanism: the
   ratio is uniform per element (poc p3/p2 on p1-units: 0,5865, min=max;
   ladder cross-check 0,582–0,586) and follows the schedule exactly —
   exp(−0,1·Σlr) = 0,5798 over the full 10k-iteration schedule (Σlr = 5,451),
   divided by the init-from-`_best` tail (≈0,9885) gives 0,5866, matching the
   measured 0,5865. The leak compounds: after 11 phases the en-phase units
   retain ≈ 2,7·10⁻³ of their original amplitude. Fix decision belongs to
   the BDH side (options: wd=0 + manual decoupled decay on trainable rows
   only, post-step compensation, or per-row parameter groups); until fixed,
   the per-phase factor is deterministic and can be divided out analytically
   before cross-phase amplitude comparisons.

---

## 8. Comparability

- **Across BDH phase checkpoints** (the intended use): same architecture, same
  slots; `compare` strict mode rejects only on spec mismatch — panel compare
  handles differing unit counts via `aligned_interp`, exactly like MoE expert
  panels with differing expert counts. The p1→p2→p3 delta should show the
  decay factor in the prefix and the structural change in the suffix.
- **Against transformers**: only through the shared universal slots
  (`embed`, `lm_head`, `rope_freqs`) and the flat/head-raster views. The BDH
  slots themselves have no transformer equivalent — by design; forcing them
  into `mlp_*` slots would imply a correspondence that does not exist.
- **Spec versioning**: `spec_version` stays 4 (additive extension, hard-reject
  semantics untouched). Older specs (v1–v2.3) predate the name_map block and
  use the in-code fallback, which now includes the BDH rules — mapping stays
  consistent for old-spec contexts.

---

## 9. Limitations and non-goals

1. **Only the route-aware grow path is modelled.** Checkpoints trained with
   plain fine-tune (`init_from` without `grow_mult`) or write-gating
   (`gate_from`) parse fine (they are BDH-layout checkpoints) but their
   phase structure is *not* frozen-prefix/suffix; lattice panels still show
   per-unit structure, just without the phase interpretation.
2. **Cross-phase precision floor: the decay dominates, bf16 is the residual.**
   Cross-phase comparisons of "frozen" prefixes differ by rel-L2 ≈ 0,42
   (p1→p2) — that difference is almost entirely the deterministic
   weight-decay shrink of §7.3 (uniform, structure-preserving, cos ≥ 0,9999).
   The bf16 autocast + fp32 storage roundtrip adds only ~2–4·10⁻³ of
   non-trend noise on top. Amplitude comparisons across phases must detrend
   by the decay factor (deterministic and analytically known, see §7.3) or
   use scale-invariant statistics (stable_rank, effective_rank), which are
   unaffected by both.
3. **Optimizer state discarded.** Deliberate: exp_avg/exp_avg_sq are moment
   estimates, not weights. A future "optimizer diagnosis" scan could expose
   them, but that is a different product.
4. **Single checkpoint per scan.** Sharded/distributed `.pt` bundles are out
   of scope; `_discover_pt_files` merges multiple `.pt` files in one directory
   under the same rules but no real BDH workflow uses that yet.
5. **bdh_prime variant.** `bdh_prime_wikitext2_primes_best.pt` has extra
   attention tensors (`attn.wf/bf/wi/bi`); they pass through monolithically
   and land on `other` — safe, not informative. Dedicated slots only when a
   prime-model workflow needs them.
6. **Pickle coverage.** The unpickler handles the observed PyTorch v3 ZIP
   layout (protocol 2, `_rebuild_tensor_v2`). Other rebuild variants
   (`_rebuild_tensor_v3`/`_rebuild_tensor` with metadata dicts) and
   `torch.load(..., weights_only=True)`'s newer zip serialization format are
   untested; they fail loudly on the whitelist rather than mis-parsing.
7. **No GPU required for scans** (slicing + numpy stats only), but the BDH
   *trainer* is CUDA-bound — scanning during training is safe, the two do not
   contend.
8. **Lattice panels skip the fractal/Blender renderers** (which build from the
   primary language raster only) — same policy as MoE expert panels today.

---

## 10. File map

| Concern | Location |
|---|---|
| Loader, unpickler, BDH expansion, slice geometry | `src/weight_atlas/loaders/pytorch_loader.py` |
| Format detection (ZIP magic → `pytorch`) | `src/weight_atlas/core/types.py` (`detect_loader`) |
| BDH name rules (canonical + fallback) | `specs/atlas_spec.v2.4.json` (`name_map`), `src/weight_atlas/core/name_map.py` |
| Slots `bdh_*` | `specs/atlas_spec.v2.4.json` (`slots`) |
| Lattice panels, flat field | `src/weight_atlas/fields/rasterizer.py` |
| Scan wiring (panels, flat skip, loader import) | `src/weight_atlas/scan.py` |
| CLI loader choices | `src/weight_atlas/cli.py` (`list_loaders()`) |
| Tests (synthetic pickle fixture, RCE guard) | `tests/test_pytorch_loader.py` |

---

## 11. Verification

```bash
cd /media/data/coding/weight-atlas && .venv/bin/python -m pytest tests/  # 707 passed, 1 skipped
```

- `tests/test_pytorch_loader.py` (8 tests): fixture is a hand-assembled
  PyTorch protocol-2 pickle (no torch needed); covers expansion names/shapes,
  slice-value correctness on both layouts, plain-passthrough, name mapping
  (incl. T5-style non-collision), lattice rasterization, flat-field skip, and
  the unpickler whitelist.
- Slice geometry additionally verified bit-exact against the real p3
  checkpoint (`np.array_equal` on corner units, both tensor layouts).
- Full-suite determinism and mapping-coverage tests unchanged and green.

Manual end-to-end (CPU-only, ~3 min for the ~70M checkpoint):

```bash
.venv/bin/python -m weight_atlas.cli scan \
    /media/data/coding/bdh/out/bdh_europarl_poc-ra2-a09-p3_last.pt \
    --out /tmp/bdh_p3 --loader pytorch
# → fingerprint.json (loader=pytorch, loader_metadata.bdh_*),
#   field_flat_*, field_*.tif (8-row head raster),
#   field_expert_bdh_{encoder,encoder_v,decoder}_{height,tint,rough}_{raw,smooth}.tif
```
