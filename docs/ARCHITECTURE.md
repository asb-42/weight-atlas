# Architecture

## Pipeline

```
safetensors ─► TensorHandle (lazy) ─► Statistic.compute ─► TensorStats
                                                                │
                                                                ▼
                                                            rasterize
                                                                │
                                                                ▼
                                                       Field2D (per channel)
                                                       ┌──────┴──────┐
                                                     scale        smooth+upsample
                                                       │              │
                                                       ▼              ▼
                                                  field_<ch>_raw.tif  field_<ch>_smooth.tif
                                                       │
                                                       ▼
                                               fingerprint.json + manifest.json
                                                       │
                           ┌────────────────────────────┼────────────────────────────┐
                           │                            │                            │
                           ▼                            ▼                            ▼
                    Matplotlib Sheet              Blender Renderer            (future: web)
                    (hillshade+hypsometric        (3D ortho top-view
                     + contours)                   + OBJ mesh)
```

Renderers (matplotlib, Blender) read **only** artefacts (TIFF + JSON), never weights.



## v0.2.0 — Real-Model-Calibration (Bonsai-8B)

### Spec Changes
- `spec_version`: 2 (was 1)
- `tint` channel: `effective_rank` → `stable_rank` (log1p((frobenius/spectral_norm)²))
- `rough` channel: `log1p` → `quantile_clip` (1–99%)
- `height` channel: unchanged (spectral_norm + log1p)

### New Slots
- `attn_q_norm`, `attn_k_norm`: QK-Norm scale tensors (Bonsai-8B)

### Name Audit System
- Bonsai-8B fixture: `tests/fixtures/names_bonsai_8b.json`
- `diagnose` CLI command: `weight-atlas diagnose <path>`
- `mapping_coverage` block in fingerprint.json
- CLI warning when `in_slots < 80%`

### Degeneration Guards
- Per-channel diagnostics: `valid_fraction`, `normalized_std`
- Std < eps OR valid < 50% → CLI warning + warnings block + UI banner
- Module: `weight_atlas/fields/degenerations.py`

### UI Changes
- Artifact route: `GET /models/{id}/artifacts/{name}` (allowlist + traversal protection)
- `.tif` entries shown as "not inline displayable — download" link
- Detail page uses only the artifact route
- Scan job auto-renders sheets after completion
- Degeneration warnings banner on detail page

### Migration
**v0.2.0 requires re-scan.** Existing v1 fingerprints cannot be compared with v2. The compare command hard-rejects spec_version mismatches.

To migrate:
```bash
# Re-scan all models with v2 spec
weight-atlas scan ./models/my_model --out ./artefacts_v2
# Verify mapping coverage
weight-atlas diagnose ./models/my_model
```
## Conventions

- **Raster**: rows = layer index, columns = slot order from `atlas_spec.v2.2.json`. Missing cells = `NaN`, never filled.
- **Channels** (v2.1):
  - `height`: `spectral_norm` → `log1p` → `rank_scale(per_column)` — outlier suppression + [0,1] mapping
  - `tint`: `stable_rank` → `log1p` → `robust_scale(1-99%)` — outlier suppression + [0,1] mapping
  - `rough`: `kurtosis` → `rank_scale(per_column)` — unified with other channels
- **RNG**: all random state seeded from `spec.seeds.svd` (currently only randomized SVD).
- **Artefacts**: no timestamps; PNG metadata fixed (`Software: weight-atlas`, `Creation Time: 1970-01-01T00:00:00Z`); TIFF byte-identical on second run (verified by SHA-256 manifest).
- **fingerprint.json**: top-level block includes `spec_version`, `tool_version`, `loader` for cross-spec comparability.

## Sheet Clarification

The matplotlib sheet is a **pure height map**: hillshade + hypsometric tint + contours all derived from the **height channel only**. Tint and rough channels are separate fields intended for Blender (M2) and future sheets. This is a deliberate design decision to keep the 2D sheet focused on topographic readability.

## Contour Convention

Contours on the 2D sheet use deterministic, comparable levels: `np.linspace(0.02, 0.98, spec.sheet.contour_levels)` applied to the **scaled height field** (already in [0,1]). Because `robust_scale` guarantees a well-distributed [0,1] range, fixed percentile levels are globally comparable without per-model recomputation. Line color is fixed black with alpha 0.4.

## Blender Pipeline

### Data flow
```
field_height_smooth.tif ─┐
                          ├─► .npy tempdir ─► render_terrain.py (bpy) ─► terrain_smooth.png
field_tint_smooth.tif ──┘                                          │
                                                                     └──► terrain.obj
```

### Design decisions

- **Engine**: `BLENDER_WORKBENCH` (headless-safe, deterministic, no GPU/EGL requirement). Cycles deferred as beauty-option to backlog.
- **Binary resolution**: `WEIGHT_ATLAS_BLENDER` env var → `shutil.which("blender")` → clear error message with install hint. No pip dependency.
- **Mesh**: 1024² grid (spec value), vertex-Z from height (normalised + z-scaled), vertex-color from tint. Mesh generation via `foreach_set` for performance.
- **Camera**: orthographic, fixed 18° pitch (not top-down; reveals relief while
  staying orthographic — no perspective distortion, comparable across models).
  ortho_scale computed from pitch + effective z-scale so the tilted grid stays
  in frame (never smaller than 2.2).
- **Lighting**: Sun lamp, azimuth 315° (NW), altitude 45°, energy 1.0. Studio shading with vertex colors.
- **Height normalisation**: robust percentile clip (1–99%, spec `clip`) before
  rescaling to [0,1] — a single outlier hotspot can no longer flatten the bulk.
  `adaptive_z_scale` (opt-in) rescales Z so relief std is constant across
  fields (base_z_scale / std, capped at 5.0); **breaks absolute-amplitude
  comparability** — document as purely visual.
- **OBJ export**: plain-text Wavefront OBJ at 256² downsample, written directly by wrapper (no bpy-ops). Deterministic, diffable fingerprint artefact, uses the same robust normalisation as the PNG render.
- **World**: fixed dark grey (0.05, 0.05, 0.05), no HDR/noise.
- **Determinism**: same height+tint inputs → byte-identical PNG (locally verified by smoke test) + byte-identical OBJ (unit tested). Workbench must provide pixel-identical output; if not, documented in smoke-test log with root-cause analysis (never SSIM).

### Spec extension

The `atlas_spec.v2.2.json` may include a `blender` block (all optional, defaults shown):
```json
{
  "blender": {
    "grid": 1024,
    "resolution": 2048,
    "z_scale": 0.3,
    "pitch": 18.0,
    "clip": 0.01,
    "adaptive_z_scale": false
  }
}
```
`pitch`: camera tilt in degrees (0 = top-down, 18 = default relief view).
`clip`: percentile band for robust height normalisation (0 = plain min/max).
`adaptive_z_scale`: if true, effective z = `z_scale / std(height)` (capped at
5.0) — amplifies weak relief but makes amplitudes relative, not absolute.

`spec_version` remains 1 (pre-release); extension documented here per spec.

### Smoke test

`scripts/smoke_blender.sh` (local, not CI):
1. Scan fixture model
2. `weight-atlas render OUT_DIR --renderer blender`
3. Second render → SHA-256 comparison of PNGs

CI tests only the wrapper (dry-run, subprocess mocked, binary not required).

## Design decisions

- **TIFF over EXR**: Blender reads Float32 TIFF natively; no C builds required. EXR deferred to M2+.
- **Shared truncated spectrum (one SVD per tensor)**: spectral norm, effective rank and stable rank all derive from the same singular values (`stats/spectrum.py`). Computing them independently ran up to three SVDs per tensor — the dominant cost when scanning MoE models (tens of thousands of expert tensors). The spectrum is computed once per tensor (exact SVD for `min(m,n) <= 512`, else the Halko randomized SVD with `k=16, q=2` in float32, seeded from `spec.seeds.svd`). float32 matmuls differ from float64 by <1e-8 relative on the singular values.
- **Chunked O(n) statistics**: kurtosis, frobenius, kernel norm and sparsity accumulate in float64 over 1M-element chunks of the float32 payload instead of materializing full-size float64 arrays (the old kurtosis allocated `diff`/`diff**2`/`diff**4` copies and dominated scan time). Values agree with the vectorized computation to float64 rounding.
- **Bounded memory on large scans**: every tensor's memoized float32 payload is released (`TensorHandle.clear()`) right after its statistics are computed, so the whole model is never held in RAM (~4 bytes/parameter; a 35B MoE would otherwise be ~140 GB).
- **Parallel statistics workers**: per-tensor statistics are independent and deterministic, so `scan(..., jobs=N)` (default `min(8, cpu_count)`) computes them in a thread pool with per-worker BLAS capped to one thread (`threadpoolctl`). Output is byte-identical for any `jobs`.
- **Effective rank from truncated spectrum**: using `k=16` singular values introduces a small downward bias on very high-rank matrices, but keeps runtime bounded and noise low. Documented here per spec.
- **argparse over click**: core stays dependency-minimal; flat command hierarchy doesn't need click's nesting features.
- **1-D tensors**: spectral norm = L2, effective rank = 1. Bias vectors still computed (spec requirement).
- **conftest imports registering modules**: decorator-based registration only runs on import; `registry.reset()` in tests would wipe entries. Centralised in conftest. **Known limitation**: tests that call `registry.reset()` must re-import registering modules. Future: entry-point registration for plugin ecosystem (backlog).
- **quantile_clip normalizes to [0,1]**: after clipping to [1%, 99%] quantiles, values are min-max normalized to a fixed [0,1] range. This makes the tint channel comparable across models (the goal).
- **PNG Creation Time fixed**: spec requires `1970-01-01T00:00:00Z` to prevent encoder metadata from breaking determinism. Currently set via matplotlib's `metadata` kwarg.
- **Render discovery from manifest**: the render command uses `manifest.json` as source of truth for which channels exist, not filename globbing. Filenames remain convention, but manifest is authoritative.

## Web UI (M3)

### Data flow
```
Browser ─► FastAPI routes ─► JobQueue (SQLite) ─► In-process worker ─► scan.py
                                                                          │
                                                                          ▼
                                                               fingerprint.json + TIFFs
                                                               ┌────────────────────┐
                                                               │ Matplotlib Sheet   │
                                                               │ Blender Renderer   │
                                                               └────────────────────┘
```

### Design decisions

- **HTMX via CDN**: no npm build, no node dependencies. HTMX 1.9 from `unpkg.com`. Pure HTML-over-the-wire; no JavaScript framework.
- **SQLite for job persistence**: job state survives server restarts. Single DB file at `./data/jobs.db`.
- **In-process worker thread**: one background thread picks up QUEUED jobs and runs them sequentially. Simple, no task queue infrastructure needed.
- **Jinja2 templates with HTMX partials**: `_job_status.html` is a partial rendered via HTMX polling (`hx-trigger="every 2s"`).
- **Read-only artefact views**: the UI reads artefacts from disk, never accesses raw weights directly.
- **Form-encoded job submission**: `hx-post="/api/jobs"` submits `model_path`; server creates job + out_dir.

### Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Model list page |
| `/api/jobs` | POST | Submit new scan job |
| `/api/jobs/{id}` | GET | Job status JSON |
| `/jobs/{id}` | GET | Job progress page (HTMX polling) |
| `/models/{id}` | GET | Model detail (sheet, terrain, stats, spec) |
| `/api/models/{id}/fingerprint` | GET | Fingerprint JSON |
| `/api/jobs/{id}/status` | GET | HTMX partial: status badge + progress bar |
| `/compare` | GET | Compare page (select two models) |
| `/api/compare` | POST | Submit new compare job |
| `/compare/{id}` | GET | Compare report (delta visualizations + metrics) |

### Directory structure
```
src/weight_atlas/api/
├── __init__.py
├── main.py          # FastAPI app factory
├── jobs.py          # JobQueue (SQLite + worker)
└── routes.py        # HTTP routes

src/weight_atlas/ui/
├── templates/       # Jinja2 templates
│   ├── base.html
│   ├── models.html
│   ├── detail.html
│   ├── job_progress.html
│   ├── compare.html
│   ├── compare_report.html
│   └── _job_status.html
└── static/
    └── style.css
```

### Running the web UI
```bash
uv sync --extra web
uvicorn weight_atlas.api.main:app --reload
# Open http://localhost:8000
```

## Comparison/Delta Layer (M4)

### Data flow
```
scan artefacts (A) ─┐
                    ├─► align.py ─► delta.py ─► compare_summary.json
scan artefacts (B) ─┘                    │
                                         ▼
                                  delta_sheet.py (renderer)
                                         │
                                         ▼
                                  delta_sheet_<channel>.png
                                  delta_profile_<channel>.png
```

### Design decisions

- **Two modes**: `strict` (same architecture, identical indices) and `aligned` (cross-architecture, normalized depth to t∈[0,1], resampled to common grid).
- **Δ-fields computed on scaled channel values** (B - A after applying channel scale).
- **Summary metrics computed on raw stats** (rel. L2, cosine similarity before scaling).
- **Hard-reject on spec_version mismatch**: raises ValueError, comparison not allowed.
- **Warning on tool_version mismatch**: logs warning, comparison still allowed.
- **Aligned mode**: normalizes depth to t∈[0,1], resamples to common grid (≥64 depth samples).
- **Aligned row resampling** (`compare.aligned_interp`): `linear` (default, bilinear
  interpolation via scipy zoom) or `nearest` (nearest-layer matching — each row
  copies the nearest real layer by depth; preserves layer structure and NaN holes).
  Rows are normalized depth, NOT absolute layer indices; layer maps per row are
  recorded in `compare_summary.json` → `alignment`.
- **Hotspot ranking**: top-k locations ranked by |delta|, with NaN positions filtered out.
- **Read-only**: compare reads only artefacts (TIFF + JSON), never weights.
- **Diverging colormap**: RdBu_r centered at zero for delta visualization.
- **Symmetric limits**: ±q(diverging_clip) per channel (default q=0.98), capped at a
  robust spread (median + 4.4826·MAD ≈3σ) so a few outliers cannot flatten the bulk.
- **Empty columns**: all-NaN columns (slots missing in one model) are dropped before
  rendering; the original→kept mapping is exposed as `kept_cols` on the renderer.
- **Profile strip**: 1×L "ablitation bar" showing per-layer relative L2 (hot colormap).

### compare_summary.json Schema
```json
{
  "mode": "strict",
  "spec_version": 1,
  "model_a": { "tool_version": "...", "loader": "...", "n_tensors": 0, "n_layers": 0 },
  "model_b": { "tool_version": "...", "loader": "...", "n_tensors": 0, "n_layers": 0 },
  "warnings": [],
  "channels": {
    "height": {
      "rel_l2": 0.123,
      "cosine_sim": 0.987,
      "hotspot_layer": 2,
      "hotspot_slot": "mlp_down",
      "hotspot_value": 0.456,
      "argmax": [2, "mlp_down"],
      "hotspot_ranking": [
        { "layer": 2, "slot": "mlp_down", "abs_delta": 0.456 },
        { "layer": 3, "slot": "attn_o", "abs_delta": 0.789 }
      ]
    }
  }
}
```

### Spec extension
The `atlas_spec.v2.2.json` may include a `compare` block:
```json
{
  "compare": {
    "modes": ["strict", "aligned"],
    "default_mode": "strict",
    "aligned_grid": 64,
    "colormap": "RdBu_r",
    "diverging_clip": 0.98,
    "aligned_interp": "linear"
  }
}
```
`spec_version` remains 1 (extension documented here per spec).
`aligned_interp` accepts `"linear"` (bilinear interpolation, default) or
`"nearest"` (nearest-layer matching by normalized depth; preserves layer
structure and NaN holes).

### CLI
```bash
weight-atlas compare DIR_A DIR_B --out DIR --mode {strict,aligned} [--interp {linear,nearest}]
```

### Renderer placement
The delta renderer lives at `compare/render/delta_sheet.py` (registry ID `"delta"`, not `render/`) because it is owned by the compare module and shares its namespace. This is a cohesion-driven decision; the backlog item "Entry-point plugin registration" will unify renderer discovery across modules.

### Localization test
Fixture A + mutated B (layers.2.mlp.down_proj set to 100.0, rank-1 perturbation on layers.3.self_attn.o_proj). `compare --mode strict` → hotspot ranking reports (2, mlp_down) and (3, attn_o) as Top-2 for height channel. argmax == (2, mlp_down).

## GGUF Loader (M5)

### Data flow
```
*.gguf file ─► GGUFReader (mmap) ─► TensorHandle (lazy) ─► dequantize ─► float32 ─► pipeline
```

### Design decisions

- **Dependency**: official `gguf` package, only in `gguf` extra (not in core)
- **Registry-ID**: `gguf`
- **Auto-Detect**: via magic bytes (GGUF vs safetensors header)
- **Dequant**: canonical float32 (F32, F16, BF16, Q8_0, Q4_0)
- **BF16**: bit-shift (uint16 → uint32 view), **Q8_0/Q4_0**: block-wise (32-element blocks, f16 scale)
- **Name mapping**: second, loader-independent rule set for GGUF names (blk.N.attn_q → attn_q, etc.)
- **Same slot IDs as safetensors**: raster remains loader-intercomparable
- **Fingerprint**: per-tensor ggml_type, top-level quantization summary
- **Cross-loader compare**: loader/quantization mismatch → warning (CLI-Log + UI banner), no reject
- **UI**: Detail page shows ggml_type distribution table

### Supported quantization types (M5 scope)
- F32 (type 0), F16 (type 1), BF16 (type 30), Q8_0 (type 8), Q4_0 (type 2)
- Other types → clear error message with type name + backlog entry for "full k-quant support"

### GGUF name mapping
| GGUF name | Slot |
|-----------|------|
| blk.N.attn_q/k/v | attn_q/k/v |
| blk.N.attn_output | attn_o |
| blk.N.ffn_gate/up/down | mlp_gate/up/down |
| blk.N.attn_norm | norm_attn |
| blk.N.ffn_norm | norm_mlp |
| token_embd | embed |
| output | lm_head |
| output_norm | norm_mlp |

### Cross-loader warning rule
- Loader mismatch (safetensors vs gguf) → warning + run allowed
- Quantization mismatch (F32 vs Q8_0) → warning + run allowed
- Rationale: quantization noise is real part of signature, but for abliteration both sides should be identically quantized

## MoE Expert Panel (M6)

### Data flow
```
Expert Tensors ─► rasterize_expert_panels ─► ExpertPanel (Layer × Expert)
                                           ─► field_expert_<slot>_<channel>_{raw,smooth}.tif
                                           ─► sheet renderer (generic over Field2D)
                                           ─► delta renderer for panel comparison
```

### Design decisions

- **Main raster unchanged**: Expert tensors are excluded from the main raster; they flow into separate ExpertPanel fields
- **Shared expert decision**: Shared expert tensors occupy the mlp slots (mlp_gate/mlp_up/mlp_down) in the main raster
- **Expert panel = own field class**: `ExpertPanel` with shape (Layer × Expert IDs), same channel definitions as main raster
- **Expert panels use cheap channels (`expert_channels`, spec v2.4)**: expert tensors are the vast majority of a MoE model's tensors, so the panels consume the spec's `expert_channels` block — O(n) statistics only (height=frobenius, tint=kurtosis, rough=sparsity) — instead of the SVD-based main channels. The shared spectrum (spectral_norm/stable_rank) is reserved for the few dense tensors, keeping MoE scans practical (measured: ~8.4 min for the 24.6k expert tensors of a 35B-A3B model with 8 parallel workers). Specs without `expert_channels` fall back to the main channels.
- **Fingerprint**: model block includes `moe: {num_experts, shared_expert}` — derived from tensor presence (safetensors) or metadata (GGUF)
- **Blender**: `--field expert_mlp_down` renders terrain from expert panel

### GGUF 3D expert split
GGUF stores MoE expert tensors as 3D arrays (hidden, hidden, n_experts). The loader splits these into lazy sub-handles:
- `ffn_gate_exps.weight` → `ffn_gate_exps.weight[0]`, `ffn_gate_exps.weight[1]`, ...
- Each sub-handle has `expert_id` set and `load()` returns a 2D slice
- Shared experts (`ffn_*_shexp`) map directly to mlp slots

### Compare panel convention
- Panels compared when both sides have same shape (strict mode)
- Shape mismatch → `expert_panels: {status: "skipped", reason: "..."}`
- No crash on mismatch — comparison continues with available panels
- Panel delta sheets via existing delta renderer (generic over Field2D)

### MoE name mapping (HF)
| Pattern | Slot |
|---------|------|
| mlp.gate.weight | router |
| mlp.experts.{e}.(gate\|up\|down)_proj | expert |
| shared_expert.(gate\|up\|down)_proj | mlp_gate/up/down |
| shared_expert_gate | other |

### MoE name mapping (GGUF)
| Pattern | Slot |
|---------|------|
| blk.N.ffn_gate_inp | router |
| blk.N.ffn_(gate\|up\|down)_exps | expert (3D stacked) |
| blk.N.ffn_(gate\|up\|down)_shexp | mlp_gate/up/down |

### Rule order importance
`mlp.gate.weight` (router) must match before `mlp.gate_proj` (mlp_gate). The name_map rules are ordered to ensure this.

### Spec-driven name mapping (`name_map` block, spec v2.4)
The tensor-name → slot tables are **spec-driven**: the canonical default spec
carries a top-level `name_map` block (per-convention ordered rules for
hf/gguf in moe/base/hybrid/kimi groups, `layer` index patterns,
`non_layer_order`, and the vision rules). `core/name_map.py` compiles it at
runtime and falls back to its in-code tables for older specs without the key.

Adding a new tensor family is now a spec edit: add the rules to the `name_map`
block **and** a new `slots` entry (in v2.4; `spec_version` stays 4 — additive).
`ssm_ba` (Mamba B-matrix, Qwen3-Next) was the first such family: previously it
fell through to `other` and left 36 tensors unmapped per model; it now maps to
its own `ssm_ba` slot.

### Artefacts
- `field_expert_mlp_{gate,up,down}_<channel>_{raw,smooth}.tif`
- Sheet PNGs via existing sheet renderer (generic over Field2D)
- Delta panel sheets via existing delta renderer

## VLM Vision Tower (multimodal models)

### Design decision

Vision towers are **not excluded** from the fingerprint to inflate mapping
coverage — they are mapped into their own slot taxonomy and rasterized into a
separate sheet, so a vision tower present in a multimodal model (and absent in
a text-only model) becomes a visible fingerprint difference.

- `map_vision()` (in `core/name_map.py`) maps vision tensors to
  `(vision_block_index, vision_slot)` for the major naming families: GGUF
  llama.cpp (`v.blk.N.*`, `mm.model.mlp.N`), HF Qwen3-VL / CLIP
  (`vision_model.encoder.layers.N.*`), HF Qwen2-VL (`visual.blocks.N.*`) and
  Kimi K3 (`vision_tower.encoder.blocks.N.*`).
- `map_name()` delegates to `map_vision` and returns `(None, slot)`, so vision
  tensors never collide with transformer layers in the main raster.
- Spec blocks: `vision_slots` (own column taxonomy: `v_attn_*`, `v_mlp_*`,
  `v_patch_embed`, `v_pos_emb`, `mm_projector`, ...) and `vision_channels`
  (own statistics; `height` uses `kernel_norm` — mean per-output-channel L2
  norm of a conv kernel — instead of spectral norm, since vision towers are
  Conv/ViT-structured, not attention-matrix structured).
- `rasterize_vision()` builds a `vision_block × vision_slot` field; global
  tensors (patch_embed, pos_embed, projector) land in a final `"global"` row
  and are mean-aggregated per slot cell.
- Artefacts: `field_vision_<channel>_{raw,smooth}.tif`, rendered as
  `vision_<channel>_raw.png` sheets next to the transformer sheets.
- Fingerprint: `model.vision = {present, n_tensors, n_blocks, n_global}` and
  `mapping_coverage.vision_tensors`. Text-only models have no `model.vision`
  block and no vision artefacts.

## Extras (lazy)

`umap` is declared but empty – imports must stay out of core.
