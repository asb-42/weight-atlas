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
                    Matplotlib Sheet              Blender Renderer             Web UI (M3)
                    (hillshade+hypsometric        (3D ortho top-view   (FastAPI + HTMX,
                     + contours)                   + OBJ mesh)           serves artefacts)
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

- **Raster**: rows = layer index, columns = slot order from `atlas_spec.v2.4.json`. Missing cells = `NaN`, never filled.
- **Channels** (v2.1):
  - `height`: `spectral_norm` → `log1p` → `rank_scale(per_column)` — outlier suppression + [0,1] mapping
  - `tint`: `stable_rank` → `log1p` → `robust_scale(1-99%)` — outlier suppression + [0,1] mapping
  - `rough`: `kurtosis` → `rank_scale(per_column)` — unified with other channels
- **RNG**: all random state seeded from `spec.seeds.svd` (currently only randomized SVD).
- **Artefacts**: no timestamps; PNG metadata fixed (`Software: weight-atlas`, `Creation Time: 1970-01-01T00:00:00Z`); TIFF byte-identical on second run (verified by SHA-256 manifest).
- **fingerprint.json**: top-level block includes `spec_version`, `tool_version`, `loader` for cross-spec comparability.

## Sheet Clarification

The matplotlib sheet is a **pure height map**: hillshade + hypsometric tint + contours all derived from the **height channel only**. Tint and rough channels are separate fields intended for Blender (M2) and future sheets. This is a deliberate design decision to keep the 2D sheet focused on topographic readability.

## Normalized-Depth Projection (Ebene 2)

Level-1 fields (`Field2D`) are indexed by absolute layer index, so models with different layer counts have incomparable row axes, and absent slots leave NaN "perforation" holes. `fields/normalize.py` (`project_normalized_depth`) re-maps each column onto a fixed set of depth landmarks (0 % … 100 % of the layer stack) by linear interpolation over the measured rows, returning an **interpolation mask** marking every estimated cell.

- Enabled per spec: `sheet.normalized_depth` (default `false`), `sheet.normalized_depth_landmarks` (default `21`).
- The web UI can apply it **per render**: the overview's "Normalized depth" checkbox sends a `normalized_depth` sheet_knob to `/api/jobs/{id}/render/sheet`, which the worker overlays onto the recorded spec for that render only (`_apply_sheet_knobs`) — the scan's spec stays untouched, and re-renders without the checkbox go back to the default.
- Landmarks outside a column's measured range stay NaN (extreme depth regions are honestly left as holes, never extrapolated).
- A landmark cell is marked *measured* only when a measured row lies within half a landmark spacing; everything else is interpolated.
- The sheet renderer shades interpolated cells with a subtle grey veil (`alpha=0.25`) and appends `· normalized-depth` to the title, so estimates stay visible.
- Same Ebenen hierarchy: **Ebene 1** = raw layer-index field; **Ebene 2** = normalized-depth projection; **Ebene 3** = display-only empty-column compression; smoothing/upsample happen afterwards.

## Empty-column compression (Ebene 3)

Slot families absent from a model leave **all-NaN white columns** on the
sheets (this is not a mapping error — the model genuinely has no such
tensors). `sheet.drop_empty_cols` (default `false`, spec v2.4 knob) drops
those columns display-only and lets imshow's `extent` re-space the survivors
evenly, so the sheet reads cleanly for models with few slot families.

- **Display-only**: the TIFF fields and fingerprints keep full width — only
  the rendered PNG compresses. Smooth (upsampled) fields drop whole slot
  blocks (`n_cols // n_labels`), never partial columns.
- **Only fully-empty columns drop**: a slot with any measured cell survives;
  partial NaN stays as-is (never dropped, never stretched).
- **Honest output**: the title appends `columns dropped: <slot, …>` so the
  sheet stays truthful about which families were compressed away.
- **Cross-model trade-off**: enabled, two models' sheets are no longer
  column-aligned (same slot at different x-positions). Off by default to
  preserve cross-model comparability; opt-in per scan via the spec or per
  render via the UI's "Drop empty columns" checkbox (same sheet_knob flow as
  `normalized_depth`).

## Raster pixel budget

The sheet renderer bounds the raster before rasterizing: it scales to a fixed pixel budget (`_MAX_RENDER_PIXELS = 12_000_000`, aspect-ratio preserving) instead of honouring raw tensor dimensions. Without this, a 736×7168 expert panel allocates ~95 GB and is OOM-killed; a long-edge-only cap crushes it to 4096×420 (uninspectable layers). This is a pure render-side cap — the TIFF fields themselves keep full resolution.

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
- **Mesh**: 1024² grid (spec value), vertex-Z from height (normalised + z-scaled), vertex-color from tint. Mesh generation via `foreach_set` for performance. Geometry smoothing (terrain, not raw values): height is bilinearly resampled to the grid (`resample_bilinear`, pure NumPy — no scipy inside bpy), the mesh is smooth-shaded (auto-smooth 30° cutoff) and Catmull-Clark subdivided (`blender.subsurf_levels`, default 1, 0=off).
- **Camera**: orthographic, fixed 18° pitch (not top-down; reveals relief while
  staying orthographic — no perspective distortion, comparable across models).
  ortho_scale computed from pitch + effective z-scale so the tilted grid stays
  in frame (never smaller than 2.2).
- **Lighting**: Workbench STUDIO shading with vertex colors and `use_scene_lights`/`use_scene_world` enabled (without these the scene SUNs are a no-op in the render). Two fixed SUNs: NW key (azimuth 315°, altitude 45°, energy 1.0) plus a soft SE fill (azimuth 135°, altitude 25°, energy `blender.fill_light_energy`, default 0.35) so the shadow side of the terrain reads instead of blacking out.
- **Height normalisation**: robust percentile clip (1–99%, spec `clip`) before
  rescaling to [0,1] — a single outlier hotspot can no longer flatten the bulk.
  `adaptive_z_scale` (opt-in) rescales Z so relief std is constant across
  fields (base_z_scale / std, capped at 5.0); **breaks absolute-amplitude
  comparability** — document as purely visual.
- **OBJ export**: plain-text Wavefront OBJ at 256² bilinear downsample (same `resample_bilinear` helper as the renderer, not nearest-neighbour), written directly by wrapper (no bpy-ops). Deterministic, diffable fingerprint artefact, uses the same robust normalisation as the PNG render.
- **World**: fixed dark grey (0.05, 0.05, 0.05), no HDR/noise.
- **Determinism**: same height+tint inputs → byte-identical PNG (locally verified by smoke test) + byte-identical OBJ (unit tested). Workbench must provide pixel-identical output; if not, documented in smoke-test log with root-cause analysis (never SSIM).

### Spec extension

The `atlas_spec.v2.4.json` may include a `blender` block (all optional, defaults shown):
```json
{
  "blender": {
    "grid": 1024,
    "resolution": 2048,
    "z_scale": 0.3,
    "pitch": 18.0,
    "clip": 0.01,
    "adaptive_z_scale": false,
    "subsurf_levels": 1,
    "fill_light_energy": 0.35
  }
}
```
`pitch`: camera tilt in degrees (0 = top-down, 18 = default relief view).
`clip`: percentile band for robust height normalisation (0 = plain min/max).
`adaptive_z_scale`: if true, effective z = `z_scale / std(height)` (capped at
5.0) — amplifies weak relief but makes amplitudes relative, not absolute.
`subsurf_levels`: Catmull-Clark subdivision levels (0 = raw flat-shaded mesh,
default 1).
`fill_light_energy`: energy of the soft SE fill sun lifting the shadow side
(default 0.35; 0 disables the fill).

`spec_version` stays 4 (additive extension documented here per spec; never
bump for new keys).

## Fractal Terrain Pipeline

### Data flow
```
fingerprint.json ─► slot_stat_medians ─► stats_to_params ─► slot_fractal_params
                                                            │
                                    slot_fractal_field (fBm)├─► .npy tempdir
                                    (per-slot strips)      │
                                                            ▼
                                     render_terrain.py (bpy) ─► terrain_fractal.png
                                                                 └──► terrain_fractal.obj

                                   ── SDF mode (fractal.mode = "sdf") ──
fingerprint.json ─► slot_stat_medians ─► stats_to_params ─► slot_sdf_params
                                                            │
                                    sdf_volume + surface_nets ├─► .npy tempdir
                                    build_sdf_mosaic         │
                                    (per-slot mini-SDFs)     ▼
                                     render_sdf.py (bpy) ─► terrain_fractal.png
                                                               └──► terrain_fractal.obj
```

### Design decisions

- **Genuine fractal geometry, not a texture**: per-slot fBm parameters
  (octaves, persistence, lacunarity, base frequency) are derived from the
  slot's *real* tensor statistics — effective_rank → octaves, kurtosis →
  persistence, sparsity → lacunarity, spectral_norm → base_freq (spec
  `fractal.mapping`). The height field *is* the fractal; statistics feed the
  formula, not a heightmap with a fractal painted on top.
- **Per-slot character**: each slot column in the raster is its own fBm strip
  (fixed per-slot seed = base seed + slot index), so adjacent slots with
  different stats render visibly different self-similar structure. Layout
  mirrors the (layers × slots) raster (`fractal.cell_h`/`cell_w` cells per
  logical cell).
- **SDF mode is a per-slot mosaic of mini-SDFs** (`fractal.mode = "sdf"`):
  instead of a height field, each slot cell gets its own 3D Menger-sponge or
  Mandelbulb object, extracted from the slot's SDF parameters
  (iterations/power/scale ← slot stats via `fractal.sdf.mapping`) with a
  deterministic naive Surface Nets iso-extraction. Objects are normalised per
  cell (each fills its footprint regardless of family/params) and merged into
  one mosaic mesh; tint encodes the slot column. Iteration counts are clamped
  to the lattice (`grid`) so coarse grids never alias into empty cells.
- **Rendered through the same Blender pipeline**: the fractal height/tint
  fields reuse `render_terrain.py` (same smoothing, lights, subsurf, PNG
  metadata stripping); the SDF mosaic reuses its world/light/camera/engine
  helpers via `render_sdf.py`. Fractal and plain terrain renders stay directly
  comparable. Tint is a second, independently-seeded per-slot fBm strip.
- **Determinism**: pure NumPy value noise on a fixed integer-lattice hash and
  pure NumPy SDF + surface-nets (no RNG, no timestamps) → byte-identical PNG
  + OBJ for identical inputs.
- **One render per model**: the fractal depends on the fingerprint + seed,
  not the channel. The API/CLI call `render()` once per channel (height, tint,
  rough, vision_*); a per-instance dedupe keyed on
  `(out_dir, seed, mode, layout)` makes Blender run once and all channels
  reuse the identical artefacts. `mode` + layout are part of the key so fBm
  and SDF renders, and channels with different rasters, never cross-pollinate
  the dedupe cache.
- **Primary raster only, cell budget**: the fractal is built from the primary
  language raster — `expert_*` (one column per expert) and `vision_*` channels
  are skipped, so an 896-expert panel can never define the layout. Rasters
  exceeding `fractal.sdf.max_cells` (default 1024) are deterministically
  decimated with aspect-preserving strides (objects keep their true positions
  and tints); without this an expert panel would be 80k+ objects (~115M verts)
  and crash Blender.
- **Per-render mode toggle**: the UI's "Fractal mode" `<select>` sends a
  `fractal_mode` form field to `/api/jobs/{id}/render/fractal`; the worker
  overlays it onto the recorded spec's `fractal.mode` for that render only
  (same `job.sheet_knobs` / `_apply_sheet_knobs` flow as the sheet checkboxes)
  — the scan's spec stays untouched, re-renders without the field fall back to
  the recorded mode.

### Spec extension

The `atlas_spec.v2.4.json` may include a `fractal` block (all optional,
defaults shown):
```json
{
  "fractal": {
    "seed": 0,
    "cell_h": 8,
    "cell_w": 8,
    "mode": "fbm",
    "mapping": {
      "octaves":     {"stat": "effective_rank", "lo": 4, "hi": 8},
      "persistence": {"stat": "kurtosis",       "lo": 0.4, "hi": 0.7},
      "lacunarity":  {"stat": "sparsity",       "lo": 1.8, "hi": 2.4},
      "base_freq":   {"stat": "spectral_norm",  "lo": 1.0, "hi": 2.5}
    },
    "sdf": {
      "family": "menger",
      "grid": 16,
      "max_cells": 1024,
      "mapping": {
        "iterations": {"stat": "effective_rank", "lo": 1, "hi": 4},
        "scale":      {"stat": "kurtosis",       "lo": 2.5, "hi": 3.5},
        "power":      {"stat": "kurtosis",       "lo": 2, "hi": 8}
      }
    }
  }
}
```
`seed`: base lattice seed (also `seeds.fractal`); `cell_h`/`cell_w`: fractal
cells per logical layer/slot; `mode`: `"fbm"` (height field, default) or
`"sdf"` (per-slot mini-SDF mosaic); `mapping`: per-target stat + linear
min→max range (clamped). Slots with NaN stats fall back to the target range
midpoint. `sdf.family`: `"menger"` (uses `scale`) or `"mandelbulb"` (uses
`power`); `sdf.grid`: SDF lattice per mini-SDF; `sdf.max_cells`: max number
of mini-SDF objects per mosaic (larger rasters are deterministically
decimated); `sdf.mapping`: same stat→range mechanism, iterations clamped to
`round(grid/6)` so the lattice resolves the fold features.

`spec_version` stays 4 (additive extension documented here per spec; never
bump for new keys).

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
| `/api/browse` | GET | File picker (HTMX fragment, confined to allowed roots) |
| `/api/jobs` | POST | Submit new scan job |
| `/api/import` | POST | Import an existing scan directory into the job DB |
| `/api/jobs/{job_id}` | GET | Job status JSON |
| `/jobs` | GET | Job list page (scans, compares, renders) |
| `/jobs/{job_id}` | GET | Job progress page (HTMX polling) |
| `/api/jobs/{job_id}/rescan` | POST | Re-run the full scan pipeline for a job |
| `/api/jobs/{job_id}/render/{renderer}` | POST | Enqueue a render job (e.g. `blender`, `preview`) |
| `/models/{job_id}` | GET | Model detail: tabbed sub-pages (`?tab=` overview/sheets/terrain/stats/spec, `?page=` stats pagination) |
| `/api/models/{job_id}/fingerprint` | GET | Fingerprint JSON |
| `/api/jobs/{job_id}/status` | GET | HTMX partial: status badge + progress bar |
| `/compare` | GET | Compare page (select two models) |
| `/api/compare` | POST | Submit new compare job |
| `/compare/{job_id}` | GET | Compare report (delta visualizations + metrics) |

### Directory structure
```
src/weight_atlas/api/
├── __init__.py
├── main.py          # FastAPI app factory (+ QueryError handler)
├── jobs.py          # JobQueue (SQLite + worker)
├── routes.py        # HTTP routes (web UI)
└── query_routes.py  # LLM query API routes (/api, /api/model/*)

src/weight_atlas/ui/
├── templates/       # Jinja2 templates
│   ├── base.html
│   ├── models.html
│   ├── detail.html          # slim shell: tabbed sub-page partials
│   ├── _model_tabs.html     # tab nav + HTMX lazy fragment loader
│   ├── _model_overview.html # overview + fingerprint highlights
│   ├── _model_sheets.html   # sheet/field PNGs
│   ├── _model_terrain.html  # Blender terrain renders
│   ├── _model_stats.html    # server-paginated tensor stats (200/page)
│   ├── _model_spec.html     # spec summary
│   ├── jobs.html            # job list (scans, compares, renders)
│   ├── _job_status.html     # status badge + progress bar partial
│   ├── _file_browser.html   # file picker fragment (/api/browse)
│   ├── job_progress.html
│   ├── compare.html
│   └── compare_report.html
└── static/
    └── style.css
```

### Running the web UI
```bash
uv sync --extra web
uvicorn weight_atlas.api.main:app --reload
# Open http://localhost:8000
```

## Web API / LLM Query API (v0.2)

Machine-readable read endpoints for LLM agents, mounted alongside the web UI
(spec: `docs/2026-08-16_weight-atlas-api-spec-v0.2.md`). The web UI stays the
interface for humans; the API is the interface for agents.

### Design contracts

- **model_id == job_id**: a "model" in the API is any DONE scan job whose
  `out_dir/fingerprint.json` exists. No separate model registry — the job DB is
  the source of truth.
- **Read-side engine is pure**: `api/query.py` holds the analytics (records,
  baselines, slices, anomalies, histograms, deltas, discovery/schema) as pure
  functions over `fingerprint.json`; `api/query_routes.py` is a thin
  `APIRouter` factory mapping query params → engine calls.
- **Determinism is a feature**: fixed ordering, no timestamps in analytical
  output, floats rounded to 4 decimals; `/query` caps at 500 rows with
  `has_more`/`next_offset` pagination.
- **Fingerprint caching**: parsed fingerprints are cached keyed by
  `(path, mtime_ns, size)` (max 16 entries), so re-scans are picked up without
  reloading multi-MB JSON per request.
- **Error envelope**: every error is `{error: {code, type, message, hint}}`
  raised as `QueryError` in the engine and handled in `main.py`.
- **Derived type labels**: `slot` is the authoritative grouping (raster
  columns); `type` is a display label (`self_attn.q_proj`, `mlp.gate_proj`,
  `expert.{id}.{gate|up|down}_proj`, …) via `_SLOT_TYPE` + `derive_type`.
  Type filters use prefix semantics (`self_attn` matches `self_attn.q_proj`).
- **Slice grammar**: dot-concatenated `key:value` predicates
  (`layer:42.type:mlp.gate_proj`); splitter only breaks on dots followed by a
  known key so dotted type values survive.
- **Tiered `/delta`**: tier 1 (weight-space) when a DONE paired/edit compare
  job pairs the two scans (`model_path == "dir_a|dir_b"` + `preset == "edit"`
  in `compare_summary.json`); else tier 2 diffs fingerprint statistics.

### Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/api` | GET | Self-description (discovery) |
| `/api/schema` | GET | Machine-readable field schema |
| `/api/models` | GET | List all completed scans |
| `/api/model/{model_id}` | GET | Scan metadata + baseline |
| `/api/model/{model_id}/summary` | GET | Model-wide aggregates (group by type/layer) |
| `/api/model/{model_id}/layer/{n}` | GET | All tensors in one layer, intra-layer comparison |
| `/api/model/{model_id}/anomalies` | GET | Statistically unusual tensors (p99 default) |
| `/api/model/{model_id}/query` | GET | Filtered, sorted, paginated tensor list |
| `/api/model/{model_id}/compare` | GET | Two slices within one model |
| `/api/model/{model_id}/histogram` | GET | Distribution of a metric |
| `/api/model/{model_id}/tensor/{name}` | GET | Full detail for one tensor |
| `/api/model/{model_id}/delta` | GET | Cross-scan comparison (weight-space tier preferred) |

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
  "spec_version": 4,
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
The `atlas_spec.v2.4.json` may include a `compare` block:
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
`spec_version` stays 4 (additive extension documented here per spec).
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
its own `ssm_ba` slot. The MTP draft head (`blk.N.nextn.*`, Qwen3-Next
multi-token prediction block) follows the same pattern: `eh_proj` →
`mtp_eh_proj`, `enorm`/`hnorm`/`shared_head_norm` → their own `mtp_*` slots,
so MTP weights are no longer reported unmapped. GPT-OSS / Qwen3-Next
per-layer attention-sink registers (`blk.N.attn_sinks`, a small `[64]`
sink vector per layer) map to their own `attn_sinks` slot.

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

## Embedding Projection (M7)

### Data flow
```
token_embd / embed_tokens ─► project ─► embedding_{pca,umap}.npy
                                        ├─► field_embed_density_{raw,smooth}.tif (PCA)
                                        ├─► embedding_scatter.npy (PCA, subsampled)
                                        └─► embedding_meta.json
```

### Design decisions

- **Spec-driven**: `spec.embedding` block (`method` pca|umap, `grid`, `components`,
  `subsample_scatter`, `seeds`). Default method is `pca`; UMAP requires the
  `umap` extra (`umap-learn`) and is imported lazily in `scan.py`.
- **PCA is dependency-free and deterministic**: randomized SVD with seeded RNG
  (`spec.embedding.seeds.pca`). Embedding tensor found by name suffix
  (`model.embed_tokens.weight` / `token_embd.weight`, incl. prefixed VLM names
  such as Kimi K3's `language_model.model.embed_tokens.weight`).
- **Density field**: PCA projects the (V, D) embedding matrix, then the first
  two components are binned into a fixed `grid × grid` density field and
  written as `field_embed_density_{raw,smooth}.tif` (log1p + upsample +
  smooth, same pipeline as channels). UMAP saves only the projected
  coordinates (2-D), no density field.
- **Scatter overlay**: `embedding_scatter.npy` (subsampled, seeded RNG) is
  overlaid on the sheet when the sheet renderer runs against the density
  field — the PCA footprint is drawn as white dots over the density raster.
- **Metadata**: `embedding_meta.json` records method, explained variance,
  `n_components`, sign convention, and scatter subsample/seed.

## Quantization Impact (M9)

### Data flow
```
weights (A) ────────────────┐
                             ├─► impact.py (name-level pairing, chunked
weights (B) ────────────────┘      float64 metrics, jobs=N)
                                     │
                                     ├─► field_impact_<metric>_{raw,smooth}.tif
                                     ├─► field_qtype_raw.tif + qtype_map.json
                                     ├─► field_expert_impact_<slot>_<metric>_*.tif
                                     ├─► field_vision_impact_<metric>_*.tif
                                     ├─► impact_summary.json
                                     └─► impact_*.png (fixed-anchor sheets)
                                                     │
                           compare --noise-floor ────┘ (sub-floor delta veil)
```

### Design decisions

- **Pairing is a name-level join** (NOT `align.py` field alignment): tensors pair
  on `map_name(name)` → identical `(layer, slot)` plus matching tensor name.
  Same `(layer, slot)` with different names across formats (GGUF `blk.N.attn_q`
  vs HF `self_attn.q_proj`) pair via slot+layer and record both names.
  Tensors present on one side only (MTP head, attn_sinks, vision tower) are
  `"skipped"` (never a shape-mismatch crash). Shape mismatch on a paired tensor
  → ValueError listing the tensor. Expert tensors key on
  `("expert", layer, moe_slot, expert_id)`; vision tensors on their block/slot;
  non-layer tensors (embed, lm_head) pair by slot with `layer=None` encoded as
  -1 in the sortable join key.
- **Metrics**: `sqnr_db` (10·log10(‖A‖²/‖B−A‖²)), `rel_l2`, `cos`, `zflip`
  (zero-ness flip fraction), `dmax` (max |Δ|); `dspec` (spectral norm of B−A via
  seeded rSVD, `spec.seeds.svd`) is opt-in via `qimpact.operator_impact` and is
  never emitted as all-NaN fields when disabled.
- **Reference side**: `--ref-side a|b` picks the signal energy in the
  SQNR/rel-L2 numerator (difference energy is symmetric). `sqnr_db`/`rel_l2`
  are asymmetric by definition.
- **Chunked float64 accumulation**: per 1M-element chunk (`qimpact.chunk_size`)
  accumulators for Σa², Σb², Σab, Σd², max|Δ|, zero-flip count. Sequential
  chunk order per tensor; tensor-level parallelism via `jobs=N` thread pool
  (threadpoolctl pins BLAS to 1 thread). Byte-identical for any `jobs`.
- **Strict-only**: impact measurement hard-rejects any non-`"strict"` mode
  with ValueError (identical tensor shapes and layer indices required). Use the
  compare subcommand for aligned/cross-architecture comparison.
- **Fixed-anchor sheets**: impact sheets bypass `filled_norm` and
  `per_row_normalize`; `qimpact.db_range` (default [5, 60] dB) maps directly to
  the `qimpact.colormap` (default `magma_r`), preserving absolute dB anchors
  across models. Title appends `· q-impact`. Profile strip = 1×L per-layer
  median rel-L2 (hot colormap). qtype map = discrete tab20 map of the
  non-reference side's quantization types.
- **Fingerprint dtype**: scan fingerprints record a per-tensor `dtype` for all
  handles (not just GGUF `ggml_type`), so impact summaries can report type maps
  for safetensors reference models too.
- **Determinism**: all TIFFs, PNGs (fixed metadata), summary JSON, and the
  SHA-256 manifest are byte-identical for identical inputs. PNG metadata fixed
  (`Software: weight-atlas`, `Creation Time: 1970-01-01T00:00:00Z`).
- **Noise-floor veil**: `compare --noise-floor CALIB_DIR` reads the calibration
  compare job's `field_delta_<channel>_raw.tif`; cells where current
  |delta| ≤ calibration |delta| get a grey veil (alpha=0.25, `Greys` cmap) on
  the delta sheet, and the title appends `· noise-floor veiled`. The mask is
  computed on the raw full-width grid and aligned to column-dropped sheets.
  Compare jobs always emit `field_delta_<channel>_{raw,smooth}.tif` so any
  compare can serve as a noise-floor calibration source.

### Spec extension
The `atlas_spec.v2.4.json` may include a `qimpact` block (canonical-only,
like `name_map` — older spec versions simply lack the key):
```json
{
  "qimpact": {
    "metrics": ["sqnr_db", "rel_l2", "cos", "zflip", "dmax", "dspec"],
    "operator_impact": false,
    "db_range": [5, 60],
    "colormap": "magma_r",
    "profile_strip": true,
    "type_map": true,
    "chunk_size": 1048576
  }
}
```
`spec_version` stays 4 (additive extension documented here per specs policy).

### Edit preset (`--preset edit`, edit signatures / abliteration)

`run_paired` gained a second preset. `qimpact` remains the default; `edit`
measures the weight-space delta B−A and classifies *what kind of edit* a model
difference is. CLI: `weight-atlas paired SCAN_A SCAN_B --preset edit [--weights-a W] [--weights-b W]`.
The `qimpact` subcommand stays available as an alias of `paired`.

- **Δ-spectrum**: per pair, `dspec` (spectral norm of B−A), `delta_stable_rank`
  (‖Δ‖_F²/σ₁(Δ)², i.e. the participation ratio of the delta spectrum),
  `spectral_share` (σ₁(Δ)²/‖Δ‖_F²), and `rel_l2`/`cos` on the weights. The
  delta spectrum always computes for the edit preset (unlike quant's opt-in
  `operator_impact`), because classification needs it. Reuses
  `stats/spectrum.py` `spectrum_of_matrix`/`top_left_singular_vector`
  (exact SVD ≤ 512 else seeded Halko k=16 q=2, serialized behind
  `_spectrum_lock`).
- **Classification** (`edit_signature.classification`, first-match-wins):
  - `identical`: no tensor exceeds `band_floor` rel-L2.
  - median `delta_stable_rank` over edited tensors ≤ `rank_low` (default 2):
    - `band_mass_share ≥ band_mass_share` (default 0.7) → `low_rank_localized`
      (abliteration-like: rank-1-ish Δ concentrated in a layer band).
    - else → `low_rank_diffuse`.
  - full-rank: no bands → `full_rank_uniform` (quantization/rounding-like);
    else → `diffuse` (full finetune with layer localization).
- **Edit bands**: per-layer median rel-L2 over *edited* tensors only (unedited
  layers count 0, so a slot-concentrated edit stands out); layers above
  `max(band_floor, band_threshold_factor × all-layer median)` form contiguous
  bands. Each band records `start_layer`/`end_layer`/`n_layers` and the
  concentrated slots (per-slot within-band median > band median). `band_mass_share`
  = band layer mass / total layer mass.
- **u1 coherence** (opt-in `edit.u1_coherence`): per pair, the top left
  singular vector of the delta; across edited tensors sharing an output dim,
  the mean pairwise cosine. Sign-fixed (largest-|component| positive, same as
  `embedding/pca.py`) so the comparison is meaningful. Δ-spectrum fields stay
  non-NaN when disabled.
- **Noise floor**: `_noise_floor_policy` compares loader + per-tensor `dtype`
  fingerprints; `identical` policy means the pair shares a dequant pipeline
  (edit signal trustworthy); `mismatched` appends a warning that the signal
  may be at/below quantization noise. Recorded in `noise_floor` + `warnings`.
- **Output**: edit preset writes `compare_summary.json` (body adds `preset`,
  `edit_signature` {classification, stats, bands, `hotspot_ranking_rel_l2`},
  `noise_floor`) plus `field_edit_*` TIFFs and `edit_*.png` sheets
  (rel-L2 log-anchored via `edit.rel_l2_log_range`, Δ stable-rank via
  `edit.rank_log_range`, per-layer profile strip). The paired sheet renderer
  caps the raster to `_MAX_RENDER_PIXELS` so huge smooth fields stay
  cheap to draw.
- **Spec block** (canonical-only, like `qimpact`):
```json
{
  "edit": {
    "metrics": ["rel_l2", "cos", "dspec", "delta_stable_rank", "spectral_share"],
    "u1_coherence": false,
    "rank_low": 2.0,
    "band_threshold_factor": 3.0,
    "band_floor": 1e-4,
    "band_mass_share": 0.7,
    "rel_l2_log_range": [-4, -0.5],
    "rank_log_range": [-1, 3],
    "colormap": "magma_r",
    "profile_strip": true,
    "chunk_size": 1048576
  }
}
```
`spec_version` stays 4 (additive extension).

### impact_summary.json Schema
```json
{
  "ref_side": "a",
  "model_a": { "loader": "...", "n_tensors": 0, "quantization": {} },
  "model_b": { "loader": "...", "n_tensors": 0, "quantization": {} },
  "alignment": { "mode": "strict", "n_pairs": 0, "n_skipped": 0,
                 "skipped": [ { "name": "...", "side": "a", "reason": "not in B" } ] },
  "global": { "median_sqnr_db": 0.0, "p05_sqnr_db": 0.0, "median_rel_l2": 0.0 },
  "per_type": { "ggml_2": { "n": 0, "median_sqnr_db": 0.0 } },
  "hotspot_ranking": [ { "layer": 0, "slot": "...", "name_a": "...",
                         "name_b": "...", "sqnr_db": 0.0, "rel_l2": 0.0 } ],
  "warnings": []
}
```


## Extras (lazy)

`umap` extra is declared (only in `.[umap]`) and `embedding/umap.py` is
imported lazily inside `scan.py`; imports must stay out of core.
