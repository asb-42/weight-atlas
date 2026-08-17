## [Unreleased]

### Fractal Terrain Renderer (fBm)

- New `"fractal"` renderer (`weight-atlas render OUT_DIR --renderer fractal`,
  also via `/api/jobs/{id}/render/fractal` + "Render Fractal Terrain" button
  on the model overview): genuine per-slot fBm geometry whose parameters are
  derived from real tensor statistics — effective_rank → octaves, kurtosis →
  persistence, sparsity → lacunarity, spectral_norm → base_freq (spec
  `fractal.mapping`, linear slot-range scaling, NaN → midpoint).
- Pure NumPy value noise on a fixed splitmix64 integer-lattice hash (no RNG,
  no timestamps) → byte-identical `terrain_fractal.png` + `terrain_fractal.obj`
  for identical inputs; the height field *is* the fractal, not a texture.
- Per-slot character: each slot column is its own fBm strip (fixed per-slot
  seed = base seed + slot index), tint is a second independently-seeded strip.
- Rendered through the existing Blender terrain pipeline (`render_terrain.py`,
  same smoothing/lights/metadata-strip), so fractal and plain terrain renders
  are directly comparable.
- One render per model: the fractal depends on the fingerprint + seed, not the
  channel — per-instance dedupe means the per-channel API/CLI loop runs Blender
  once and all channels reuse the identical artefacts (the primary language
  raster's layout, never overwritten by the smaller vision layout).
- New spec keys `fractal` + `seeds.fractal` added to all spec versions
  (v1–v2.4, additive, spec_version unchanged). New `tests/test_fractal_renderer.py`
  (19 tests, dry-run — mocked subprocess).

### Fractal SDF Mode (per-slot mini-SDF mosaic)

- New `fractal.mode = "sdf"` for the `"fractal"` renderer: instead of an fBm
  height field, each slot cell renders its own 3D Menger-sponge or Mandelbulb
  object — a per-slot mosaic of mini-SDFs ("sculpture garden").
- Own deterministic SDF families in pure NumPy (`menger_sdf`, `mandelbulb_sdf`,
  masked bail-out so high-power Mandelbulbs stay finite) extracted with a
  deterministic naive Surface Nets iso-extraction (watertight, outward
  normals) — no external tools (Mandelbulber etc.) that would break the
  byte-identity contract.
- Per-slot SDF parameters (iterations/power/scale ← slot stats via spec
  `fractal.sdf.mapping`) with iteration counts clamped to `round(grid/6)` so
  coarse lattices never alias into empty cells.
- Rendered via new `render_sdf.py` bpy script reusing the terrain pipeline's
  world/light/camera/engine helpers; OBJ export is the full-resolution mosaic
  mesh. Same determinism: byte-identical PNG + OBJ for identical inputs.
- New spec keys `fractal.mode` + `fractal.sdf` added to all spec versions
  (v1–v2.4, additive, spec_version unchanged). `tests/test_fractal_renderer.py`
  extended with SDF/surface-nets/mosaic tests (now 32 tests, dry-run).
- UI toggle: "Fractal mode" `<select>` (fBm/SDF mosaic) next to the Render
  Fractal Terrain button. The selection is sent as a `fractal_mode` form field
  and overlaid onto the job's recorded spec's `fractal.mode` for that render
  only (via `job.sheet_knobs` — the recorded spec is never mutated). The
  fractal dedupe cache key now includes `mode` so fBm and SDF renders of the
  same model never cross-pollinate. New tests for the mode knob in
  `tests/test_api.py` (2 tests).

### Fractal SDF scalability (expert-panel fix)

- **Root cause**: the SDF mosaic builds one mini-SDF per (row × col) raster
  cell. MoE expert panels (`expert_mlp_*`, one column per expert) made the
  first rendered channel a 92×896 = 82,432-cell raster → ~46 min of naive
  Surface Nets extraction (single-core, ~33 ms/cell) and a ~115M-vert mesh
  that crashed Blender with SIGSEGV while loading it (`from_pydata`). The
  job then ended "done" with the failure recorded in `artefacts`.
- **Fixes**: (1) the fractal is now built from the primary language raster
  only — `expert_*`/`vision_*` channels are skipped (`render()` returns `[]`),
  so an expert panel can never define the layout; (2) new spec key
  `fractal.sdf.max_cells` (default 1024) deterministically decimates rasters
  that exceed it with aspect-preserving strides (objects keep their true
  row/col positions and tints) — the expert channel now builds in ~33 s to
  1.4M verts instead of 46 min to 115M verts; (3) the fractal dedupe key now
  includes the layout `(n_rows, n_cols, slot labels)` so channels with
  different rasters can never cross-contaminate the cache.
- Added to all spec versions (v1–v2.4, additive, spec_version unchanged).
  `tests/test_fractal_renderer.py` extended with decimation/determinism,
  expert/vision skipping, and layout-keyed dedupe tests (now 36 tests,
  dry-run).

### Fractal SDF sculpture garden (usable relief + variation)

- **Root cause**: the SDF mosaic normalised *all* axes by the lateral span, so
  ~6.4-unit-tall objects were crushed to z ≈ ±0.009 in a ±1.1 frame — the
  render was a perfectly symmetric grid of flat boxes. Parameters barely
  varied (iterations {1,2,3}, scales ~{2.5,3.0,3.5}) and tint was a flat
  blue→orange column gradient, so the output read as noise-free, symmetric,
  and flat.
- **Fixes**: the mosaic now normalises the lateral footprint into the
  [-1, 1]² frame while keeping real z-relief — objects stand up to
  `fractal.sdf.relief` (default 1.0, *before* the render's `z_scale`
  exaggeration, so relief matches the fBm terrain). `fractal.sdf.variation`
  (default true) breaks grid symmetry via a deterministic per-cell size
  (0.6–1.4) and yaw from the cell-lattice splitmix64 hash (seeded by the
  fractal seed, no RNG). Tint is now meaningful: `fractal.sdf.tint_stat`
  (default `"effective_rank"`) normalises a real per-slot statistic onto
  [0, 1] (`slot_stat_tint`, missing/NaN → 0.5) so each sculpture's colour
  carries the slot's statistic instead of a column gradient.
- New spec keys `fractal.sdf.variation`/`relief`/`tint_stat` added to all spec
  versions (v1–v2.4, additive, spec_version unchanged).
  `tests/test_fractal_renderer.py` extended with relief, variation,
  slot_tint, and `slot_stat_tint` tests (now 40 tests, dry-run).

### Blender Terrain Geometry Smoothing

- Geometry smoothing (terrain, not raw values): height is now bilinearly
  resampled to the render grid (`resample_bilinear`, pure NumPy — no scipy
  inside bpy) instead of nearest-neighbour block-sampling.
- Mesh is smooth-shaded (auto-smooth 30° cutoff) and Catmull-Clark
  subdivided (`blender.subsurf_levels`, default 1, 0=off) so the terrain
  renders as continuous relief rather than flat facets.
- Workbench lighting now enables `use_scene_lights`/`use_scene_world` (the
  scene SUNs were previously a no-op in the render) and adds a soft SE fill
  sun (`blender.fill_light_energy`, default 0.35) lifting the shadow side.
- OBJ export downsample is now bilinear too (same `resample_bilinear` helper).
- New spec keys `subsurf_levels` + `fill_light_energy` added to all spec
  versions (v1–v2.4, additive, spec_version unchanged). Determinism contract
  unchanged — all steps are fixed topology/light operations.
- Fixes found while verifying on a machine with Blender 4.0: `shade_smooth()`
  is a bpy.ops operator on 4.0 (not an Object method) — the script crashed
  every render; fixed via the `mesh.polygons[].use_smooth` flag. The wrapper
  now fails the render on a Python traceback in Blender's stderr (Blender
  exits 0 even when `-P` crashes), so stale PNGs are no longer silently
  served as fresh output. Blender's `Date`/`RenderTime` PNG tEXt chunks are
  stripped after rendering so two renders are truly byte-identical.

### LLM Query API (v0.2)

- New read-only REST layer for LLM agents (spec
  `docs/2026-08-16_weight-atlas-api-spec-v0.2.md`, implemented): `api/query.py`
  pure read-side engine + `api/query_routes.py` APIRouter mounted at `/api`.
  `model_id` == DONE scan job ID; a model is any DONE job with
  `fingerprint.json`.
- Endpoints: `/api` (self-description), `/api/schema`, `/api/models`,
  `/api/model/{id}` + `.../summary`, `.../layer/{n}`, `.../anomalies`,
  `.../query`, `.../compare`, `.../histogram`, `.../tensor/{name}`, `.../delta`.
- Deterministic bodies: fixed ordering, no timestamps in analytics, floats
  rounded to 4 decimals; `/query` caps at 500 rows with `has_more`/`next_offset`
  pagination and `fields` column trimming. Parsed fingerprints cached by
  (path, mtime_ns, size), max 16 entries.
- Unified error envelope `{error: {code, type, message, hint}}` via `QueryError`
  (handled in `main.py`).
- Tiered `/delta`: tier 1 weight-space when a DONE paired/edit compare job pairs
  the two scans, else tier 2 fingerprint statistic diff.
- New `tests/test_query_api.py` (31 tests) covering discovery, metadata, layer,
  anomalies, query filtering/sorting/pagination, compare slices, histogram,
  tensor detail, and both delta tiers.

### Quantization Impact (M9)

- New `weight-atlas qimpact` subcommand: measures per-tensor quantization
  impact (sqnr_db, rel_l2, cos, zflip, dmax; dspec opt-in) between two weight
  snapshots via name-level pairing across formats (GGUF `blk.N.*` ↔ HF
  `model.layers.N.*`). Strict-only — `--mode aligned` raises ValueError.
- Artefacts: `field_impact_<metric>_{raw,smooth}.tif`, `field_qtype_raw.tif`
  + `qtype_map.json`, expert/vision impact fields, `impact_summary.json`
  (global stats, per-type medians, top-5 hotspot ranking), fixed-anchor
  `impact_*.png` sheets (bypass `filled_norm`/`per_row_normalize`), and a
  SHA-256 `manifest.json`. Byte-identical for any `--jobs`.
- `compare --noise-floor CALIB_DIR`: grey veil over cells whose |delta| is at
  or below the calibration compare's |delta|; compare jobs always emit
  `field_delta_<channel>_{raw,smooth}.tif` as veil source.
- `specs/atlas_spec.v2.4.json`: new canonical-only `qimpact` block
  (spec_version unchanged — additive extension).
- Scan fingerprints record per-tensor `dtype` for all handles.
- `gguf_dequant.py`: Q4_0 dequantisation now uses the canonical block layout
  (first 16 values in low nibbles, last 16 in high nibbles); pinned by
  `test_q4_0_canonical_layout`. Previously interleaved `(2j, 2j+1)` — real
  Q4_0 files would dequantize scrambled.

### Edit Signatures (M9, `--preset edit`)

- `weight-atlas qimpact` generalized into `paired` with presets: the
  `qimpact` subcommand is preserved as an alias of `paired --preset quant`
  (default). New `paired --preset edit` measures the weight-space delta
  B−A (rel_l2, cos, dspec, delta_stable_rank, spectral_share) and classifies
  the edit kind.
- `edit_signature.classification` decision tree: `identical` → `low_rank_localized`
  (abliteration-like, band mass share ≥ 0.7) / `low_rank_diffuse` →
  `full_rank_uniform` (quantization-like) / `diffuse`. Edit bands group
  contiguous layers whose edited-tensor median rel-L2 exceeds the noise-floor
  threshold; each band lists its concentrated slots.
- Opt-in `edit.u1_coherence`: mean pairwise cosine of the delta's top left
  singular vector across edited tensors sharing an output dim (sign-fixed to
  the pca convention). Shared `stats/spectrum.py`
  `spectrum_of_matrix`/`top_left_singular_vector` (exact ≤512 else seeded
  Halko rSVD, serialized behind `_spectrum_lock`).
- Noise-floor policy: compares loader + per-tensor `dtype` fingerprints;
  `mismatched` appends a warning that the edit signal may be at/below
  quantization noise. Recorded in `noise_floor` + `warnings`.
- Output: `compare_summary.json` (adds `preset`, `edit_signature`
  {classification, stats, bands, `hotspot_ranking_rel_l2`}, `noise_floor`),
  `field_edit_*` TIFFs, and `edit_*.png` sheets (log-anchored rel-L2 and Δ
  stable-rank, per-layer profile strip). Paired sheet renderer caps the
  raster to a pixel budget so large smooth fields stay cheap to draw.
- `specs/atlas_spec.v2.4.json`: new canonical-only `edit` block (spec_version
  unchanged — additive extension).

### Name Mapping — Gemma-4 "ultra/heretic" (MoE) support

- `core/name_map.py`: new GGUF rules for Gemma-4's extra per-layer tensors:
  `pre_ffw_norm_2`, `post_ffw_norm`, `post_ffw_norm_1`, `post_ffw_norm_2`
  (dedicated norm slots) and `layer_output_scale` (per-layer gain); global
  `rope_freqs` maps to its own slot.
- `core/name_map.py`: vision rule for the Gemma-4 mmproj input projection
  (`mm.input_projection` → `mm_projector`).
- `specs/atlas_spec.v2.4.json`: new slots `pre_ffw_norm_2`, `post_ffw_norm`,
  `post_ffw_norm_1`, `post_ffw_norm_2`, `layer_output_scale`, `rope_freqs`
  (spec_version unchanged — additive slot extension).
- New fixture `tests/fixtures/names_gemma4_heretic_gguf.json` + audit tests.
- Verified on `gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_S.gguf`
  (4468/4468) and `gemma-4-26B-A4B-it-mmproj-BF16.gguf` (356/356): mapping
  coverage 100% (was 96.9%).

## [0.2.0] - 2026-08-09

### Real-Model-Calibration (Bonsai-8B)

**Migration Note**: v0.2.0 requires re-scan. Existing v1 fingerprints cannot be compared with v2. The compare command hard-rejects spec_version mismatches.

#### Spec v2 Changes
- `spec_version`: 2 (was 1)
- `tint` channel: `effective_rank` → `stable_rank` = log1p((frobenius/spectral_norm)²)
- `rough` channel: `log1p` → `quantile_clip` (1–99%)
- `height` channel: unchanged (spectral_norm + log1p)
- New slots: `attn_q_norm`, `attn_k_norm` (QK-Norm for Bonsai-8B)

#### Name Audit System
- Bonsai-8B fixture: `tests/fixtures/names_bonsai_8b.json`
- `diagnose` CLI command: `weight-atlas diagnose <path>`
- `mapping_coverage` block in fingerprint.json
- CLI warning when `in_slots < 80%`
- Rule ordering: `attn_q_norm`/`attn_k_norm` before `attn_q`/`attn_k`

#### Degeneration Guards
- Per-channel diagnostics: `valid_fraction`, `normalized_std`
- Std < eps OR valid < 50% → CLI warning + warnings block + UI banner
- Module: `weight_atlas/fields/degenerations.py`

#### UI Changes
- Artifact route: `GET /models/{id}/artifacts/{name}` (allowlist + traversal protection)
- `.tif` entries shown as "not inline displayable — download" link
- Detail page uses only the artifact route
- Scan job auto-renders sheets after completion
- Degeneration warnings banner on detail page

#### New Files
- `specs/atlas_spec.v2.json`
- `src/weight_atlas/stats/stable_rank.py`
- `src/weight_atlas/fields/degenerations.py`
- `tests/test_name_audit.py`
- `tests/test_degenerations.py`
- `tests/fixtures/names_bonsai_8b.json`
- `docs/MODEL_FAMILIES.md`

# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-08-08

### Three Core Guarantees

1. **Artifacts are renderer-independent and canonical**: All outputs (TIFF, PNG, JSON, OBJ) follow a versioned specification (`atlas_spec.v1.json`). Renderers never access raw weights — they consume only artifacts.
2. **Render/Compare never read weights**: The entire visualization and comparison pipeline operates on artifacts (statistics, fields, projections). Raw model weights are never loaded by renderers or comparators.
3. **Determinism is part of the measurement protocol**: All random number generators are seeded from the spec. Byte-identical outputs are guaranteed on the same machine (verified by SHA-256 manifest and second-run tests).

### Milestone Summary

- **M0 — Scaffolding**: Project structure, CLI entry point, plugin registry, core types, name mapping, spec file. CI with GitHub Actions.
- **M1 — Vertical Slice**: Safetensors loader, statistics (Frobenius, spectral norm, effective rank, kurtosis, sparsity), rasterizer, scaling, smoothing, matplotlib sheet renderer.
- **M1.5 — Fixup Batch**: Contours on raw height field, fixed PNG creation time, spec_version/tool_version/loader in fingerprint, render discovery from manifest.
- **M2 — Blender Renderer**: 1024² grid terrain rendering, OBJ mesh export, ortho top-view camera, NW lighting, worldbench engine.
- **M3 — Web UI**: FastAPI app, SQLite-backed job queue, Jinja2 templates, HTMX polling, model list/detail/compare pages.
- **M3.5 — Fixup**: Raw + smooth terrain in manifest, security note in README, HTMX vendoring backlog entry.
- **M4 — Comparison/Delta Layer**: Strict and aligned modes, delta fields on scaled values, summary metrics on raw stats, hotspot ranking, hard-reject on spec_version mismatch, warning on tool_version mismatch.
- **M5 — GGUF Loader**: GGUFReader with mmap, lazy tensor handles, F32/F16/BF16/Q8_0/Q4_0 dequantization, auto-detect via magic bytes, cross-loader comparison with warnings.
- **M6 — MoE Expert Panel**: ExpertPanel field class (Layer × Expert), main raster unchanged, shared expert to mlp slots, GGUF 3D stacked tensor splitting, panel comparison with skip-on-mismatch.
- **M7 — Embedding Sheet**: PCA projection with sign convention (max |loading| positive), density field via 2D histogram, UMAP support (optional extra), scatter overlay on sheets.
- **M8 — Activity Mode ("fMRI")**: Forward-pass activation capture via PyTorch hooks, versioned stimulus protocol (8 states), Layer×Position residual RMS fields, Layer×Expert usage fields for MoE, scanner discipline (threads=1, deterministic algorithms).

### Key Conventions

- **Spec version stays 1**: All extensions (blender, compare, embedding, activity) are additive. No breaking changes to scan output.
- **Slots**: 13 fixed slots (embed, attn_q/k/v/o, mlp_gate/up/down, norm_attn/mlp, router, lm_head, other).
- **Channels**: height (spectral_norm + log1p), tint (effective_rank + quantile_clip), rough (kurtosis + log1p).
- **Determinism**: Seeds from spec, threads=1 for activity, SHA-256 manifest per artifact.
