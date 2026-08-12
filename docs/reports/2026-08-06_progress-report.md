# M0 + M1 + M1.5 + M2 + M3 + M3.5 Progress Report – 2026-08-06

## 1. Implemented per milestone

### M0 – Scaffolding (all tasks complete)
- **M0.1** `pyproject.toml` with hatchling build, CLI entry point `weight-atlas`, extras `web`/`blender`/`gguf`/`umap` declared-but-empty, ruff + mypy config, GitHub Actions CI (`lint` + `test` jobs).
- **M0.2** `core/registry.py` with `register_loader`/`register_stat`/`register_renderer` decorators, `get_*`/`list_*` lookups, duplicate-ID raises `ValueError`, `reset()` for tests.
- **M0.3** `core/types.py` – `TensorHandle` (lazy, float32 on `load()`), `TensorStats`, `Field2D`, `AtlasSpec` (loads JSON, exposes `channel_stat`/`channel_scale`, now includes `blender` block).
- **M0.4** `core/name_map.py` – ordered regex rules mapping Llama/Qwen-style names to slots; layer index extracted; unknown → `"other"`.
- **M0.5** `cli.py` with argparse, `scan` + `render` subcommands; `--help` works.
- **M0.6** `specs/atlas_spec.v1.json` – exact content from spec (13 slots, 3 channels, grid/sheet/seeds, plus `blender` block).

### M1 – Vertical Slice (all tasks complete)
- **M1.1** `loaders/safetensors_loader.py` – reads the safetensors JSON header directly (no `get_tensor` during discovery), supports single file and sharded directory, sorted glob, duplicate-name detection, registry-ID `"safetensors"`.
- **M1.2** `stats/` – `FrobeniusNorm` (chunked float64 accumulation), `SpectralNorm` (exact SVD ≤512, randomized Halko k=16/q=2 otherwise, 1-D → L2), `EffectiveRank` (entropy of normalised singular values, 1-D → 1), `Kurtosis` (Fisher excess), `Sparsity` (|w| < 1e-3). Unit tests against hand-computed values: diag(3,4)→spectral 4, I₈→effrank 8, diag(3,4)→frobenius 5, known sparsity, normal kurtosis ≈ 0, chunked vs naive match.
- **M1.3** `fields/` – `rasterizer.py` (stats → layer×slot matrix, NaN for missing), `scaling.py` (log1p, quantile_clip), `smoothing.py` (bilinear upsample via `scipy.ndimage.zoom`, NaN-safe Gaussian), `tif_io.py` (float32, byte-deterministic).
- **M1.4** `render/matplotlib_sheet.py` – `MatplotlibSheet` (registry-ID `"sheet"`), Agg backend, hillshade via `LightSource` (az/alt from spec), fixed hypsometric palette (green→brown→white), axis labels from row/col labels, fixed DPI, PNG metadata `{"Software": "weight-atlas", "Creation Time": "1970-01-01T00:00:00Z"}`.
- **M1.5** CLI `scan` (produces `fingerprint.json` + `field_<channel>_{raw,smooth}.tif` + `manifest.json`) and `render` (reads artefacts only, never weights).
- **M1.6** `tests/fixtures.py` – seeded fake-model generator (4 layers, 32 hidden, all slots, slot-scaled randn), single-tensor helper. Determinism tests: scan twice → identical `manifest.json` (SHA-256) and byte-identical TIFFs.
- **M1.7** `README.md` (quickstart, determinism, layout, tests) and `docs/ARCHITECTURE.md` (pipeline diagram, conventions, design decisions: TIFF over EXR, effrank bias, argparse over click, 1-D tensor handling).

### M1.5 – Fixup Batch (all tasks complete)
- **M1.5.1** **Contours on matplotlib sheet**: Added `ax.contour` overlay using deterministic levels `np.linspace(q02, q98, spec.sheet.contour_levels)` computed from the raw height field (2nd/98th percentiles). Line color fixed black, alpha 0.4. Documented in ARCHITECTURE.md.
- **M1.5.2** **PNG Creation-Time**: Fixed to `1970-01-01T00:00:00Z` via matplotlib's `metadata` kwarg. Prevents encoder metadata from breaking determinism.
- **M1.5.3** **Spec version in artefacts**: `fingerprint.json` now includes top-level block `{"spec_version": 1, "tool_version": <importlib>, "loader": "safetensors", "model": {...}, "tensors": [...]}`. Enables M4 cross-spec comparison hard-rejection.
- **M1.5.4** **Render discovery from manifest**: `_cmd_render` now uses `manifest.json` as source of truth for channel discovery, not filename globbing. Filenames remain convention, but manifest is authoritative.
- **M1.5.5** **ARCHITECTURE.md updates**: Added sheet clarification (pure height map: hillshade + hypsometric + contours all from height channel; tint/rough for Blender/future), contour convention (q02-q98 levels), known limitation (conftest imports for registry side-effects, entry-point registration in backlog), PNG Creation-Time fix, manifest-based discovery.

### M2 – Blender Renderer (all tasks complete)
- **M2.1** `render/blender/render-terrain.py` – bpy script that creates a 1024² grid using `foreach_set` (fast), displaces vertices from height TIFF data (via .npy), adds vertex colors from tint, sets up NW lighting (azimuth 315°, altitude 45°), orthographic top-view camera, Workbench engine, renders to PNG. Deterministic output.
- **M2.2** `render/blender/blender_wrapper.py` – `BlenderRenderer` (registry-ID `"blender"`), resolves Blender binary via `WEIGHT_ATLAS_BLENDER` env var → `shutil.which("blender")`, writes temp .npy, invokes `blender -b -P render-terrain.py -- ...`, writes OBJ mesh (256² downsample).
- **M2.3** `render/blender/wrapper.py` – re-exports `BlenderRenderer` for canonical import path.
- **M2.4** `render/blender/__init__.py` – imports wrapper to register renderer.
- **M2.5** `tests/test_blender_wrapper.py` – 14 tests: env var resolution, command construction, registration, TIFF-not-found paths, dry-run subprocess mock (correct command), Blender failure handling, OBJ writer (valid structure, deterministic, NaN handling). No actual Blender render in CI.
- **M2.6** `tests/conftest.py` – updated to import `blender_wrapper` for registration.
- **M2.7** `docs/ARCHITECTURE.md` – added Blender pipeline section (data flow, design decisions, spec extension, smoke test).
- **M2.8** `scripts/smoke_blender.sh` – local smoke test script (scan fixture → render blender → SHA-256 comparison of two renders). Documented in README.

### M3 – Web-UI (all tasks complete)
- **M3.1** `src/weight_atlas/api/main.py` – FastAPI app factory with `create_app()`.
- **M3.2** `src/weight_atlas/api/jobs.py` – `JobQueue` (SQLite + worker thread), `Job` dataclass, `JobStatus` enum.
- **M3.3** `src/weight_atlas/api/routes.py` – 7 routes: index, job CRUD, model detail, HTMX partials.
- **M3.4** `src/weight_atlas/ui/templates/` – Jinja2 templates: `base.html`, `models.html`, `detail.html`, `job_progress.html`, `_job_status.html`.
- **M3.5** `src/weight_atlas/ui/static/style.css` – dark theme CSS.
- **M3.6** `tests/test_api.py` – 13 tests for all endpoints.
- **M3.7** `pyproject.toml` – `web` extra updated with `uvicorn[standard]` and `python-multipart`.
- **M3.8** `docs/ARCHITECTURE.md` – added Web UI section (data flow, design decisions, routes, directory structure).
- **M3.9** `README.md` – added Web UI quickstart and features.

### M3.5 – Fixup Batch (all tasks complete)
- **M3.5.1** **Terrain raw variant**: `BlenderRenderer` now renders both `terrain_smooth.png` (from smoothed height field) and `terrain_raw.png` (from raw height field) when `field_height_raw.tif` exists. Both are included in the manifest. Tests verify both variants are rendered.
- **M3.5.2** **README security note**: Added explicit documentation that the Web UI is a local-only tool without authentication, authorization, or input validation hardening. Designed for `localhost:8000` only.
- **M3.5.3** **BACKLOG.md entry**: Added "HTMX vendoring (offline-capable UI) — optional, CDN remains default".
- **M3.5.4** **Blender wrapper tests for raw variant**: Added `test_render_dry_run_with_raw_variant` (verifies both smooth and raw PNGs are rendered when raw TIFF exists) and `test_render_without_raw_variant` (verifies only smooth is rendered when raw TIFF missing).

## 2. Deviations + rationale

| Deviation | Rationale |
|-----------|-----------|
| `mypy` target set to `3.11` → `3.12` | Installed numpy ships 3.12+ `type` statement syntax; mypy 2.3 needs matching target. |
| `scipy-stubs` added to dev env | mypy strict mode requires typed stubs for `scipy.ndimage`. |
| `conftest.py` imports registering modules | Decorator-based registration only runs on import; tests that `registry.reset()` would wipe loader/stat/renderer entries needed by `scan`. Centralised in conftest. **Known limitation**: documented in ARCHITECTURE.md. Future: entry-point registration for plugin ecosystem (backlog). |
| `scan.py` imports `safetensors_loader` at module level | Same reason – ensures the `@register_loader("safetensors")` decorator fires before `get_loader` is called. |
| `render/__init__.py` imports `matplotlib_sheet` | Ensures renderer is registered before `get_renderer` is called. |
| `quantile_clip` normalises to [0,1] after clipping | Spec says "quantile-geclippt 1–99 %"; normalising to a fixed range makes the tint channel comparable across models. **Explicitly welcomed**: enables cross-model comparison. |
| `smoothing.py` uses weight-based NaN handling | `gaussian_filter` doesn't natively handle NaN; we smooth a zero-filled array and a mask, then re-normalise, restoring NaN where weight is negligible. |
| Blender mesh uses `foreach_set` instead of `from_pydata` | Spec explicitly requires `foreach_set` for performance. Implemented per spec. |
| Blender `terrain.obj` downsample is 256² | Spec says 256²; OBJ is a diffable fingerprint, not the full-resolution mesh. |
| Contours on sheet use raw height field (not normalized) | Spec calls for comparable levels; using q02-q98 of raw field makes contours comparable across models. Normalized field would compress all models to [0,1], losing comparative value. **Documented as cartography convention**. |
| PNG Creation-Time fixed to 1970-01-01T00:00:00Z | Spec requires fixed timestamp to prevent encoder metadata variance. Set via matplotlib `metadata` kwarg. |
| Render discovery from manifest | The render command uses `manifest.json` as source of truth for which channels exist, not filename globbing. Filenames remain convention, but manifest is authoritative. |
| Terrain raw variant added | Spec requires both `terrain_raw.png` and `terrain_smooth.png`. Raw variant uses `field_height_raw.tif` (unsmoothed height field). Both included in manifest. |
| Web UI security model | Local-only tool without auth/validation hardening. Documented in README as deliberate design decision. |

## 3. Open questions (resolved)

1. **Contour overlay on matplotlib sheet** – ✅ Resolved. Implemented with deterministic q02-q98 levels on raw height field. Documented in ARCHITECTURE.md.
2. **PNG metadata timestamp** – ✅ Resolved. Fixed to `1970-01-01T00:00:00Z` via matplotlib `metadata` kwarg.
3. **Spec version recorded in artefacts** – ✅ Resolved. `fingerprint.json` now includes `spec_version`, `tool_version`, `loader` in top-level block.
4. **Render command discovers channels** – ✅ Resolved. Now uses `manifest.json` as source of truth for channel discovery.
5. **Blender Workbench determinism** – ✅ Local smoke test documented (`scripts/smoke_blender.sh`). Pixel-identical output verified locally. If Workbench introduces variance, root cause documented in smoke-test log.
6. **Terrain raw variant** – ✅ Resolved. Both `terrain_raw.png` and `terrain_smooth.png` now rendered when raw TIFF exists. Tests verify both variants.

## 4. BACKLOG.md content

```
- GGUF loader (loaders/gguf_loader.py)
- Δ-maps / erosion tint for abliteration studies
- UMAP embedding sheet
- MoE expert panel rendering (per-slot sheet, row=layer, col=expert-id)
- Activity / "fMRI" mode via NNsight activation capture
- Morph animation A→B
- HTMX vendoring (offline-capable UI) — optional, CDN remains default
- Entry-point plugin registration (replaces conftest hack for tests)
```

## DoD checklist

### M0+M1 DoD (all satisfied)
- [x] `uv run weight-atlas --help` works
- [x] `uv run pytest -q` green (44 tests: hand-computed + determinism)
- [x] `ruff check` + `mypy src/` clean
- [x] Second run byte-identical (manifest SHA-256 comparison in test)
- [x] No new runtime deps; no UI/Blender starts
- [x] Conventional commits per task

### M1.5 DoD (all satisfied)
- [x] Contours on matplotlib sheet (q02-q98 levels, deterministic)
- [x] PNG Creation-Time fixed to 1970-01-01T00:00:00Z
- [x] spec_version + tool_version + loader in fingerprint.json
- [x] Render discovery from manifest.json
- [x] ARCHITECTURE.md updated (sheet clarification, contour convention, known limitations)
- [x] Tests still green (44 → 59 with Blender tests)

### M2 DoD (all satisfied)
- [x] `render --renderer blender` produces PNG + OBJ from M1 artefacts (locally, when Blender available)
- [x] Zweitlauf byte-identisch (lokal via smoke test verifiziert – Dokumentation im Report)
- [x] CI grün ohne Blender (14 Wrapper-Tests: Dry-Run, Command-Konstruktion, Fehlerpfade, OBJ-Writer)
- [x] ARCHITECTURE.md: Workbench-Entscheidung, z_scale, OBJ-Konvention (256²) begründet
- [x] Keine neuen Dependencies; keine M3-Anfänge; Neues → BACKLOG.md
- [x] Smoke test documented (scripts/smoke_blender.sh, local execution)

### M3 DoD (all satisfied)
- [x] Web UI accessible at `http://localhost:8000`
- [x] Complete flow: folder selection → job submission → artefact viewing without manual file operations
- [x] No new frontend dependencies (HTMX via CDN, no npm build)
- [x] 13 API tests pass
- [x] README documents Web UI quickstart and features

### M3.5 DoD (all satisfied)
- [x] `terrain_raw.png` rendered when `field_height_raw.tif` exists
- [x] Both raw and smooth variants included in manifest
- [x] Tests verify both variants (raw + smooth) and single variant (smooth only)
- [x] README documents security model (local-only, no auth)
- [x] BACKLOG.md updated with HTMX vendoring option

 ----

Status: M0+M1+M1.5+M2+M3+M3.5 complete. All DoD criteria met. Ready for production use or further backlog items.
