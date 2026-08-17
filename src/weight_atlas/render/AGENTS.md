# AGENTS.md — render

## Purpose

Visualise scanned fields: matplotlib topographic sheets (hillshade + tint +
contours), preview thumbnails, and Blender 3D terrain renders (+ OBJ mesh).

## Ownership

- `matplotlib_sheet.py` (registered `"sheet"`), `preview.py` (registered
  `"preview"`), `blender/` (`blender_wrapper.py` + `render_terrain.py`,
  registered `"blender"`). `compare/render/delta_sheet.py` is owned by the
  compare doc.

## Local Contracts

- **Renderer registry**: renderers register by id (`register_renderer`);
  `cli.py` `render` subcommand and api jobs select by id.
- **Determinism is a feature**: sheets, previews, PNGs, and the OBJ export
  must be byte-identical for identical inputs. Workbench engine (no GPU
  sampling noise), fixed world colour, fixed light, no timestamps in PNG.
  Blender stamps every PNG with `Date`/`RenderTime` tEXt chunks; the terrain
  script strips them after rendering (`_strip_png_metadata`), so two renders
  of the same input are byte-identical.
- **Blender is external**: binary resolved from `WEIGHT_ATLAS_BLENDER` env →
  `shutil.which("blender")`; never a pip dependency. Never assert on Blender
  output in this repo's tests — they are dry-run (mocked subprocess.run);
  real renders run on a separate machine via `scripts/smoke_blender.sh`.
  **Blender exits 0 even when the `-P` script crashes** — the wrapper fails
  the render if stderr contains a Python traceback, so stale PNGs can never be
  served as fresh output.
- **Blender terrain**: orthographic camera at fixed pitch (spec
  `blender.pitch`, default 18°) — orthographic projection preserved, so
  renders stay comparable across models. Height is robustly normalised
  (percentile clip `blender.clip`, default 1–99%) so outliers cannot flatten
  the bulk. `adaptive_z_scale` (opt-in) rescales Z to constant relief std —
  purely visual, breaks absolute-amplitude comparability; document as such.
- **Blender geometry smoothing (terrain, not raw values)**: height is
  bilinearly resampled to the grid (`resample_bilinear`, pure NumPy — no scipy
  inside bpy), the mesh is smooth-shaded (auto-smooth 30° cutoff) and
  Catmull-Clark subdivided (`blender.subsurf_levels`, default 1, 0=off).
  Rendering uses Workbench STUDIO with `use_scene_lights`/`use_scene_world`
  enabled so the scene SUNs apply: a NW key (azimuth 315°, 45° alt, energy 1)
  plus a soft SE fill (`blender.fill_light_energy`, default 0.35). Without
  `use_scene_lights` the SUNs are a no-op in the render.
- **Sheet renderer**: the matplotlib sheet is a pure height map (hillshade +
  hypsometric tint + contours from the height channel only). Optional display
  knobs in `spec.sheet` (defaults off): `normalized_depth` (project rows onto
  normalized-depth landmarks, shading interpolated cells) and
  `drop_empty_cols` (drop all-NaN slot columns display-only — never the TIFF
  fields; only fully-empty columns/slot-blocks drop, title lists them).
  `per_row_normalize` (default off) normalizes each row independently.
- **OBJ export**: 256² bilinear downsample (same `resample_bilinear` helper
  as the renderer, not nearest-neighbour), plain text, diffable, uses the
  same normalisation as the PNG render.

## Work Guidance

- Keep Blender-side code (render_terrain.py, runs inside bpy) free of repo
  imports where possible; pass data via `.npy` + `--args`.
- New renderers: register id, add dry-run test, add spec `render.*` keys only
  if reusable.

## Verification

- `tests/test_blender_wrapper.py`, `tests/test_preview_renderer.py`,
  `tests/test_render_cli.py`, `tests/test_determinism.py`. Run via
  `cd /media/data/coding/weight-atlas && .venv/bin/python -m pytest tests/test_blender_wrapper.py tests/test_determinism.py`.