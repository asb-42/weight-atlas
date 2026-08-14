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
- **Blender is external**: binary resolved from `WEIGHT_ATLAS_BLENDER` env →
  `shutil.which("blender")`; never a pip dependency. Never assert on Blender
  output in this repo's tests — they are dry-run (mocked subprocess.run);
  real renders run on a separate machine via `scripts/smoke_blender.sh`.
- **Blender terrain**: orthographic camera at fixed pitch (spec
  `blender.pitch`, default 18°) — orthographic projection preserved, so
  renders stay comparable across models. Height is robustly normalised
  (percentile clip `blender.clip`, default 1–99%) so outliers cannot flatten
  the bulk. `adaptive_z_scale` (opt-in) rescales Z to constant relief std —
  purely visual, breaks absolute-amplitude comparability; document as such.
- **OBJ export**: 256² downsample, plain text, diffable, uses the same
  normalisation as the PNG render.

## Work Guidance

- Keep Blender-side code (render_terrain.py, runs inside bpy) free of repo
  imports where possible; pass data via `.npy` + `--args`.
- New renderers: register id, add dry-run test, add spec `render.*` keys only
  if reusable.

## Verification

- `tests/test_blender_wrapper.py`, `tests/test_preview_renderer.py`,
  `tests/test_render_cli.py`, `tests/test_determinism.py`. Run via
  `cd /media/data/coding/weight-atlas && .venv/bin/python -m pytest tests/test_blender_wrapper.py tests/test_determinism.py`.