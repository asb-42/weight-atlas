# AGENTS.md — render/fractal

## Purpose

Statistics-driven fractal terrain renderer (registered `"fractal"`): genuine
fractal geometry whose per-slot parameters derive from real tensor statistics,
rendered through the shared Blender terrain pipeline. Two modes: fBm height
field (default) and a per-slot mosaic of mini-SDFs (Menger/Mandelbulb).

## Ownership

- `fbm.py` — deterministic value-noise fBm (`_hash_lattice` splitmix64,
  `value_noise`, `fbm`, `terrain_field`, `slot_fractal_field`).
- `params.py` — stat→param mapping (`slot_stat_medians`,
  `load_fingerprint_stats`, `stats_to_params`, `slot_fractal_params`,
  `slot_sdf_params`).
- `sdf.py` — deterministic SDF families (`menger_sdf`, `mandelbulb_sdf`,
  `sdf_volume`).
- `surface_nets.py` — deterministic naive Surface Nets iso-extraction
  (`surface_nets`) with watertight outward triangles.
- `mosaic.py` — per-slot mini-SDF mosaic mesh builder (`build_sdf_mosaic`).
- `wrapper.py` — `FractalRenderer` (registry id `"fractal"`), fBm/SDF mode
  dispatch, per-model dedupe, OBJ export, Blender invocation.

## Local Contracts

- **Fractal geometry, not texture**: slot statistics feed the formula
  parameters (spec `fractal.mapping`: octaves ← effective_rank, persistence ←
  kurtosis, lacunarity ← sparsity, base_freq ← spectral_norm), never a height
  field with a fractal painted on top.
- **Determinism is a feature**: pure NumPy value noise on a fixed
  integer-lattice hash (splitmix64) and pure NumPy SDF + surface nets — no
  RNG, timestamps, or hash-order-dependent logic. Same inputs → byte-identical
  PNG + OBJ. Do not switch to `np.random` or external tools (Mandelbulber etc.)
  that break byte-identity.
- **Two modes, one renderer**: `fractal.mode` — `"fbm"` (default, height
  field) or `"sdf"` (per-slot mini-SDF mosaic). SDF mode: one Menger/Mandelbulb
  object per slot cell, extracted via `surface_nets` and merged by
  `build_sdf_mosaic`; per-object normalisation makes every mini-SDF fill its
  cell regardless of family/params. `sdf.grid` is the per-object lattice;
  iterations are clamped to `round(grid/6)` so coarse grids never alias to
  empty cells.
- **Cell budget**: `fractal.sdf.max_cells` (default 1024) caps the number of
  mini-SDF objects per mosaic. Rasters exceeding it (e.g. MoE expert panels
  with one column per expert) are decimated deterministically with
  aspect-preserving strides — sampled objects keep their true row/col
  positions and tints, only their count is bounded. Without the cap an
  896-expert panel would produce 80k+ objects (~115M verts) and crash Blender.
- **Primary raster only**: the fractal is one artefact per model built from
  the primary language raster. Expert (`expert_*`) and vision (`vision_*`)
  channels are skipped entirely (`render()` returns `[]`) — their auxiliary
  layouts must never define the fractal.
- **Per-slot character**: each slot column is its own fBm strip with its own
  parameters and fixed seed (base seed + slot index × 1009); tint is a second
  strip seeded base + 17. In SDF mode tint encodes the slot column; layout
  mirrors the (layers × slots) raster (`fractal.cell_h`/`cell_w` cells per
  logical cell).
- **Param mapping**: stats observed per slot are scaled linearly onto the spec
  target range and clamped; NaN stats fall back to the range midpoint. Mapping
  is fully data-driven via `spec.fractal.mapping` / `spec.fractal.sdf.mapping`
  — extend the spec, don't hard-code.
- **One render per model**: output depends on the fingerprint + seed, not the
  channel. Per-instance dedupe keyed on `(out_dir, seed, mode, layout)` where
  layout = (n_rows, n_cols, slot labels) — the API/CLI call `render()` once
  per channel (height, tint, rough) but Blender runs once; every channel
  reuses the identical artefacts. `mode` is part of the key so an fBm render
  and an SDF render of the same model never cross-pollinate the dedupe cache,
  and the layout part so channels with different rasters can't either.
- **Requires slot columns**: `field.col_labels` must be present (raises
  ValueError otherwise).
- **Blender reuse**: uses `build_blender_env`/`run_blender_command` and the
  `render_terrain.py` helpers from the `blender` renderer (traceback guard
  included); `render_sdf.py` imports the shared bpy helpers from
  `render_terrain.py` via a sys.path guard. Never asserts on Blender output in
  tests — they are dry-run (mocked `subprocess.run`).

## Work Guidance

- Add new fractal families (e.g. SDF/Menger/Mandelbulb) as additional modes in
  the same renderer, keeping the determinism contract.
- Keep bpy-side code free of repo imports; pass data via `.npy` + `--args`.

## Verification

- `tests/test_fractal_renderer.py` (36 tests): fbm determinism/range, slot
  field shape/determinism/distinct columns, param mapping ranges, NaN→midpoint
  fallback, per-slot seeds, SDF determinism/inside-outside, surface-nets
  watertight/outward/deterministic sphere, mosaic shape/footprint/tint,
  mosaic decimation (large rasters bounded + deterministic), expert/vision
  channel skipping, layout-keyed dedupe, dry-run subprocess + artefacts +
  traceback guard, byte-identical OBJ, per-channel dedupe, SDF-mode dry-run +
  determinism. Run via
  `cd /media/data/coding/weight-atlas && .venv/bin/python -m pytest tests/test_fractal_renderer.py`.