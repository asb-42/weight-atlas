# AGENTS.md — render/fractal

## Purpose

Statistics-driven fractal terrain renderer (registered `"fractal"`): genuine
fBm geometry whose per-slot parameters derive from real tensor statistics,
rendered through the shared Blender terrain pipeline.

## Ownership

- `fbm.py` — deterministic value-noise fBm (`_hash_lattice` splitmix64,
  `value_noise`, `fbm`, `terrain_field`, `slot_fractal_field`).
- `params.py` — stat→param mapping (`slot_stat_medians`,
  `load_fingerprint_stats`, `stats_to_params`, `slot_fractal_params`).
- `wrapper.py` — `FractalRenderer` (registry id `"fractal"`), per-model
  dedupe, OBJ export, Blender invocation.

## Local Contracts

- **Fractal geometry, not texture**: slot statistics feed the formula
  parameters (spec `fractal.mapping`: octaves ← effective_rank, persistence ←
  kurtosis, lacunarity ← sparsity, base_freq ← spectral_norm), never a height
  field with a fractal painted on top.
- **Determinism is a feature**: pure NumPy value noise on a fixed
  integer-lattice hash (splitmix64) — no RNG, timestamps, or hash-order-
  dependent logic. Same inputs → byte-identical PNG + OBJ. Do not switch to
  `np.random` or external tools (Mandelbulber etc.) that break byte-identity.
- **Per-slot character**: each slot column is its own fBm strip with its own
  parameters and fixed seed (base seed + slot index × 1009); tint is a second
  strip seeded base + 17. Layout mirrors the (layers × slots) raster
  (`fractal.cell_h`/`cell_w` cells per logical cell).
- **Param mapping**: stats observed per slot are scaled linearly onto the spec
  target range and clamped; NaN stats fall back to the range midpoint. Mapping
  is fully data-driven via `spec.fractal.mapping` — extend the spec, don't
  hard-code.
- **One render per model**: output depends on the fingerprint + seed, not the
  channel. Per-instance dedupe keyed on `(out_dir, seed)` — the API/CLI call
  `render()` once per channel (height, tint, rough, vision_*) but Blender runs
  once; every channel reuses the identical artefacts (primary language
  raster's layout, never overwritten by the smaller vision layout).
- **Requires slot columns**: `field.col_labels` must be present (raises
  ValueError otherwise).
- **Blender reuse**: uses `build_blender_env`/`run_blender_command` from the
  `blender` renderer (traceback guard included); never asserts on Blender
  output in tests — they are dry-run (mocked `subprocess.run`).

## Work Guidance

- Add new fractal families (e.g. SDF/Menger/Mandelbulb) as additional modes in
  the same renderer, keeping the determinism contract.
- Keep bpy-side code free of repo imports; pass data via `.npy` + `--args`.

## Verification

- `tests/test_fractal_renderer.py` (19 tests): fbm determinism/range, slot
  field shape/determinism/distinct columns, param mapping ranges, NaN→midpoint
  fallback, per-slot seeds, dry-run subprocess + artefacts + traceback guard,
  byte-identical OBJ, per-channel dedupe. Run via
  `cd /media/data/coding/weight-atlas && .venv/bin/python -m pytest tests/test_fractal_renderer.py`.