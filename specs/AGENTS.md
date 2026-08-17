# AGENTS.md — specs

## Purpose

Atlas specification JSON files that drive the whole pipeline (channels,
scales, grid, render, compare, blender) plus the activity protocol schema.

## Ownership

- `atlas_spec.v1.json`, `atlas_spec.v2.json`, `atlas_spec.v2.1.json` …
  `atlas_spec.v2.4.json` (active/default is v2.4), `activity_protocol.v1.json`.
- `src/weight_atlas/core/types.py` (AtlasSpec parsing) is owned by the parent
  (weight_atlas) doc but is the consumer contract for these files.

## Local Contracts

- **Edit surgically**: files use compact single-line blocks (e.g. the
  `blender` line). Add/change one line at a time via text edits; never
  re-format whole files (`json.dump(indent=…)` is forbidden).
- **Keep versions in sync**: shared extension keys must exist in all spec
  versions — e.g. `compare.aligned_interp` (linear/nearest) and the
  `blender` block keys (`pitch`, `clip`, `adaptive_z_scale`,
  `subsurf_levels`, `fill_light_energy`) and the `fractal` block keys
  (`seed`, `cell_h`, `cell_w`, `mode` (`"fbm"`/`"sdf"`), `mapping`, `sdf`
  (`family`, `grid`, `max_cells`, `mapping`)). Adding a key to v2.4 but not v1
  is a contract violation.
- **Spec version is hard**: `spec_version` mismatch is a hard reject in
  compare. Never bump `spec_version` for additive extensions — document them
  in `docs/ARCHITECTURE.md` instead.
- **Unknown keys are tolerated**: AtlasSpec parses compare/blender as dicts
  without schema validation, so extensions are safe; keep them optional with
  documented defaults.
- **Consumers**: channels (stats + scale), `grid` (upsample/smooth), `sheet`
  (contours/lighting/dpi), `blender` (render settings), `fractal` (fBm/SDF
  renderer mapping: `seed`, `cell_h`/`cell_w`, `mode`, per-target `stat` +
  `lo`/`hi` range; `sdf` block adds `family`/`grid`/`max_cells`/`mapping`),
  `compare` (alignment/mode/interp), `embedding` (PCA/UMAP method),
  `name_map` (tensor-name → slot mapping registry, read by
  `core/name_map.py`).

## Name-mapping registry (`name_map` block)

- The **canonical default spec** (v2.4) carries a top-level `name_map` block:
  per-convention ordered regex rules (`hf`/`gguf`, grouped as
  moe/base/hybrid/kimi with first-match-wins order), `layer` index patterns,
  `non_layer_order`, and `vision` rules. `core/name_map.py` compiles it at
  runtime; new tensor families are added by editing this block, **not** code.
- **Sync policy**: the block is version-independent mapping knowledge, so it
  lives only in the canonical spec. Older spec versions (v1–v2.3) deliberately
  have no `name_map` key — `name_map.py` falls back to its in-code tables for
  them. Keep the fallback tables in sync with the v2.4 block when the mapping
  changes.

## `qimpact` block (canonical-only)

- The **canonical default spec** (v2.4) carries a `qimpact` block: metrics,
  `operator_impact`, `db_range`, `colormap`, `profile_strip`, `type_map`,
  `chunk_size` (M9 quantization impact). Older spec versions (v1–v2.3)
  deliberately have no `qimpact` key — `AtlasSpec` defaults it to `{}` and
  the pipeline uses documented defaults. Same canonical-only policy as
  `name_map`.

## `edit` block (canonical-only)

- The **canonical default spec** (v2.4) also carries an `edit` block (M9 edit
  signatures / abliteration, `--preset edit`): metrics (`rel_l2`, `cos`,
  `dspec`, `delta_stable_rank`, `spectral_share`), `u1_coherence` (opt-in),
  `rank_low`, `band_threshold_factor`, `band_floor`, `band_mass_share`,
  `rel_l2_log_range`, `rank_log_range`, `colormap`, `profile_strip`,
  `chunk_size`. Older spec versions have no `edit` key — `AtlasSpec` defaults
  it to `{}` and the pipeline uses documented defaults. Same canonical-only
  policy as `name_map`/`qimpact`.

## Work Guidance

- Read `src/weight_atlas/core/types.py` before adding a key to confirm how it
  is parsed and defaulted.
- After changing specs, run spec-parsing tests + a full suite run.

## Verification

- `tests/test_types.py` (parsing/defaults), full suite:
  `cd /media/data/coding/weight-atlas && .venv/bin/python -m pytest tests/`.