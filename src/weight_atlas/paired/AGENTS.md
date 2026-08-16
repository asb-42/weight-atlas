# AGENTS.md — paired

## Purpose

M9 paired tensor-difference pipeline comparing two weight snapshots
tensor-by-tensor via name-level pairing (`map_name`). Two presets share the
pairing, rasterisation and determinism machinery:

- `quant` (default): quantization impact — SQNR, rel-L2, cosine, zero-flip,
  max-delta, opt-in `dspec`. Summary `impact_summary.json`.
- `edit`: edit signatures / abliteration — rel-L2, cosine, Δ-spectrum metrics
  (`dspec`, `delta_stable_rank`, `spectral_share`), opt-in `u1_coherence`,
  classification heuristic, edit bands, weight-space hotspot ranking. Summary
  `compare_summary.json` (with `edit_signature` + `noise_floor` blocks).

## Ownership

- `paired.py` (engine: `run_paired`, `run_impact` alias, pairing,
  `_pair_metrics`, classification/bands, artefact writing),
  `render.py` (`ImpactSheet` registered `"impact"`, `EditSheet` registered
  `"edit"`), `__init__.py` (exports `run_paired`, `run_impact`).
- CLI: `paired` subcommand (`--preset {quant,edit}`, `qimpact` alias) lives in
  `cli.py`, owned by the parent (weight_atlas) doc.

## Local Contracts

- **Pairing is a name-level join**: tensors pair on `map_name(name)` →
  identical `(layer, slot)` plus matching tensor name. Same `(layer, slot)`
  with different names across formats pair via slot+layer and record both
  names. One-sided tensors are `"skipped"`, never a crash. Shape mismatch on a
  paired tensor → ValueError naming the tensor. `layer=None` encodes as −1 in
  the sortable join key; expert tensors key on
  `("expert", layer, moe_slot, expert_id)`.
- **Strict-only**: `mode != "strict"` raises ValueError. Use the compare
  subcommand for aligned/cross-architecture comparison.
- **Determinism is a feature**: per-tensor chunk loop runs sequentially, float64
  accumulators in fixed order; tensors may process in any order (jobs=N thread
  pool, threadpoolctl pins BLAS to 1 thread) but each accumulation is
  order-independent float64 → byte-identical for any `jobs`. Δ-spectrum reuses
  `stats/spectrum.py` (same lock, same seeded rSVD) so spectral values are
  deterministic too. PNG metadata fixed; sheet raster capped to a pixel budget.
- **Preset metric sets**: `dspec` is opt-in for quant (`operator_impact`); the
  edit preset always computes the Δ-spectrum because classification needs it.
- **Classification (edit)**: decision tree on edited tensors (rel-L2 >
  `band_floor`): none → `identical`; median `delta_stable_rank` ≤ `rank_low` →
  `low_rank_localized` (band share ≥ `band_mass_share`) / `low_rank_diffuse`;
  else no bands → `full_rank_uniform`, bands → `diffuse`.
- **Edit bands**: per-layer median over *edited* tensors only; unedited layers
  count 0 so slot-concentrated edits stand out. Contiguous layers above
  `max(band_floor, band_threshold_factor × all-layer median)` form bands;
  concentrated slots = per-slot within-band median > band median.
- **Spec coupling**: preset knobs come from `spec.qimpact` / `spec.edit`
  (canonical-only; absent → `{}` + documented defaults). Shared keys stay in
  sync across `specs/*.json` versions.
- **Noise floor**: loader + per-tensor `dtype` fingerprints drive the
  `identical`/`mismatched` policy; `mismatched` appends a warning that the
  edit signal may be at/below quantization noise.

## Work Guidance

- New presets: extend `run_paired`'s block selection and `_write_artefacts`;
  register the sheet renderer in `render.py`; add `--preset` handling to
  `cli.py`; keep `qimpact` behaviour on the quant preset (backwards compat).
- Spectral values must come from `stats/spectrum.py` (single lock, seeded rSVD)
  — never an ad-hoc SVD that could deadlock or break determinism.

## Verification

- `tests/test_paired.py` (quant impact, edit signatures, determinism, CLI).
  Full suite: `cd /media/data/coding/weight-atlas && .venv/bin/python -m pytest tests/`.

## Child DOX Index

- None. `render.py` sheets are owned here; `stats/spectrum.py` is owned by the
  parent (weight_atlas) doc.