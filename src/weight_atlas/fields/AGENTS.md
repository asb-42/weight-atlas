# AGENTS.md — fields

## Purpose

Rasterisation and transformation of per-layer statistics into topographic
fields: scaling, smoothing, upsample, degenerations, and TIFF persistence.

## Ownership

- `rasterizer.py` (layer→field rasterisation, slot columns), `scaling.py`
  (channel scale functions: log1p, quantile_clip, robust_scale, rank_scale),
  `smoothing.py` (gaussian smoothing, upsample), `degenerations.py`
  (degenerate-field handling), `tif_io.py` (TIFF read/write).

## Local Contracts

- **Field shape**: 2D (layers × slots). Columns follow `spec.slots` order;
  missing slots carry NaN.
- **Determinism**: rasterisation/scaling/smoothing are pure NumPy — no RNG,
  no timestamps. TIFF writes are byte-deterministic (no compression defaults
  that embed timestamps).
- **NaN discipline**: NaN means "slot/layer absent". Scaling and smoothing
  must not leak NaN into neighbours incorrectly (smoothing may dilate NaN
  edges — that is expected and handled by callers).
- **Scale functions** are pure (value → value) and spec-driven
  (`channel.scale.type`); add new scales via `scaling.py` + spec, not inline.

## Work Guidance

- Prefer per-column scaling when a channel's stats span different magnitudes
  (e.g. rank_scale per_column) — but keep it spec-explicit, never implicit.

## Verification

- `tests/test_fields.py`, `tests/test_degenerations.py`,
  `tests/test_sheet_degenerate.py`. Run via
  `cd /media/data/coding/weight-atlas && .venv/bin/python -m pytest tests/test_fields.py tests/test_degenerations.py`.