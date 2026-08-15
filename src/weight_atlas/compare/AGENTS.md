# AGENTS.md — compare (M4)

## Purpose

Quantitative + cartographic comparison of two scanned models: align their
fields, compute per-channel deltas, rank hotspots, and emit a compare report.

## Ownership

- `align.py` (strict/aligned alignment, compatibility checks, resampling),
  `delta.py` (ComputeCompareSummary, ChannelDelta, hotspot ranking),
  `panel.py` (MoE expert-panel comparison), `render/delta_sheet.py`
  (diverging-colormap delta sheet renderer, registered as `"delta"`).

## Local Contracts

- **Two modes**: `strict` (same architecture, identical indices) and `aligned`
  (cross-architecture, normalized depth t∈[0,1] resampled to common grid).
- **Aligned row resampling** (`interp`): `"linear"` (bilinear zoom) or
  `"nearest"` (nearest-layer matching — copies real layer rows, preserves NaN
  holes). Source of truth: `compute_compare_summary(interp=...)`, spec key
  `compare.aligned_interp`. Aligned rows are normalized depth, NOT absolute
  layer indices — surface this in `CompareSummary.alignment` + warnings.
- **Hard-reject on spec_version mismatch**: `check_compatibility` raises
  ValueError; tool_version mismatch warns only.
- **Column labels track real field width**: `col_labels` are derived from
  `spec.slots` truncated/padded to the actual column count of the scanned
  field (fields scanned with an older spec may be narrower than the current
  `spec.slots`). Never feed full `spec.slots` to a narrower delta — downstream
  `zip(..., strict=True)` (e.g. `delta_sheet.py`) would crash. Emit a warning
  on any width mismatch. Aligned-mode common grid uses the max of the two real
  widths, NOT `len(spec.slots)` as a floor (no phantom upsampled columns).
- **Determinism**: delta TIFFs, compare JSON, and delta sheets must be
  byte-identical for identical inputs (NaN positions excluded via masks).
- **NaN discipline**: deltas are NaN where either field is NaN; hotspot
  ranking and metrics filter NaN explicitly (never treat as zero value).
- **CompareSummary.alignment**: always expose mode, layer counts, common grid,
  interp method, and per-row layer maps when aligned — the report renders
  these so readers never mistake aligned rows for absolute layers.

## Work Guidance

- Keep delta computation on scaled channel values (B - A after channel scale);
  summary metrics on raw stats (rel. L2, cosine sim) before scaling.
- New alignment strategies go through `align()` + spec key, never hard-coded.

## Verification

- `tests/test_compare.py` covers alignment, delta, summary, determinism;
  `tests/test_moe.py` covers expert panels. Run via
  `cd /media/data/coding/weight-atlas && .venv/bin/python -m pytest tests/test_compare.py`.