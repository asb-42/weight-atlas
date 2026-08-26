# AGENTS.md — compare (M4)

## Purpose

Quantitative + cartographic comparison of two scanned models: align their
fields, compute per-channel deltas, rank hotspots, and emit a compare report.

## Ownership

- `pipeline.py` (`run_compare` — the single comparison orchestration used by
  BOTH `cli.py compare` and `JobQueue._run_compare`; also exports
  `discover_channels_from_manifest`),
  `align.py` (strict/aligned alignment, compatibility checks, resampling),
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
- **Different slot counts in aligned mode**: when A and B have different
  column widths (different architectures/specs), `"nearest"` pads the
  narrower field's right side to the common width (max of the two real
  widths) with NaN columns — missing slots are absent, never fabricated —
  so the delta never broadcasts mismatched shapes. `"linear"` resamples both
  to the common grid directly. Both paths warn on the width mismatch.
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
- **Sheet vs profile scaling**: the sheet's diverging colormap scales to the
  cell-level robust vmax (`_compute_vmax`); the per-row profile strip must
  compute its OWN vmax from its row RMS values — never reuse the cell vmax.
  When the bulk of cells are ~identical, the robust cap collapses the cell
  vmax toward ~0 and reusing it saturates every profile bar to the top of the
  "hot" colormap (a fully white `delta_profile_<channel>.png`).
- **NaN discipline**: deltas are NaN where either field is NaN; hotspot
  ranking and metrics filter NaN explicitly (never treat as zero value).
  Summary metrics (rel_l2, cosine_sim) use the SAME position-aligned element
  set — the intersection of both fields' finite masks; masking independently
  shifts row-major order and the cosine compares unrelated elements.
- **Single orchestration**: `run_compare()` owns channel discovery, the
  `spec.channels` filter (vision/expert channels are not comparable here),
  delta/summary artefact writing, and delta-sheet rendering. CLI and API pass
  their extras (row labels, noise floor, progress) as parameters — do not fork
  per-entrypoint copies.
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
- `tests/test_compare.py` has 62 tests (aligned nearest/linear, phantom-column
  guard, different-slot-count padding, delta shapes, hotspot ranking,
  profile-strip self-scaling regression).