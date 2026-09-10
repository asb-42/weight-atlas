# Records-Tab Outlier Visualization (OCGQuant framing)

**Date:** 2026-09-09 · **Status:** implemented (records tab) · **Reference:** arXiv:2609.00066 (OCGQuant, EMNLP 2026)

## Problem

The records tab rendered boards as bare numbers:
`Worst outlier channels — 1.814 — model.embed_tokens.weight`. A number
without context does not answer the reader's actual questions:

- Is 1.814 extreme *within this model*, or ordinary for it?
- Where in depth do the outlier channels live — embedding row spikes?
  mid-layer MLP? concentrated in one layer?
- Why does it matter for quantization at all?

## Framing (from the paper)

OCGQuant studies NVFP4 activation quantization and names the mechanism our
`row_amax_ratio`/`col_amax_ratio` metrics already measure: a channel whose
magnitude dominates its row/column **sets the per-channel quantization
scale**, and every companion channel sharing that scale inherits the
outlier's magnitude as *collateral quantization error*. Mitigations
(AWQ/SmoothQuant-style scaling, mixed precision, clamping) all target this
dominance. The paper's contribution (outlier-companion grouping) is
activation-side; our contribution is making the weight-side signal visible
before anyone trains a mitigation.

## Implemented visualizations (server-side SVG, deterministic, no JS)

1. **Outlier impact ranking** (`_impact_svg`): top-10 tensors by
   `max(row_amax_ratio, col_amax_ratio)`, log-scaled bars, slot-group
   colors, hover carries the row/col split. Answers: *which tensors will
   hurt group quantization most*.
2. **Depth profile** (`_profile_svg`): max `row_amax_ratio` per layer with
   the peak layer circled. Answers: *where in depth outlier channels
   concentrate* (embedding-adjacent first/last layers are the classic
   pattern).
3. **Board-leader percentile strips** (`_strip_svg`): for each amax board,
   the leader's value as a marker over the model-wide distribution
   (min–max whiskers, p25–p75 box, median tick, p99 label, log-scaled when
   the span warrants). Answers: *is the board leader an outlier among
   outliers*.

All rendering lives in `api/routes.py` (`_impact_svg`, `_profile_svg`,
`_strip_svg`); the data selection is pure and deterministic in
`api/query.py` (`outlier_impact`, `layer_profile`, `distribution_strip`).
The amax metrics exist per-tensor in every fingerprint since the
distribution-shape ladder landed, so old scans render the charts without
rescanning (metrics absent → boards/cards simply skip, as before).

## What the example means, read through these charts

For `turboderp_MiniCPM5-1B-exl3`: the embedding row ratio 1.814 sits at the
top of a distribution whose typical tensors are far tamer — the impact
ranking shows it against MLP/attn competitors, the depth profile shows
whether mid-layer spikes beat it, and the strip shows it at the model's
extreme percentile. That is the "auf einen Blick" readout the raw board
could not deliver.

## Deliberately not done

- No client-side JS charting (keeps the no-JS contract of the scatter tab).
- No per-channel raw heatmaps: the UI reads only `fingerprint.json`
  (renderer/compare never load weights), and per-channel data is not in
  the fingerprint by design.
- No activation-side collateral-error simulation: OCGQuant's mechanism
  needs activation magnitudes, which a weight scan does not measure.
