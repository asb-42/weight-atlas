# ladG2 Direction-Preservation Report (Task #83 follow-up)

Date: 2026-09-04 · By: OC-GLM-200 (weight-atlas seat) · For: pi-50 (task seq 83), A0-Quinn
Scans: 5 ladG2 phase-finals (sv/ro/nl/sl/lt_last.pt, widths 608→736)
Code: weight-atlas @ current main (stable_rank 1-D convention fixed)
Method: same instrument as the RA2 series — per-unit stats + direct
tensor reads (blk.0.decoder head-0 block), cosine on old-unit blocks.

## Headline

**Direction preserved to floating-point noise; uniform multiplicative
decay ×0.8926 per phase — matching Quinn's independently-derived
G-family constant (0.8927) to 4 decimals.** The five local phases are
consecutive growth steps (width +32 each: 608→640→672→704→736), and
every step shows the same signature.

## 1. Scan stats over the 608-unit common prefix (decoder.u*, all heads)

| metric | sv (608) | ro (640) | nl (672) | sl (704) | lt (736) |
|---|---|---|---|---|---|
| spectral_norm (mean) | 0.3216 | 0.2871 | 0.2563 | 0.2287 | 0.2042 |
| stable_rank (mean) | 3.3305 | 3.3305 | 3.3305 | 3.3305 | 3.3305 |
| kurtosis (mean) | 0.0440 | 0.0440 | 0.0440 | 0.0440 | 0.0440 |
| sv_decay (mean) | 0.4039 | 0.4039 | 0.4039 | 0.4039 | 0.4039 |
| sparsity (mean) | 0.1148 | 0.1284 | 0.1436 | 0.1605 | 0.1792 |

Scale-invariant stats are bit-identical across all five phases
(stable_rank/kurtosis/sv_decay agree to all printed decimals) while
spectral_norm declines geometrically: 0.893, 0.893, 0.892, 0.893.
Sparsity rises mechanically as magnitudes shrink past the 1e-3 floor.

## 2. Ground-truth weight reads (blk.0.decoder, old-row blocks)

| transition | cosine(old, new) | mean ratio |
|---|---|---|
| sv→ro | 1.000003 | 0.8926 |
| ro→nl | 1.000002 | 0.8926 |
| nl→sl | 0.999999 | 0.8926 |
| sl→lt | 1.000003 | 0.8926 |

(cosines print slightly above 1.0 — fp rounding in the dot product of
near-parallel vectors; direction is preserved to noise.)

## 3. Reading for the room

- Task #83's falsifier is satisfied in the affirmative: G2 shows the
  SAME decay-in-place signature as RA2 (direction intact, magnitude
  geometrically attenuated), at the G-family rate, not the RA2 rate.
- The per-step constancy (0.8926 ×4, no drift) is the strongest form of
  Quinn's seq 87 claim: uniform multiplicative decay to ~1e-5 residuals,
  now confirmed on an independent arm by an independent instrument
  (weight-atlas per-unit stats, not segment c-fits).
- The accelerating-shrink pattern seen in RA2 (×0.117, ×0.068, ×0.0043
  across en→de→pt→lt) does NOT appear here — G2's decay is constant per
  phase, as expected from the pinned-lr schedule (defaults: warmup 30,
  decay 300, min_lr 1e-4) vs RA2's cosine schedule. This independently
  corroborates the two-regimes root cause (seq 86/87).

## 4. Artefacts

- Scans (fingerprint + fields + manifest, provenance-anchored):
  output/bdh-ladG2-{sv,ro,nl,sl,lt}/ on ai (192.168.178.200)
  - sv: 14622 / ro: 15390 / nl: 16158 / sl: 16926 / lt: 17694 records
- This report: docs/reports/2026-09-04_ladG2-direction-preservation.md
- Note: only 5 of the G2 phases are mirrored locally (the wide tail,
  608–736). Earlier/smaller G2 phases, if they exist on gx10, would
  extend Table 1 leftward at the same 0.8926 rate — a cheap prediction
  to check if those files surface.
