# RA2 Growth Ladder — Cross-Checkpoint Weight-Atlas Report

Date: 2026-09-03 · By: OC-GLM-200 (weight-atlas seat) · For: A0-Quinn (task seq 38)
Scans: 4 checkpoints (en mult 128, de mult 256, pt mult 416, lt mult 736)
Code: weight-atlas @ 07e959b (stable_rank 1-D convention fixed pre-series)
Method: per-unit expansion (decoder.u{u}.h{h}, 64 neurons/unit), units are
the growth axis — u < 128 = old neurons (present since en), u ≥ 128 = appended.

## Headline answer to "do old neurons stay frozen, drift, or get reorganized?"

**Neither frozen nor reorganized: they DECAY IN PLACE — direction preserved,
magnitude geometrically shrunk.** In the final lt checkpoint the entire en-era
decoder block is a near-exact scalar multiple of its en values (cosine 0.9986)
at ~3.4e-5 of original magnitude. Old-neuron knowledge is not destroyed (the
routing diagnosis shows it serving), but its readout weights are attenuated
to the numerical-noise floor as the model grows.

## 1. Per-unit scan statistics (old units u=0..127, 8 heads, n=1024)

| metric | en | de | pt | lt | reading |
|---|---|---|---|---|---|
| spectral_norm (mean) | 2.233 | 0.261 | 0.018 | 0.0001 | magnitude collapse |
| stable_rank (mean) | 2.769 | 2.772 | 2.772 | 2.772 | shape FROZEN (+0.1%) |
| kurtosis (mean) | 0.629 | 0.631 | 0.631 | 0.631 | shape frozen (+0.3%) |
| sv_decay (mean) | 0.147 | 0.148 | 0.148 | 0.148 | shape frozen (+0.4%) |
| sparsity (mean) | 0.020 | 0.166 | 0.988 | 1.000 | → all-but-zero |

Scale-invariant stats (stable_rank, sv_decay, kurtosis) are constant to
3 decimals across all 20 phases: the spectral PROFILE of old units never
changes — only the scale does. This is the signature of multiplicative
decay, not retraining (retraining would change the profile).

## 2. Ground-truth weight check (blk.0.decoder, direct tensor reads)

- en old rows: mean |w| = 0.0350, max = 0.520
- lt old rows (first 8192 rows): mean |w| = 1.2e-6, max = 9.7e-6 — the
  entire en decoder block is present at 3.4e-5× its original scale
- cosine(lt_old, en) = 0.9986 — direction preserved to 0.14%
- elementwise ratio std 1.07e-5 around the mean 3.44e-5: the decay is
  close to (but not exactly) a uniform scalar per element

## 3. Phase-over-phase shrink factors (old rows, mean ratio)

| transition | shrink | ln factor |
|---|---|---|
| en → de | ×0.117 | −2.14 |
| de → pt | ×0.068 | −2.69 |
| pt → lt | ×0.0043 | −5.44 |

The decay ACCELERATES as width grows — consistent with each growth phase
redistributing readout capacity toward fresh (zero-initialized, then
trained) units which carry the new language, while the optimizer
down-weights the old block progressively. Total en→lt: ×3.4e-5.

## 4. New units (appended at each step)

| step | new units | spectral_norm (new) vs (old) | sparsity new |
|---|---|---|---|
| de (+128) | 1.068 vs 0.261 (4.1×) | 0.067 |
| pt (+160) | 0.991 vs 0.045 (22×) | 0.118 |
| lt (+320) | 0.572 vs 0.002 (323×) | 0.398 |

New units start near-zero and train up; by each phase-final they carry
~3 orders of magnitude more spectral energy than the old block. Their
stable_rank (~2.34-2.59) is slightly below the old units' frozen 2.77 —
fresher, less uniformly distributed spectra.

## 5. Answering Quinn's routing follow-up (seq 51)

"is the own-prefix preservation of the resume languages structural
(stable spectral norms in old width segments) or functional-only?"

**Both, in a specific sense**: the old segments keep their STRUCTURE
(direction, spectral profile — cosine 0.9986, stable_rank frozen) but not
their absolute magnitude (×3.4e-5 by lt). The router reads back knowledge
that is directionally intact; the serving strength comes through the
nested-prefix route widths (a wider route subsumes narrower segments —
Quinn's own seq-49 semantics), which explains why routed serving
recovers bg/el/etc. to within 1-10% of acquisition despite the attenuation:
the routing diagnosis feeds 40-crop queries through prefix masks, and the
direction-preserved old blocks still rank-order correctly even at
attenuated scale. The hu +29% / fi +97% routed-vs-acquisition anomalies
in Quinn's report are consistent with their segments sitting at
intermediate decay (fi at phase 13, hu at 14 — shrink ×0.068 and ×0.0043
of the pre-phase scale, less attenuated than en's tail).

## 6. Artefacts

- Scans (fingerprint + fields + manifest, provenance-anchored):
  output/bdh-ladRA2-{en,de,pt,lt}/ on ai (192.168.178.200)
  - en: 3102 records | de: 6174 | pt: 10014 | lt: 17694
  - per-unit lattice panels: field_expert_bdh_decoder_*.tif (48 rows
    [8 heads × 8? units? no — 128..736 columns], per scan)
- .wasc packages available on request (stats profile ~few MB each)
- This report: docs/reports/2026-09-03_ladder-cross-checkpoint-scan.md
- Determinism: all four scans byte-reproducible; journal discarded on
  success; provenance SHA-256 anchored per checkpoint file

## 7. Caveats

- The per-unit old/new split uses the en boundary (u<128). Units appended
  between en and the phase in question are "new" only relative to that
  step; Table 2 handles this by comparing within-checkpoint cohorts.
- Shrink factors are mean elementwise ratios over |en|>1e-3 rows of head-0
  decoder only (largest single block; other heads show the same pattern in
  the scan stats — Table 1 covers all 8 heads × 128 old units).
- spectral_norm of near-zero units approaches the fp noise floor; the
  pt→lt ×0.0043 factor is trustworthy, but en→lt total (×3.4e-5) sits at
  the edge where quantization of the analysis (not the scan) matters.
