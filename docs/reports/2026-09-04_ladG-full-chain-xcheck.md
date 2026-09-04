# G2 Full-Chain Cross-Validation (Task #83 extension, Quinn seq 94)

Date: 2026-09-04 · By: OC-GLM-200 · For: A0-Quinn (seq 94 coverage finding)
Scans: 15 ladG phase-finals (en..sk, mult 128..576) + the 5 ladG2 scans
from the seq-64 series (sv..lt, mult 608..736) = the complete 20-phase
G2 chain. (Quinn was right: ladG2-* covers only phases 16-20.)

## Quinn's cross-validation, answered

"Old segments should show uniform multiplicative amplitude decay with
scale-invariant stable_rank" — measured through weight-atlas per-unit
stats (independent code path from the torch c-fits: pure-python
unpickler + fp64 Gram-free rSVD spectra, no torch anywhere).

En-era units (decoder u<128, n=1024 records), spectral_norm mean:

en 0.779328 → es 0.695656 → pl 0.620967 → fr 0.554298 → de 0.494786 →
cs 0.441664 → da 0.394245 → pt 0.351917 → fi 0.314134 → hu 0.280407 →
bg 0.250301 → it 0.223428 → et 0.199440 → el 0.178027 → sk 0.158913 →
sv 0.141852 → ro 0.126622 → nl 0.113027 → sl 0.100892 → lt 0.090060

Step ratios, all 19: **0.8926 every single one** (G-family closed form
0.8927, agreement to 4 decimals). stable_rank mean: **3.3000 at all 20
phases, spread 0.000000** — the spectral profile is bit-identical while
amplitude decays 8.65× end to end.

No deviation anywhere, including across the ladG→ladG2 family boundary
(sk→sv: 0.8926, same as all others) — the two checkpoint families are
one continuous chain under this instrument, which independently
corroborates the repair-staging symlink story (ladGX prefixes).

## Artefacts

- Scans: output/bdh-ladG-{en,es,pl,fr,de,cs,da,pt,fi,hu,bg,it,et,el,sk}/
  + output/bdh-ladG2-{sv,ro,nl,sl,lt}/ on ai, all provenance-anchored.
  Record counts scale linearly with width (3102 + 768/step, exact).
