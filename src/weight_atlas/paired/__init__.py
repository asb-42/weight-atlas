"""Paired tensor-difference analysis (M9) package.

`run_paired` compares two weight snapshots tensor-by-tensor and reports either
quantization impact (preset ``quant``: SQNR, rel-L2, cosine, zero-flip,
max-delta, optional operator norm) or edit signatures / abliteration (preset
``edit``: rel-L2, cosine, operator norm, Δ-spectrum rank metrics, opt-in
u1-coherence, classification, edit bands, weight-space hotspot ranking) as
rasterised fields + summary JSON.
"""

from weight_atlas.paired.paired import run_impact, run_paired

__all__ = ["run_paired", "run_impact"]
