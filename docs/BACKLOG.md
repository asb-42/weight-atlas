# Backlog

These are explicitly **out of scope for M0+M1+M2+M3+M4+M5** and deferred per spec rules.

- **Full k-quant support** (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ variants) — M5 covers F32/F16/BF16/Q8_0/Q4_0; block-quantized k-quants require additional dequant logic
- Δ-maps / erosion tint for abliteration studies
- UMAP embedding sheet
- MoE expert panel rendering (per-slot sheet, row=layer, col=expert-id)
- Activity / "fMRI" mode via NNsight activation capture
- Morph animation A→B
- **HTMX vendoring** (offline-capable UI) — optional, CDN remains default
- Entry-point plugin registration (replaces conftest hack for tests)
- Embedding-Alignment cross-model (Procrustes, shared-vocab) — deferred from M7
- NNsight component-level capture (beyond plain PyTorch hooks) — deferred from M8
- GGUF activity bridge (llama.cpp for GGUF models) — deferred from M8
