# Roadmap

Prioritized backlog derived from completed milestones (M0–M8).

## Priority 1: Embedding Alignment (Procrustes, Shared-Vocab)

Cross-model embedding comparison via Procrustes alignment or shared-vocabulary intersection. Currently embeddings are loader-independent but not directly comparable across models with different vocabularies.

**Deferred from M7**

## Priority 2: Full k-Quant Support (Q2_K–Q8_K, IQ variants)

M5 covers F32, F16, BF16, Q8_0, Q4_0. Block-quantized k-quant formats require additional dequantization logic (block-wise scales, quants, and dequant tables).

**Deferred from M5**

## Priority 3: NNsight Component-Level Capture

M8 uses plain PyTorch hooks for layer-level activation capture. NNsight would enable component-level (individual attention heads, MLP sublayers) capture at the cost of an additional dependency.

**Deferred from M8**

## Priority 4: GGUF Activity Bridge (llama.cpp)

M8 activity capture works with HuggingFace models (safetensors checkpoints). GGUF models require a llama.cpp bridge for forward-pass access.

**Deferred from M8**

## Priority 5: Morph Animation A→B

Animated interpolation between two model fingerprints (scan A → scan B). Requires temporal interpolation of field values and frame sequence rendering.

**Deferred from M3 vision**

## Priority 6: Cycles/Shader Contours

M2 uses Blender Workbench for terrain rendering. Cycles engine or custom shader contours would improve visual quality at the cost of GPU dependency and non-determinism.

**Deferred from M2**

## Priority 7: Entry-Point Plugin Registration

Replace the conftest-based registration hack with proper entry-point plugin registration. Would enable a proper plugin ecosystem.

**Known limitation documented in ARCHITECTURE.md**

## Priority 8: HTMX Vendoring

M3 uses HTMX via CDN. Vendoring would enable offline-capable UI at the cost of manual updates.

**Optional, CDN remains default**

## Completed Milestones

- [x] M0 — Scaffolding
- [x] M1 — Vertical Slice (safetensors, stats, sheet)
- [x] M1.5 — Fixup Batch (contours, PNG, manifest)
- [x] M2 — Blender Renderer
- [x] M3 — Web UI
- [x] M3.5 — Fixup (terrain raw+smooth, security)
- [x] M4 — Comparison/Delta Layer
- [x] M5 — GGUF Loader
- [x] M6 — MoE Expert Panel
- [x] M7 — Embedding Sheet
- [x] M8 — Activity Mode ("fMRI")
