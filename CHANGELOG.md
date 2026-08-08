# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-08-08

### Three Core Guarantees

1. **Artifacts are renderer-independent and canonical**: All outputs (TIFF, PNG, JSON, OBJ) follow a versioned specification (`atlas_spec.v1.json`). Renderers never access raw weights — they consume only artifacts.
2. **Render/Compare never read weights**: The entire visualization and comparison pipeline operates on artifacts (statistics, fields, projections). Raw model weights are never loaded by renderers or comparators.
3. **Determinism is part of the measurement protocol**: All random number generators are seeded from the spec. Byte-identical outputs are guaranteed on the same machine (verified by SHA-256 manifest and second-run tests).

### Milestone Summary

- **M0 — Scaffolding**: Project structure, CLI entry point, plugin registry, core types, name mapping, spec file. CI with GitHub Actions.
- **M1 — Vertical Slice**: Safetensors loader, statistics (Frobenius, spectral norm, effective rank, kurtosis, sparsity), rasterizer, scaling, smoothing, matplotlib sheet renderer.
- **M1.5 — Fixup Batch**: Contours on raw height field, fixed PNG creation time, spec_version/tool_version/loader in fingerprint, render discovery from manifest.
- **M2 — Blender Renderer**: 1024² grid terrain rendering, OBJ mesh export, ortho top-view camera, NW lighting, worldbench engine.
- **M3 — Web UI**: FastAPI app, SQLite-backed job queue, Jinja2 templates, HTMX polling, model list/detail/compare pages.
- **M3.5 — Fixup**: Raw + smooth terrain in manifest, security note in README, HTMX vendoring backlog entry.
- **M4 — Comparison/Delta Layer**: Strict and aligned modes, delta fields on scaled values, summary metrics on raw stats, hotspot ranking, hard-reject on spec_version mismatch, warning on tool_version mismatch.
- **M5 — GGUF Loader**: GGUFReader with mmap, lazy tensor handles, F32/F16/BF16/Q8_0/Q4_0 dequantization, auto-detect via magic bytes, cross-loader comparison with warnings.
- **M6 — MoE Expert Panel**: ExpertPanel field class (Layer × Expert), main raster unchanged, shared expert to mlp slots, GGUF 3D stacked tensor splitting, panel comparison with skip-on-mismatch.
- **M7 — Embedding Sheet**: PCA projection with sign convention (max |loading| positive), density field via 2D histogram, UMAP support (optional extra), scatter overlay on sheets.
- **M8 — Activity Mode ("fMRI")**: Forward-pass activation capture via PyTorch hooks, versioned stimulus protocol (8 states), Layer×Position residual RMS fields, Layer×Expert usage fields for MoE, scanner discipline (threads=1, deterministic algorithms).

### Key Conventions

- **Spec version stays 1**: All extensions (blender, compare, embedding, activity) are additive. No breaking changes to scan output.
- **Slots**: 13 fixed slots (embed, attn_q/k/v/o, mlp_gate/up/down, norm_attn/mlp, router, lm_head, other).
- **Channels**: height (spectral_norm + log1p), tint (effective_rank + quantile_clip), rough (kurtosis + log1p).
- **Determinism**: Seeds from spec, threads=1 for activity, SHA-256 manifest per artifact.
