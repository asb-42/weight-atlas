# AGENTS.md — weight_atlas package

## Purpose

The `weight_atlas` package implements the full M0–M4 pipeline: load weights
from safetensors/GGUF, compute per-layer statistics, rasterize them into
topographic fields, render sheets + Blender terrain, and quantitatively
compare two scanned models. The web UI (api + ui) is the primary interface.

## Ownership

- Entry points: `cli.py` (argument parsing + orchestration), `scan.py`
  (M1/M2 scan pipeline). Root-owned build/config files stay out of this doc.
- Owned by parent (root AGENTS.md): nothing inside this package — this doc
  owns everything below `src/weight_atlas/` except the child directories
  listed in the Child DOX Index.

## Local Contracts

- **Plugin registry**: loaders, stats, and renderers register by string ID via
  `core.registry.register_*`. Duplicate IDs raise ValueError. Side-effect
  imports at module bottom register plugins; keep those imports.
- **Determinism is a feature**: all outputs (TIFFs, PNGs, OBJ, compare JSON)
  must be byte-identical for identical inputs. No RNG, timestamps, or
  hash-order-dependent logic in production paths.
- **NaN handling**: fields carry NaN for missing slots/panels. Downstream code
  must treat NaN as "absent", never as a value; compare/render filter or mask
  them explicitly.
- **Spec coupling**: behaviour follows `AtlasSpec` from `core.types`
  (`spec.channels`, `spec.compare`, `spec.blender`). Extend the spec, don't
  hard-code values. Keep all `specs/*.json` versions in sync for shared keys.
- **Name mapping is spec-driven**: `core/name_map.py` compiles the `name_map`
  block from the canonical default spec. Adding a tensor family = edit the
  v2.4 spec block + add the slot; the in-code rule lists are a fallback for
  older specs and must stay in sync.
- **Architecture docs**: durable design rules live in `docs/ARCHITECTURE.md`;
  update it when pipeline contracts change.

## Work Guidance

- Follow the plugin-registry pattern for new loaders/stats/renderers.
- Read `specs/AGENTS.md` before touching spec keys; `tests/AGENTS.md` before
  adding fixtures; `docs/AGENTS.md` before writing design docs.
- Matplotlib/Pillow/Blender output must stay off the critical determinism path
  or be explicitly verified byte-identical in tests.

## Verification

- `cd /media/data/coding/weight-atlas && .venv/bin/python -m pytest tests/`
  (run from the repo root; venv python is required).
- Subtree rules: see `tests/AGENTS.md`.

## Child DOX Index

- `src/weight_atlas/api/AGENTS.md` — FastAPI web server: routes, job queue,
  file browsing, compare/render endpoints.
- `src/weight_atlas/compare/AGENTS.md` — M4 comparison pipeline: alignment
  (strict/aligned), delta computation, hotspot ranking, compare report.
- `src/weight_atlas/fields/AGENTS.md` — field rasterisation, scaling,
  smoothing, degenerations, TIFF I/O.
- `src/weight_atlas/loaders/AGENTS.md` — safetensors/GGUF loaders and
  dequantisation (mxfp4, gguf_dequant).
- `src/weight_atlas/render/AGENTS.md` — renderers: matplotlib sheet, preview,
  Blender terrain (+ OBJ export).

Owned by this doc (no child AGENTS.md): `core/` (types, registry, name_map),
`stats/` (statistics), `embedding/` (PCA/UMAP), `activity/` (forward-pass
capture), `ui/` (templates + static, consumed by api/), `cli.py`, `scan.py`.
