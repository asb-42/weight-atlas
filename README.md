# Weight Atlas

LLM weight fingerprinting and topographic visualization. Scan a model → extract tensor statistics → rasterize into 2D fields → render topographic sheets. Fully deterministic and renderer-independent.

## Three Core Guarantees

1. **Artifacts are renderer-independent and canonical**: All outputs follow a versioned specification. Renderers never access raw weights.
2. **Render/Compare never read weights**: The entire visualization and comparison pipeline operates on artifacts (statistics, fields, projections).
3. **Determinism is part of the measurement protocol**: All RNGs are seeded from the spec. Byte-identical outputs guaranteed on the same machine.

## Installation

### Recommended: Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Alternative: User Installation

```bash
pip install --user -e .
```

### Alternative: pipx (Isolated Tool)

```bash
pipx install -e .
```

### Troubleshooting Installation

**`externally-managed-environment` error**: Modern Linux systems (PEP 668) prevent system-wide pip installs. Use one of:
- Virtual environment (recommended)
- `pip install --user -e .` (user-level)
- `pipx install -e .` (isolated)

## Quickstart

```bash
pip install -e .
weight-atlas --help
weight-atlas scan ./models/my_model --out ./artefacts
weight-atlas render ./artefacts --renderer sheet
```

> **v0.2.1 requires re-scan.** Existing v2.0 fingerprints cannot be compared with v2.1. See [CHANGELOG.md](CHANGELOG.md) for migration details.

## Features

- **Scan**: Extract tensor statistics (Frobenius, spectral norm, effective rank, stable rank, kurtosis, sparsity) from safetensors or GGUF models
- **Render**: Generate topographic sheets (hillshade + hypsometric tint + contours) or Blender 3D terrain
- **Compare**: Quantitative and cartographic comparison with strict or aligned modes
- **Embedding Sheet**: PCA/UMAP projection of token embeddings as density fields
- **MoE Expert Panels**: Layer × Expert visualization for Mixture-of-Experts models
- **Vision Tower Sheet (VLM)**: multimodal models get a separate vision panel
  (vision blocks × `v_attn_*`/`mm_projector` slots) with conv-aware statistics
  (kernel norm), so a vision tower is a visible fingerprint difference instead
  of an unmapped blind spot
- **Activity Mode ("fMRI")**: Forward-pass activation capture over a versioned stimulus protocol
- **Web UI**: Browse models, submit jobs, view artifacts via browser (HTMX + FastAPI)

## Usage

### Compare Models (M4)

```bash
weight-atlas compare ./artefacts/model_a ./artefacts/model_b --out ./compare_out --mode strict
```

Produces `compare_summary.json` (rel. L2, cosine similarity, hotspot ranking) and delta visualizations.

### GGUF Loader (M5)

```bash
pip install -e ".[gguf]"
weight-atlas scan ./models/my_model.gguf --out ./artefacts
```

### Embedding Sheet (M7)

```bash
weight-atlas scan ./models/my_model --out ./artefacts
weight-atlas render ./artefacts --renderer sheet  # Includes embedding density
```

For UMAP projection: `pip install -e ".[umap]"` and set `"embedding": {"method": "umap"}` in spec.

### Activity Mode (M8)

```bash
pip install -e ".[activity]"
weight-atlas activity ./models/my_model --out ./activity_out --protocol v1
```

Create a functional ablation map (pre/post ablation on refusal stimuli):
```bash
weight-atlas activity ./models/original --out ./pre --protocol v1
weight-atlas activity ./models/abliterated --out ./post --protocol v1
weight-atlas compare ./pre ./post --out ./lesion_map --mode strict
```

### Web UI (M3)

```bash
pip install -e ".[web]"
weight-atlas serve               # LAN-reachable on http://0.0.0.0:8000
# localhost-only alternative:
uvicorn weight_atlas.api.main:app --reload
# Open http://localhost:8000
```

`weight-atlas serve` binds all interfaces (`0.0.0.0`), so the UI is reachable
from other machines on the LAN at `http://<machine-ip>:8000`. For localhost-only
access use `weight-atlas serve --host 127.0.0.1` (or run uvicorn directly).

> **Security**: the web UI has no authentication. Serving on `0.0.0.0` exposes
> the job/import/artefact API (which reads scan directories and serves
> artefacts) to anyone on the network. Run only on a trusted network or behind
> a firewall/VPN.

## What weight-atlas is / is not

**Is:**
- A fingerprinting and comparison tool for LLM weight artifacts
- A topographic visualization engine (2D sheets + 3D terrain)
- A deterministic measurement system (spec-seeded RNG, byte-identical outputs)

**Is not:**
- A benchmark tool (no capability claims, no performance metrics)
- A model editor (read-only; never modifies weights)
- A capability assessment (height = projection convention, not quality score)

## Determinism

Every RNG is seeded from `specs/atlas_spec.v2.4.json`. Artifacts contain no timestamps. A second `scan` over the same input yields byte-identical `manifest.json` (SHA-256 per artefact).

## Layout

```
src/weight_atlas/
   core/      registry, types, name_map
   loaders/   safetensors, gguf (mmap, sharded)
   stats/     frobenius, spectral, kurtosis, sparsity
   fields/    rasterizer, scaling, smoothing, tif_io
   render/    matplotlib_sheet (+ Blender in M2)
   embedding/ pca, umap projection
   compare/   align, delta, panel comparison
   activity/  forward-pass activation capture (fMRI mode)
   api/       FastAPI app, job queue, routes (M3)
   ui/        Jinja2 templates + static CSS (M3)
   cli.py
specs/atlas_spec.v2.4.json
tests/       seeded fixtures, hand-computed + determinism tests
docs/        ARCHITECTURE.md, BACKLOG.md, ROADMAP.md, CHANGELOG.md
```

## Tests

```bash
pytest -q
```

## License

GNU Affero General Public License v3.0 (AGPL-3.0)
