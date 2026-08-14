# Weight Atlas User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [CLI Reference](#cli-reference)
6. [Web UI Guide](#web-ui-guide)
7. [Plugins](#plugins)
8. [Artifact Reference](#artifact-reference)
9. [Troubleshooting](#troubleshooting)

---

## Introduction

Weight Atlas is a tool for **LLM weight fingerprinting and topographic visualization**. It scans a model's weights, extracts tensor statistics, rasterizes them into 2D fields, and renders topographic sheets or 3D terrain. It also supports quantitative comparison between models and forward-pass activation capture ("fMRI mode").

### What Weight Atlas Does

- **Fingerprint models**: Extract statistical signatures (Frobenius norm, spectral norm, effective rank, kurtosis, sparsity) from each tensor
- **Visualize weights**: Render topographic sheets (hillshade + hypsometric tint + contours) or 3D terrain (Blender)
- **Compare models**: Quantitatively and cartographically compare two models (strict or aligned modes)
- **Project embeddings**: PCA/UMAP projection of token embeddings as density fields
- **Analyze MoE models**: Layer × Expert visualization for Mixture-of-Experts architectures
- **Capture activations**: Forward-pass activation capture over a versioned stimulus protocol ("fMRI mode")

### Three Core Guarantees

1. **Artifacts are renderer-independent and canonical**: All outputs follow a versioned specification (`atlas_spec.v2.4.json`). Renderers never access raw weights.
2. **Render/Compare never read weights**: The entire visualization and comparison pipeline operates on artifacts (statistics, fields, projections).
3. **Determinism is part of the measurement protocol**: All RNGs are seeded from the spec. Byte-identical outputs guaranteed on the same machine.

### What Weight Atlas Is Not

- A **benchmark tool**: No capability claims, no performance metrics
- A **model editor**: Read-only; never modifies weights
- A **capability assessment**: Height = projection convention, not quality score

---

## Installation

### Core Installation

```bash
pip install -e .
```

Core dependencies: `numpy`, `safetensors`, `scipy`, `matplotlib`, `tifffile`

### Optional Extras

| Extra | Command | Dependencies | Purpose |
|-------|---------|--------------|---------|
| web | `pip install -e ".[web]"` | fastapi, jinja2, httpx, uvicorn | Web UI |
| gguf | `pip install -e ".[gguf]"` | gguf | GGUF model support |
| umap | `pip install -e ".[umap]"` | umap-learn | UMAP embedding projection |
| activity | `pip install -e ".[activity]"` | torch, transformers | fMRI mode |
| dev | `pip install -e ".[dev]"` | pytest, ruff, mypy | Development |

### Blender (External)

Blender is not a pip dependency. Install it separately and set `WEIGHT_ATLAS_BLENDER` environment variable if not in PATH.

---

## Quick Start

### Scan a Model

```bash
weight-atlas scan ./models/my_model --out ./artefacts
```

### Render Topographic Sheets

```bash
weight-atlas render ./artefacts --renderer sheet
```

### Compare Two Models

```bash
weight-atlas compare ./artefacts/model_a ./artefacts/model_b --out ./compare_out --mode strict
```

### Launch Web UI

```bash
pip install -e ".[web]"
weight-atlas serve               # LAN-reachable on http://0.0.0.0:8000
# Open http://<machine-ip>:8000 (or http://localhost:8000 on this machine)
```

For localhost-only access:

```bash
uvicorn weight_atlas.api.main:app --reload
# Open http://localhost:8000
```

> **Security**: the web UI has no authentication. `weight-atlas serve` binds
> `0.0.0.0` and exposes the job/import/artefact API to the LAN. Run only on a
> trusted network or behind a firewall/VPN.

---

## Core Concepts

### Pipeline

```
Model → TensorHandle (lazy) → Statistics → Rasterize → Field2D → Render
```

1. **Load**: Open model file (safetensors or GGUF), create lazy tensor handles
2. **Statistics**: Compute Frobenius, spectral norm, effective rank, kurtosis, sparsity
3. **Rasterize**: Map statistics to 2D grids (rows=layer, cols=slot)
4. **Scale + Smooth**: Apply channel scaling (log1p, quantile_clip) and Gaussian smoothing
5. **Render**: Generate PNG sheets or 3D terrain

### Slots

13 fixed slot categories:

| Slot | Description |
|------|-------------|
| embed | Token embeddings |
| attn_q/k/v | Attention query/key/value |
| attn_o | Attention output |
| mlp_gate/up/down | MLP gate/up/down projections |
| norm_attn/mlp | Layer norms |
| router | MoE router |
| lm_head | Language model head |
| other | Unrecognized tensors |

### Channels

| Channel | Statistic | Scale |
|---------|-----------|-------|
| height | spectral_norm | log1p → rank_scale (per_column) |
| tint | stable_rank | log1p → robust_scale (1-99%) |
| rough | kurtosis | rank_scale (per_column) |

### Determinism

All random number generators are seeded from `specs/atlas_spec.v2.4.json`. A second scan over the same input yields byte-identical artifacts (verified by SHA-256 manifest).

---

## CLI Reference

### Global Options

```bash
weight-atlas --help
weight-atlas <command> --help
```

### `scan` — Scan a Model

```bash
weight-atlas scan <path> --out <dir> [options]
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `path` | Path | Yes | Path to model file or directory |
| `--out` | Path | Yes | Output directory |
| `--spec` | Path | No | Path to atlas spec JSON (default: `specs/atlas_spec.v2.4.json`) |
| `--loader` | choice | No | `safetensors` or `gguf` (default: auto-detect) |

**Output**: `fingerprint.json`, `field_<channel>_raw.tif`, `field_<channel>_smooth.tif`, `embedding_pca.npy`, `embedding_meta.json`, `manifest.json`

### `render` — Render Artifacts

```bash
weight-atlas render <out_dir> [options]
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `out_dir` | Path | Yes | Directory containing scan artifacts |
| `--renderer` | str | No | Renderer plugin id (default: `sheet`) |
| `--field` | str | No | Field to render (default: `height`) |

**Available renderers**: `sheet` (matplotlib), `blender`, `delta` (comparison)

### `compare` — Compare Two Models

```bash
weight-atlas compare <dir_a> <dir_b> --out <dir> [options]
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `dir_a` | Path | Yes | Directory containing scan artifacts for model A |
| `dir_b` | Path | Yes | Directory containing scan artifacts for model B |
| `--out` | Path | Yes | Output directory for comparison artifacts |
| `--mode` | choice | No | `strict` (same architecture) or `aligned` (cross-architecture) |
| `--spec` | Path | No | Path to atlas spec JSON |

**Output**: `compare_summary.json`, `delta_sheet_<channel>.png`, `delta_profile_<channel>.png`, `delta_<channel>_raw.tif`

### `activity` — Capture Activations (fMRI Mode)

```bash
weight-atlas activity <model_path> --out <dir> [options]
```

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `model_path` | Path | Yes | Path to HuggingFace model directory |
| `--out` | Path | Yes | Output directory |
| `--protocol` | str | No | Protocol version (default: `v1`) |
| `--device` | str | No | Device (default: `cpu`) |
| `--dtype` | choice | No | `float32` or `bfloat16` |
| `--seed` | int | No | Random seed (default: 0) |
| `--max-layers` | int | No | Max layers to capture |

**Output**: `activity_meta.json`, `field_activity_<state>_residual_raw.tif`, `field_activity_<state>_experts_raw.tif`, `manifest.json`

---

## Web UI Guide

### Starting the Server

```bash
pip install -e ".[web]"
weight-atlas serve               # LAN-reachable on http://0.0.0.0:8000
```

Open http://localhost:8000 in your browser (or `http://<machine-ip>:8000` from
another machine on the LAN).

For localhost-only access use `weight-atlas serve --host 127.0.0.1` or run
`uvicorn weight_atlas.api.main:app --reload` directly.

> **Security**: the web UI has no authentication. Serving on `0.0.0.0` exposes
> the job/import/artefact API to the LAN. Run only on a trusted network or
> behind a firewall/VPN.

### Model List (`/`)

- View all completed scans and activity runs
- Activity entries show an "activity" badge
- Click "View" to see details

### Submit a Job

1. Enter a model path (local file or directory) — or click **Browse…** to open
   the server-side file picker and navigate the filesystem (folders open in the
   dialog; model files/directories get a *select* action).
2. Click "Start Scan"
3. Track progress via HTMX polling (updates every 2 seconds)

> The browse dialog uses `GET /api/browse` and lists only `.gguf`/`.safetensors`
> files plus directories. When the server is configured with an allowlist of
> model roots (`create_app(model_roots=...)`), the picker is confined to those
> roots and cannot browse elsewhere. The import form uses the same picker in
> "dir" mode (any directory is selectable).

### Model Detail (`/models/{job_id}`)

Sections:
- **Sheet**: Topographic sheet PNGs (height, tint, rough channels)
- **Terrain (Blender)**: 3D terrain renders + OBJ mesh
- **Statistics**: Table of per-tensor statistics
- **Quantization (GGUF)**: ggml_type distribution
- **MoE Architecture**: Expert count, shared expert badge (if MoE)
- **Embedding Projection**: PCA/UMAP density field, explained variance
- **Spec**: Specification details

### Compare Page (`/compare`)

1. Select two models to compare
2. Click "Run Comparison"
3. View delta visualizations, hotspot rankings, and summary metrics

### Compare Report (`/compare/{job_id}`)

- Summary metrics (rel. L2, cosine similarity, hotspot locations)
- Warnings (spec_version mismatch, tool_version mismatch, loader mismatch)
- Hotspot rankings (top-5 per channel)
- Delta visualizations
- Expert panel comparison (if MoE)

---

## Plugins

### Loader Plugins

| ID | Class | Description |
|----|-------|-------------|
| `safetensors` | `SafetensorsLoader` | Memory-mapped safetensors loader with sharding support |
| `gguf` | `GGUFLoader` | GGUF format loader with F32/F16/BF16/Q8_0/Q4_0 dequantization |

**Registration**: `@register_loader("id")`

**Auto-detection**: Magic bytes (`GGUF` vs safetensors header)

### Statistic Plugins

| ID | Class | Description |
|----|-------|-------------|
| `frobenius` | `FrobeniusNorm` | Frobenius norm with chunked float64 accumulation |
| `spectral_norm` | `SpectralNorm` | Spectral norm (largest singular value) via randomized SVD |
| `effective_rank` | `EffectiveRank` | Effective rank = exp(-sum(p_i * log p_i)) |
| `kurtosis` | `Kurtosis` | Excess kurtosis (Fisher) |
| `sparsity` | `Sparsity` | Fraction of weights with \|value\| < 1e-3 |

**Registration**: `@register_stat("id")`

### Renderer Plugins

| ID | Class | Description |
|----|-------|-------------|
| `sheet` | `MatplotlibSheet` | Hillshade + hypsometric tint + contours |
| `blender` | `BlenderRenderer` | 3D ortho terrain (18° pitch) + OBJ mesh |
| `delta` | `DeltaSheet` | Diverging colormap delta visualization + ablition bar profile |

**Registration**: `@register_renderer("id")`

### Using Plugins

```bash
# List available renderers
weight-atlas render --help

# Use specific renderer
weight-atlas render ./artefacts --renderer blender --field height
```

### Creating Custom Plugins

```python
from weight_atlas.core.registry import register_renderer
from weight_atlas.core.types import AtlasSpec, Field2D

@register_renderer("my_renderer")
class MyRenderer:
    renderer_id = "my_renderer"

    def render(self, field: Field2D, spec: AtlasSpec, out: Path) -> list[Path]:
        # Your rendering logic here
        return [output_path]
```

---

## Artifact Reference

### Scan Output

| File | Description |
|------|-------------|
| `fingerprint.json` | Per-tensor statistics + model metadata |
| `field_height_raw.tif` | Raw height channel (spectral_norm) |
| `field_height_smooth.tif` | Smoothed height channel |
| `field_tint_raw.tif` | Raw tint channel (effective_rank) |
| `field_tint_smooth.tif` | Smoothed tint channel |
| `field_rough_raw.tif` | Raw rough channel (kurtosis) |
| `field_rough_smooth.tif` | Smoothed rough channel |
| `field_embed_density_raw.tif` | Embedding PCA density field |
| `field_embed_density_smooth.tif` | Smoothed embedding density |
| `embedding_pca.npy` | PCA projection (V×3 float32) |
| `embedding_meta.json` | Explained variance, method, conventions |
| `embedding_scatter.npy` | Subsampled scatter coordinates |
| `field_expert_mlp_{gate,up,down}_<channel>_raw.tif` | MoE expert panels (if MoE) |
| `manifest.json` | SHA-256 per artifact |

### Compare Output

| File | Description |
|------|-------------|
| `compare_summary.json` | Global metrics + per-channel metrics + hotspot rankings |
| `delta_sheet_height.png` | Delta sheet (diverging colormap) |
| `delta_profile_height.png` | 1×L ablition bar |
| `delta_height_raw.tif` | Raw delta field |

### Activity Output

| File | Description |
|------|-------------|
| `activity_meta.json` | Protocol hash, device, dtype, torch/transformers versions |
| `field_activity_<state>_residual_raw.tif` | Layer × Position residual RMS |
| `field_activity_<state>_experts_raw.tif` | Layer × Expert usage (MoE only) |
| `manifest.json` | SHA-256 per artifact |

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `manifest.json not found` | Run `scan` before `render` |
| `spec_version mismatch` | Ensure both models scanned with same spec version |
| `strict mode requires identical shapes` | Use `--mode aligned` for cross-architecture comparison |
| `height TIFF not found` | Run `scan` with embedding enabled |
| `unknown loader: gguf` | Install gguf extra: `pip install -e ".[gguf]"` |
| `Unsupported GGUF quantization type` | Only F32/F16/BF16/Q8_0/Q4_0 supported (see backlog) |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (see stderr) |

### Getting Help

```bash
weight-atlas --help
weight-atlas scan --help
weight-atlas render --help
weight-atlas compare --help
weight-atlas activity --help
```

---

## Appendix: Spec File Reference

```json
{
  "spec_version": 1,
  "slots": ["embed","attn_q","attn_k","attn_v","attn_o","mlp_gate","mlp_up",
             "mlp_down","norm_attn","norm_mlp","router","lm_head","other"],
  "channels": {
    "height": {"stat": "spectral_norm", "pre": "log1p", "scale": {"type": "rank_scale", "per_column": true}},
    "tint": {"stat": "stable_rank", "pre": "log1p", "scale": {"type": "robust_scale", "lower": 0.01, "upper": 0.99}},
    "rough": {"stat": "kurtosis", "scale": {"type": "rank_scale", "per_column": true}}
  },
  "grid": {"upsample": 8, "smooth_sigma": 1.0},
  "sheet": {"contour_levels": 12, "light_azdeg": 315, "light_altdeg": 45, "dpi": 150},
  "blender": {"grid": 1024, "resolution": 2048, "z_scale": 0.3, "pitch": 18.0, "clip": 0.01, "adaptive_z_scale": false},
  "compare": {
    "modes": ["strict", "aligned"],
    "default_mode": "strict",
    "aligned_grid": 64,
    "colormap": "RdBu_r",
    "diverging_clip": 0.98,
    "aligned_interp": "linear"
  },
  "embedding": {
    "method": "pca",
    "grid": 256,
    "components": 3,
    "subsample_scatter": 5000,
    "seeds": {"pca": 0, "umap": 0}
  },
  "seeds": {"svd": 0}
}
```

`diverging_clip` (default 0.98) steuert das symmetrische Limit der Delta-Colorbar:
es ist das Quantil von |Δ| je Kanal, zusätzlich gedeckelt auf eine robuste Spannweite
(Median + 4.4826·MAD, ≈3σ), damit einzelne Ausreißer den Massenbereich nicht
"flach" Richtung Weiß drücken. Slot-Spalten, die in einem Modell komplett fehlen
(NaN), werden beim Rendern automatisch ausgeblendet; die Zuordnung Original→behalten
ist über `kept_cols` am Renderer abrufbar.
