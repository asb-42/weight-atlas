# 🧠 Weight Atlas - LLM Fingerprint Visualization

Visualize LLMs like brain scans. Each model gets a unique "fingerprint" extracted from its weight distributions, spectral properties, and attention patterns - like MRI slices through a brain or geological strata.

## Concept

Every LLM has unique characteristics imprinted by:
1. **Architecture choices** (layer count, width, attention heads)
2. **Training data distribution** 
3. **Random seed and initialization**
4. **Training dynamics** (learning rate, batch size effects)
5. **Post-training modifications** (quantization, ablation, fine-tuning)

These imprints are visible in weight distributions, spectral properties, and connectivity patterns - creating a unique "fingerprint" for each model.

## What It Extracts

| Property | What It Captures |
|----------|------------------|
| **Weight Distribution** | Mean, std, skewness, kurtosis per layer |
| **Spectral Properties** | SVD singular values, effective rank |
| **Sparsity Patterns** | Near-zero weight fraction per layer |
| **Attention Specialization** | Head diversity, entropy |
| **Cross-layer Connectivity** | Correlation between adjacent layers |

## Visualizations

### 1. Sagittal Slice
Like an MRI brain scan - shows how properties change through layers.

### 2. Activation Heatmap
Like fMRI - bright regions are information-rich, dark regions are sparse.

### 3. 3D Topography
Mountain range where peaks = high-information layers.

### 4. Fingerprint Radar
Multi-dimensional comparison - each model has a unique "shape".

### 5. Diff Map
Shows what changed between models (e.g., before/after abliteration).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Demo Mode (no model files needed)
```bash
python -m src.model_atlas --demo --output ./output
```

### Extract from Real Model
```bash
python -m src.model_atlas --model ./models/Qwen3.6-35B-A3B --output ./output
```

### Compare Two Models
```bash
python -m src.model_atlas --compare ./models/model_a ./models/model_b --output ./output
```

### Show Abliteration Effects
```bash
python -m src.model_atlas --diff ./models/original ./models/abliterated --output ./output
```

## Example: Abliteration Detection

When a model is abliterated (refusal directions removed), you'll see:
- Increased spectral entropy in modified layers (less structure)
- Changed effective rank (different dimensional usage)
- Modified attention entropy (altered head specialization)
- Localized changes in middle-to-late layers (typical ablation target)

## Architecture

```
weight-atlas/
├── src/
│   ├── __init__.py          # Package init
│   ├── extractor.py         # Weight extraction engine
│   ├── visualizer.py        # Visualization generators
│   └── model_atlas.py       # CLI entry point
├── output/                  # Generated visualizations
├── models/                  # Model files (safetensors)
├── requirements.txt
└── README.md
```

## Future: Activity Mapping

The next step is to map actual activation patterns:
- Feed real inputs through the model
- Capture per-layer, per-neuron activations
- Create "functional MRI" showing which regions activate for which inputs
- Compare activation patterns between models on identical inputs

This would show not just the "anatomy" (weights) but the "function" (activations) - like comparing structural vs functional MRI.

## License

MIT
