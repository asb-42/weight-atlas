"""
Weight Atlas - LLM Fingerprint Extraction Engine
=================================================
Extracts unique, meaningful properties from model weights to create
individual 'signatures' or 'fingerprints' - like MRI slices through a brain.

Each property captures a different aspect of what makes a model unique:
- Weight distributions (training dynamics, regularization imprints)
- Spectral properties (functional decomposition signature)
- Attention patterns (cognitive architecture fingerprint)
- Sparsity patterns (pruning/compression history)
- Information-theoretic properties (entropy, mutual information)
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
from tqdm import tqdm


@dataclass
class LayerProfile:
    """Complete profile of a single layer - like a 'brain region' scan."""
    name: str
    shape: Tuple[int, ...]
    mean: float
    std: float
    skewness: float
    kurtosis: float
    sparsity: float  # fraction of near-zero values
    spectral_entropy: float  # from SVD
    effective_rank: float  # how many dimensions are actually used
    l1_norm: float
    l2_norm: float
    max_abs: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    percentile_99: float
    # Attention-specific metrics
    attention_head_diversity: Optional[float] = None
    attention_entropy: Optional[float] = None
    # Cross-layer connectivity
    correlation_with_previous: Optional[float] = None


@dataclass
class ModelFingerprint:
    """Complete fingerprint of an LLM - the full 'brain atlas'."""
    model_name: str
    total_parameters: int
    num_layers: int
    layers: Dict[str, LayerProfile] = field(default_factory=dict)
    global_stats: Dict[str, float] = field(default_factory=dict)
    # Derived 'geographic' features
    layer_depth_profile: List[float] = field(default_factory=list)
    attention_specialization_map: Optional[np.ndarray] = None
    weight_topology: Optional[np.ndarray] = None
    
    def to_dict(self) -> dict:
        """Serialize for storage/comparison."""
        return {
            "model_name": self.model_name,
            "total_parameters": self.total_parameters,
            "num_layers": self.num_layers,
            "global_stats": self.global_stats,
            "layers": {
                name: {k: v for k, v in layer.__dict__.items() 
                       if not isinstance(v, np.ndarray)}
                for name, layer in self.layers.items()
            }
        }


class WeightExtractor:
    """
    Extracts meaningful properties from safetensors/gguf model files.
    
    The key insight: each model has a unique 'imprint' left by:
    1. Architecture choices (layer count, width, attention heads)
    2. Training data distribution
    3. Random seed and initialization
    4. Training dynamics (learning rate, batch size effects)
    5. Post-training (quantization, ablation, fine-tuning)
    
    These imprints are visible in the weight distributions, spectral
    properties, and connectivity patterns - like geological strata.
    """
    
    def __init__(self, model_path: str, model_name: Optional[str] = None):
        self.model_path = Path(model_path)
        self.model_name = model_name or self.model_path.stem
        self.weights = {}
        self.fingerprint = None
        
    def load_safetensors(self) -> Dict[str, np.ndarray]:
        """Load weights from safetensors format."""
        from safetensors import safe_open
        
        weight_dict = {}
        safetensor_files = sorted(self.model_path.glob("*.safetensors"))
        
        if not safetensor_files:
            # Try single file
            single_file = self.model_path.with_suffix(".safetensors")
            if single_file.exists():
                safetensor_files = [single_file]
            else:
                raise FileNotFoundError(f"No safetensors found in {self.model_path}")
        
        for sf in tqdm(safetensor_files, desc="Loading safetensors"):
            with safe_open(sf, framework="np") as f:
                for key in f.keys():
                    weight_dict[key] = f.get_tensor(key)
        
        self.weights = weight_dict
        print(f"Loaded {len(weight_dict)} tensors, "
              f"total params: {sum(w.size for w in weight_dict.values()):,}")
        return weight_dict
    
    def extract_layer_profile(self, name: str, tensor: np.ndarray) -> LayerProfile:
        """Extract comprehensive profile from a single weight tensor."""
        flat = tensor.astype(np.float32).flatten()
        
        # Basic statistics
        mean = float(np.mean(flat))
        std = float(np.std(flat))
        
        # Higher-order moments (distribution shape fingerprint)
        if std > 1e-10:
            normalized = (flat - mean) / std
            skewness = float(np.mean(normalized ** 3))
            kurtosis = float(np.mean(normalized ** 4) - 3.0)  # excess kurtosis
        else:
            skewness = 0.0
            kurtosis = 0.0
        
        # Sparsity (how many near-zero weights)
        sparsity = float(np.mean(np.abs(flat) < 1e-6))
        
        # Spectral properties (for 2D weight matrices)
        spectral_entropy = 0.0
        effective_rank = 0.0
        if tensor.ndim >= 2:
            matrix = tensor.astype(np.float32)
            if tensor.ndim > 2:
                # Reshape conv/attention weights to 2D
                matrix = matrix.reshape(tensor.shape[0], -1)
            
            # SVD for spectral analysis
            try:
                u, s, vt = np.linalg.svd(matrix, full_matrices=False)
                # Normalize singular values
                s_norm = s / (s.sum() + 1e-10)
                # Spectral entropy - how spread out is the information
                spectral_entropy = float(-np.sum(s_norm * np.log(s_norm + 1e-10)))
                # Effective rank - how many dimensions matter
                cumulative = np.cumsum(s_norm)
                effective_rank = float(np.searchsorted(cumulative, 0.95) + 1)
            except np.linalg.LinAlgError:
                pass
        
        # Percentiles (distribution shape)
        p25, p50, p75, p99 = np.percentile(np.abs(flat), [25, 50, 75, 99])
        
        # Norms
        l1_norm = float(np.sum(np.abs(flat)))
        l2_norm = float(np.sqrt(np.sum(flat ** 2)))
        max_abs = float(np.max(np.abs(flat)))
        
        return LayerProfile(
            name=name,
            shape=tensor.shape,
            mean=mean,
            std=std,
            skewness=skewness,
            kurtosis=kurtosis,
            sparsity=sparsity,
            spectral_entropy=spectral_entropy,
            effective_rank=float(effective_rank),
            l1_norm=l1_norm,
            l2_norm=l2_norm,
            max_abs=max_abs,
            percentile_25=float(p25),
            percentile_50=float(p50),
            percentile_75=float(p75),
            percentile_99=float(p99),
        )
    
    def extract_attention_properties(self, name: str, tensor: np.ndarray) -> Tuple[float, float]:
        """
        Extract attention-specific properties.
        
        Attention heads have diverse specializations (syntax, semantics, etc.)
        This diversity is unique to each model's training.
        """
        if "attn" in name.lower() or "attention" in name.lower():
            if tensor.ndim >= 3:
                # Multi-head attention: analyze head diversity
                n_heads = tensor.shape[0]
                heads = tensor.reshape(n_heads, -1)
                
                # Normalize heads
                norms = np.linalg.norm(heads, axis=1, keepdims=True) + 1e-10
                heads_normalized = heads / norms
                
                # Head diversity: average pairwise cosine distance
                similarity = heads_normalized @ heads_normalized.T
                diversity = float(1 - np.mean(similarity))
                
                # Entropy of head norms (how balanced are heads?)
                head_norms = np.linalg.norm(heads, axis=1)
                head_norms_norm = head_norms / head_norms.sum()
                entropy = float(-np.sum(head_norms_norm * np.log(head_norms_norm + 1e-10)))
                
                return diversity, entropy
        
        return 0.0, 0.0
    
    def build_fingerprint(self) -> ModelFingerprint:
        """Build complete model fingerprint from all weights."""
        if not self.weights:
            self.load_safetensors()
        
        fingerprint = ModelFingerprint(
            model_name=self.model_name,
            total_parameters=sum(w.size for w in self.weights.values()),
            num_layers=len(self.weights),
        )
        
        prev_flat = None
        
        for name, tensor in tqdm(self.weights.items(), desc="Extracting profiles"):
            profile = self.extract_layer_profile(name, tensor)
            
            # Attention-specific analysis
            head_div, head_ent = self.extract_attention_properties(name, tensor)
            profile.attention_head_diversity = head_div
            profile.attention_entropy = head_ent
            
            # Cross-layer correlation (connectivity fingerprint)
            flat = tensor.flatten().astype(np.float32)
            if prev_flat is not None and len(flat) == len(prev_flat):
                correlation = float(np.corrcoef(flat, prev_flat)[0, 1])
                profile.correlation_with_previous = correlation
            prev_flat = flat
            
            fingerprint.layers[name] = profile
        
        # Global statistics (the 'big picture' properties)
        all_means = [l.mean for l in fingerprint.layers.values()]
        all_stds = [l.std for l in fingerprint.layers.values()]
        all_spectral = [l.spectral_entropy for l in fingerprint.layers.values()]
        all_ranks = [l.effective_rank for l in fingerprint.layers.values()]
        
        fingerprint.global_stats = {
            "mean_of_means": float(np.mean(all_means)),
            "std_of_means": float(np.std(all_means)),
            "mean_std": float(np.mean(all_stds)),
            "std_std": float(np.std(all_stds)),
            "mean_spectral_entropy": float(np.mean(all_spectral)),
            "mean_effective_rank": float(np.mean(all_ranks)),
            "total_sparsity": float(np.mean([l.sparsity for l in fingerprint.layers.values()])),
        }
        
        # Layer depth profile - how properties change through the network
        # Like geological strata showing the model's 'formation history'
        sorted_layers = sorted(fingerprint.layers.keys())
        fingerprint.layer_depth_profile = [
            fingerprint.layers[name].std for name in sorted_layers
        ]
        
        self.fingerprint = fingerprint
        return fingerprint
    
    def save_fingerprint(self, output_path: str):
        """Save fingerprint to disk for later comparison."""
        output = Path(output_path)
        output.mkdir(parents=True, exist_ok=True)
        
        # Save JSON summary
        with open(output / f"{self.model_name}_fingerprint.json", "w") as f:
            json.dump(self.fingerprint.to_dict(), f, indent=2)
        
        # Save full weight data as numpy arrays for visualization
        np.savez(
            output / f"{self.model_name}_profiles.npz",
            names=list(self.fingerprint.layers.keys()),
            means=np.array([l.mean for l in self.fingerprint.layers.values()]),
            stds=np.array([l.std for l in self.fingerprint.layers.values()]),
            skewness=np.array([l.skewness for l in self.fingerprint.layers.values()]),
            kurtosis=np.array([l.kurtosis for l in self.fingerprint.layers.values()]),
            sparsity=np.array([l.sparsity for l in self.fingerprint.layers.values()]),
            spectral_entropy=np.array([l.spectral_entropy for l in self.fingerprint.layers.values()]),
            effective_rank=np.array([l.effective_rank for l in self.fingerprint.layers.values()]),
        )
        
        print(f"Fingerprint saved to {output}")
