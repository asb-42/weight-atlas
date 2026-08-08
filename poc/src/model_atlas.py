"""
Weight Atlas - Main Entry Point
================================
Command-line interface for extracting and visualizing LLM fingerprints.

Modes:
    --model PATH       Extract from a real safetensors model
    --demo             Run demo with synthetic models (no model files needed)
    --compare A B      Compare two models side-by-side
    --diff A B         Show what changed between models (e.g., abliteration)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from .extractor import WeightExtractor, ModelFingerprint, LayerProfile
from .visualizer import AtlasVisualizer


def create_synthetic_fingerprint(name: str, seed: int = 42) -> ModelFingerprint:
    """
    Create a synthetic model fingerprint for demonstration.
    
    This simulates what a real LLM fingerprint looks like,
    showing the kinds of patterns that emerge.
    """
    rng = np.random.RandomState(seed)
    
    fingerprint = ModelFingerprint(
        model_name=name,
        total_parameters=7_000_000_000,  # 7B
        num_layers=32,
    )
    
    # Simulate realistic layer patterns
    for i in range(32):
        # Create layer name
        layer_name = f"model.layers.{i}.self_attn.q_proj.weight"
        
        # Simulate training dynamics:
        # - Early layers: low std (shallow features)
        # - Middle layers: high std (rich representations)  
        # - Late layers: medium std (refined outputs)
        depth_factor = np.sin(np.pi * i / 31)  # Bell curve
        
        # Add some randomness for individuality
        layer_seed = seed + i
        layer_rng = np.random.RandomState(layer_seed)
        
        # Simulate weight statistics
        std = 0.02 + 0.03 * depth_factor + layer_rng.uniform(-0.005, 0.005)
        mean = layer_rng.uniform(-0.001, 0.001)
        spectral_entropy = 3.0 + 1.5 * depth_factor + layer_rng.uniform(-0.2, 0.2)
        sparsity = 0.05 + 0.1 * (1 - depth_factor) + layer_rng.uniform(-0.02, 0.02)
        
        profile = LayerProfile(
            name=layer_name,
            shape=(4096, 4096),
            mean=float(mean),
            std=float(std),
            skewness=float(layer_rng.uniform(-0.5, 0.5)),
            kurtosis=float(layer_rng.uniform(0, 3)),
            sparsity=float(sparsity),
            spectral_entropy=float(spectral_entropy),
            effective_rank=float(50 + 100 * depth_factor + layer_rng.uniform(-10, 10)),
            l1_norm=float(std * 4096 * 0.8),
            l2_norm=float(std * 4096 * 0.6),
            max_abs=float(std * 4),  # ~4 sigma
            percentile_25=float(std * 0.67),
            percentile_50=float(std * 0.9),
            percentile_75=float(std * 1.33),
            percentile_99=float(std * 3.3),
            attention_head_diversity=float(0.3 + 0.4 * depth_factor + layer_rng.uniform(-0.1, 0.1)),
            attention_entropy=float(2.0 + 0.5 * depth_factor + layer_rng.uniform(-0.2, 0.2)),
        )
        fingerprint.layers[layer_name] = profile
    
    # Add MLP layers
    for i in range(32):
        layer_name = f"model.layers.{i}.mlp.gate_proj.weight"
        depth_factor = np.sin(np.pi * i / 31)
        layer_seed = seed + i + 100
        layer_rng = np.random.RandomState(layer_seed)
        
        std = 0.015 + 0.02 * depth_factor + layer_rng.uniform(-0.003, 0.003)
        
        profile = LayerProfile(
            name=layer_name,
            shape=(4096, 11008),
            mean=float(layer_rng.uniform(-0.001, 0.001)),
            std=float(std),
            skewness=float(layer_rng.uniform(-0.3, 0.3)),
            kurtosis=float(layer_rng.uniform(0, 2)),
            sparsity=float(0.08 + 0.05 * (1 - depth_factor)),
            spectral_entropy=float(2.5 + 1.0 * depth_factor + layer_rng.uniform(-0.2, 0.2)),
            effective_rank=float(40 + 80 * depth_factor),
            l1_norm=float(std * 4096 * 0.8),
            l2_norm=float(std * 4096 * 0.6),
            max_abs=float(std * 4),
            percentile_25=float(std * 0.67),
            percentile_50=float(std * 0.9),
            percentile_75=float(std * 1.33),
            percentile_99=float(std * 3.3),
        )
        fingerprint.layers[layer_name] = profile
    
    # Global stats
    all_stds = [l.std for l in fingerprint.layers.values()]
    all_spectral = [l.spectral_entropy for l in fingerprint.layers.values()]
    all_ranks = [l.effective_rank for l in fingerprint.layers.values()]
    
    fingerprint.global_stats = {
        "mean_std": float(np.mean(all_stds)),
        "std_std": float(np.std(all_stds)),
        "mean_spectral_entropy": float(np.mean(all_spectral)),
        "mean_effective_rank": float(np.mean(all_ranks)),
        "total_sparsity": float(np.mean([l.sparsity for l in fingerprint.layers.values()])),
        "mean_of_means": float(np.mean([l.mean for l in fingerprint.layers.values()])),
        "std_of_means": float(np.std([l.mean for l in fingerprint.layers.values()])),
    }
    
    fingerprint.layer_depth_profile = [
        fingerprint.layers[f"model.layers.{i}.self_attn.q_proj.weight"].std 
        for i in range(32)
    ]
    
    return fingerprint


def create_abliterated_fingerprint(base_name: str, seed: int = 42) -> ModelFingerprint:
    """
    Simulate an abliterated model - where certain 'refusal' directions
    have been removed. This shows up as localized changes in specific layers.
    """
    # Start from base
    fp = create_synthetic_fingerprint(base_name, seed)
    
    # Abliteration typically modifies attention layers in the middle-to-late network
    rng = np.random.RandomState(seed + 1000)
    
    for i in range(16, 28):  # Middle-to-late layers
        for suffix in ["q_proj", "o_proj"]:
            name = f"model.layers.{i}.self_attn.{suffix}.weight"
            if name in fp.layers:
                layer = fp.layers[name]
                # Abliteration increases entropy slightly (removes structure)
                # and changes the spectral properties
                layer.spectral_entropy *= 1.15 + rng.uniform(0, 0.1)
                layer.std *= 0.95 + rng.uniform(-0.02, 0.05)
                layer.effective_rank *= 1.1 + rng.uniform(0, 0.1)
                layer.attention_entropy *= 1.2 + rng.uniform(0, 0.1)
                # Slight increase in kurtosis (distribution becomes more peaked)
                layer.kurtosis += rng.uniform(0.1, 0.3)
    
    # Update global stats
    all_spectral = [l.spectral_entropy for l in fp.layers.values()]
    fp.global_stats['mean_spectral_entropy'] = float(np.mean(all_spectral))
    
    return fp


def run_demo(output_dir: str):
    """Run demonstration with synthetic models."""
    print("\n" + "=" * 60)
    print("🧪 WEIGHT ATLAS - DEMO MODE")
    print("=" * 60)
    print("\nGenerating synthetic model fingerprints...")
    print("(In production, these would come from real .safetensors files)")
    
    # Create two different "models"
    qwen_35b = create_synthetic_fingerprint("Qwen3.6-35B-A3B", seed=42)
    qwen_27b = create_synthetic_fingerprint("Qwen3.6-27B", seed=123)
    
    # Create abliterated version
    qwen_35b_abliterated = create_abliterated_fingerprint("Qwen3.6-35B-A3B-Abliterated", seed=42)
    
    # Visualize
    viz = AtlasVisualizer(style="mri")
    
    print("\n📸 Generating visualizations for Model A...")
    viz.create_complete_atlas(qwen_35b, output_dir)
    
    print("\n📸 Generating visualizations for Model B...")
    viz.create_complete_atlas(qwen_27b, output_dir)
    
    # Comparison radar chart
    print("\n📊 Creating comparison radar chart...")
    viz.create_fingerprint_radar(
        [qwen_35b, qwen_27b, qwen_35b_abliterated],
        Path(output_dir) / "model_comparison_radar.png"
    )
    plt.close()
    
    # Diff map (showing abliteration effects)
    print("\n📊 Creating diff map (abliteration effects)...")
    viz.create_diff_map(
        qwen_35b, qwen_35b_abliterated,
        Path(output_dir) / "abliteration_diff_map.png"
    )
    plt.close()
    
    # Save fingerprints
    import json
    output_path = Path(output_dir)
    
    for fp in [qwen_35b, qwen_27b, qwen_35b_abliterated]:
        with open(output_path / f"{fp.model_name}_fingerprint.json", "w") as f:
            json.dump(fp.to_dict(), f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated files in: {output_dir}")
    print("\nView these files:")
    print(f"  📁 {output_dir}/Qwen3.6-35B-A3B_sagittal_slice.png")
    print(f"  📁 {output_dir}/Qwen3.6-35B-A3B_activation_map.png")
    print(f"  📁 {output_dir}/Qwen3.6-35B-A3B_topography_3d.png")
    print(f"  📁 {output_dir}/Qwen3.6-27B_sagittal_slice.png")
    print(f"  📁 {output_dir}/model_comparison_radar.png")
    print(f"  📁 {output_dir}/abliteration_diff_map.png")
    print(f"  📁 {output_dir}/*.json (raw fingerprint data)")


def run_extraction(model_path: str, output_dir: str, model_name: Optional[str] = None):
    """Extract fingerprint from a real model."""
    print(f"\n🔍 Extracting fingerprint from: {model_path}")
    
    extractor = WeightExtractor(model_path, model_name)
    fingerprint = extractor.build_fingerprint()
    
    extractor.save_fingerprint(output_dir)
    
    viz = AtlasVisualizer(style="mri")
    viz.create_complete_atlas(fingerprint, output_dir)
    
    print(f"\n✅ Complete! Results saved to {output_dir}")


def run_comparison(model_a_path: str, model_b_path: str, output_dir: str):
    """Compare two models."""
    print(f"\n📊 Comparing models:")
    print(f"  Model A: {model_a_path}")
    print(f"  Model B: {model_b_path}")
    
    extractor_a = WeightExtractor(model_a_path, "Model_A")
    fp_a = extractor_a.build_fingerprint()
    
    extractor_b = WeightExtractor(model_b_path, "Model_B")
    fp_b = extractor_b.build_fingerprint()
    
    viz = AtlasVisualizer(style="mri")
    
    # Generate individual atlases
    viz.create_complete_atlas(fp_a, output_dir)
    viz.create_complete_atlas(fp_b, output_dir)
    
    # Comparison visualizations
    viz.create_fingerprint_radar(
        [fp_a, fp_b],
        Path(output_dir) / "comparison_radar.png"
    )
    plt.close()
    
    viz.create_diff_map(
        fp_a, fp_b,
        Path(output_dir) / "comparison_diff_map.png"
    )
    plt.close()
    
    print(f"\n✅ Comparison complete! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Weight Atlas - Visualize LLMs like brain scans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --demo
  %(prog)s --model ./models/Qwen3.6-35B-A3B --output ./output
  %(prog)s --compare ./models/model_a ./models/model_b --output ./output
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", action="store_true",
                      help="Run demo with synthetic models")
    group.add_argument("--model", type=str,
                      help="Path to safetensors model directory")
    group.add_argument("--compare", nargs=2, metavar=("MODEL_A", "MODEL_B"),
                      help="Compare two models")
    group.add_argument("--diff", nargs=2, metavar=("MODEL_A", "MODEL_B"),
                      help="Show changes between models (e.g., abliteration)")
    
    parser.add_argument("--output", type=str, default="./output",
                       help="Output directory (default: ./output)")
    parser.add_argument("--name", type=str, default=None,
                       help="Model name (for --model mode)")
    parser.add_argument("--style", type=str, default="mri",
                       choices=["mri", "geographic", "neural", "ocean", "fingerprint"],
                       help="Visualization style")
    
    args = parser.parse_args()
    
    output_dir = args.output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.demo:
        run_demo(output_dir)
    elif args.model:
        run_extraction(args.model, output_dir, args.name)
    elif args.compare:
        run_comparison(args.compare[0], args.compare[1], output_dir)
    elif args.diff:
        run_comparison(args.diff[0], args.diff[1], output_dir)


if __name__ == "__main__":
    main()
