"""
Weight Atlas - LLM Fingerprint Visualizer
==========================================
Transforms extracted model properties into beautiful, informative
visualizations - like MRI brain scans or geographic maps.

Key visualization types:
1. 'Sagittal Slice' - Layer-depth profile showing model 'geology'
2. 'Activation Map' - Heatmap of weight properties across architecture
3. 'Spectral Topography' - 3D landscape of singular value distributions
4. 'Fingerprint Radar' - Multi-dimensional model comparison
5. 'Attention Constellation' - Head specialization patterns
6. 'Diff Map' - What changed between two models (e.g., abliteration)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


class AtlasVisualizer:
    """
    Creates aesthetically compelling visualizations of LLM fingerprints.
    
    Design philosophy: Make the invisible visible. Model weights contain
    rich information about training history, architecture, and function.
    Like geological strata, these patterns tell a story.
    """
    
    # Professional color schemes inspired by scientific imaging
    COLORMAPS = {
        "mri": "magma",           # MRI-style thermal
        "geographic": "terrain",  # Topographic map
        "neural": "inferno",      # Neural activity style
        "ocean": "viridis",       # Deep ocean mapping
        "fingerprint": "plasma",  # High contrast for comparison
    }
    
    def __init__(self, style: str = "mri"):
        self.style = style
        plt.style.use('dark_background')
        self.fig_size = (16, 10)
        self.dpi = 150
    
    def _get_cmap(self, name: str):
        """Get colormap by name, compatible with newer matplotlib versions."""
        return matplotlib.colormaps[name]
    
    def create_sagittal_slice(self, fingerprint, output_path: Optional[str] = None):
        """
        Create a 'sagittal slice' visualization - the signature view.
        
        Shows how weight properties change through the network layers,
        like an MRI slice through a brain or geological cross-section.
        Each band tells a story about what that layer learned.
        """
        fig, axes = plt.subplots(2, 2, figsize=self.fig_size)
        fig.suptitle(f"Neural Atlas: {fingerprint.model_name}\n"
                    f"Sagittal Slice - {fingerprint.num_layers} layers, "
                    f"{fingerprint.total_parameters:,} parameters",
                    fontsize=14, fontweight='bold', color='white')
        
        layer_names = list(fingerprint.layers.keys())
        x = np.arange(len(layer_names))
        
        # 1. Weight Distribution Topography (top-left)
        ax1 = axes[0, 0]
        means = np.array([fingerprint.layers[n].mean for n in layer_names])
        stds = np.array([fingerprint.layers[n].std for n in layer_names])
        
        # Create gradient fill effect
        ax1.fill_between(x, means - stds, means + stds, alpha=0.3, color='cyan')
        ax1.plot(x, means, color='white', linewidth=1.5, label='Mean')
        ax1.errorbar(x[::max(1, len(x)//20)], 
                    means[::max(1, len(x)//20)],
                    yerr=stds[::max(1, len(x)//20)],
                    fmt='o', color='cyan', markersize=3, alpha=0.7)
        ax1.set_title('Weight Distribution Profile', fontweight='bold')
        ax1.set_xlabel('Layer Depth')
        ax1.set_ylabel('Weight Value')
        ax1.legend()
        
        # 2. Spectral Entropy Map (top-right) - the 'activity' map
        ax2 = axes[0, 1]
        spectral = np.array([fingerprint.layers[n].spectral_entropy for n in layer_names])
        # Create pseudo-2D effect with color mapping
        cmap = self._get_cmap(self.COLORMAPS[self.style])
        colors = cmap(spectral / spectral.max())
        ax2.bar(x, spectral, color=colors, width=1.0, edgecolor='none')
        ax2.set_title('Spectral Entropy (Information Content)', fontweight='bold')
        ax2.set_xlabel('Layer Depth')
        ax2.set_ylabel('Entropy')
        
        # 3. Sparsity Strata (bottom-left)
        ax3 = axes[1, 0]
        sparsity = np.array([fingerprint.layers[n].sparsity for n in layer_names])
        effective_rank = np.array([fingerprint.layers[n].effective_rank for n in layer_names])
        
        ax3_twin = ax3.twinx()
        ax3.bar(x, sparsity, color='steelblue', alpha=0.7, label='Sparsity')
        ax3_twin.plot(x, effective_rank, color='orange', linewidth=2, label='Effective Rank')
        ax3.set_title('Sparsity & Effective Rank', fontweight='bold')
        ax3.set_xlabel('Layer Depth')
        ax3.set_ylabel('Sparsity Ratio', color='steelblue')
        ax3_twin.set_ylabel('Effective Rank', color='orange')
        
        # 4. Kurtosis/Skewness Distribution (bottom-right) - distribution fingerprint
        ax4 = axes[1, 1]
        skewness = np.array([fingerprint.layers[n].skewness for n in layer_names])
        kurtosis = np.array([fingerprint.layers[n].kurtosis for n in layer_names])
        
        scatter = ax4.scatter(skewness, kurtosis, c=x, cmap=self.COLORMAPS[self.style],
                             s=50, alpha=0.8, edgecolors='white', linewidth=0.5)
        ax4.set_title('Distribution Shape Space\n(colored by layer depth)', fontweight='bold')
        ax4.set_xlabel('Skewness')
        ax4.set_ylabel('Kurtosis')
        plt.colorbar(scatter, ax=ax4, label='Layer Depth')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='black', edgecolor='none')
            print(f"Sagittal slice saved to {output_path}")
        
        return fig
    
    def create_activation_heatmap(self, fingerprint, output_path: Optional[str] = None):
        """
        Create a 2D heatmap showing weight patterns across the model.
        
        Like fMRI activation maps - bright regions are 'active' (high information),
        dark regions are 'quiet' (sparse/concentrated).
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        fig.suptitle(f"Activation Atlas: {fingerprint.model_name}",
                    fontsize=14, fontweight='bold', color='white')
        
        layer_names = list(fingerprint.layers.keys())
        n_layers = len(layer_names)
        
        # Build property matrix
        properties = ['std', 'spectral_entropy', 'sparsity', 'effective_rank', 
                      'skewness', 'kurtosis']
        
        # Normalize each property to [0, 1]
        prop_matrix = np.zeros((n_layers, len(properties)))
        for j, prop in enumerate(properties):
            values = np.array([getattr(fingerprint.layers[n], prop) for n in layer_names])
            if values.max() > values.min():
                prop_matrix[:, j] = (values - values.min()) / (values.max() - values.min())
            else:
                prop_matrix[:, j] = 0.5
        
        # 1. Property correlation heatmap
        ax1 = axes[0]
        corr = np.corrcoef(prop_matrix.T)
        sns.heatmap(corr, annot=True, fmt='.2f', cmap=self.COLORMAPS[self.style],
                   xticklabels=properties, yticklabels=properties, ax=ax1,
                   square=True, cbar_kws={'shrink': 0.8})
        ax1.set_title('Property Correlations', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Layer property map
        ax2 = axes[1]
        # Interpolate for smoother visualization
        from scipy.ndimage import zoom
        smooth_matrix = zoom(prop_matrix, (5, 1), order=1)
        
        im = ax2.imshow(smooth_matrix.T, aspect='auto', cmap=self.COLORMAPS[self.style],
                       interpolation='bilinear')
        ax2.set_yticks(range(len(properties)))
        ax2.set_yticklabels(properties)
        ax2.set_xlabel('Layer Depth')
        ax2.set_title('Layer Property Map', fontweight='bold')
        plt.colorbar(im, ax=ax2, shrink=0.8)
        
        # 3. Layer clustering dendrogram
        ax3 = axes[2]
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import pdist
        
        # Cluster layers by their property profiles
        dist_matrix = pdist(prop_matrix, metric='euclidean')
        linked = linkage(dist_matrix, method='ward')
        
        dendrogram(linked, ax=ax3, orientation='right', no_labels=True,
                  color_threshold=0)
        ax3.set_title('Layer Similarity Tree\n(clusters = functional regions)',
                     fontweight='bold')
        ax3.set_xlabel('Distance')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='black', edgecolor='none')
            print(f"Activation heatmap saved to {output_path}")
        
        return fig
    
    def create_weight_topography_3d(self, fingerprint, output_path: Optional[str] = None):
        """
        Create a 3D topographic map of the model's weight landscape.
        
        X = layer depth, Y = property dimension, Z = property value
        Like a mountain range where peaks are high-information layers.
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        layer_names = list(fingerprint.layers.keys())
        n_layers = len(layer_names)
        
        # Create mesh grid
        properties = ['std', 'spectral_entropy', 'effective_rank', 'kurtosis']
        x = np.arange(n_layers)
        y = np.arange(len(properties))
        X, Y = np.meshgrid(x, y)
        
        # Z values
        Z = np.zeros_like(X, dtype=float)
        for i, prop in enumerate(properties):
            values = np.array([getattr(fingerprint.layers[n], prop) for n in layer_names])
            if values.max() > values.min():
                Z[i, :] = (values - values.min()) / (values.max() - values.min())
        
        # Surface plot
        surf = ax.plot_surface(X, Y, Z, cmap=self.COLORMAPS[self.style],
                              alpha=0.8, linewidth=0, antialiased=True)
        
        # Add contour lines
        ax.contour(X, Y, Z, zdir='z', offset=-0.2, cmap=self.COLORMAPS[self.style],
                  alpha=0.5)
        
        ax.set_xlabel('Layer Depth')
        ax.set_ylabel('Property')
        ax.set_zlabel('Normalized Value')
        ax.set_yticks(range(len(properties)))
        ax.set_yticklabels(properties)
        ax.set_title(f'Weight Topography: {fingerprint.model_name}',
                    fontweight='bold', pad=20)
        
        fig.colorbar(surf, shrink=0.5, aspect=10)
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='black', edgecolor='none')
            print(f"3D topography saved to {output_path}")
        
        return fig
    
    def create_fingerprint_radar(self, fingerprints: List, output_path: Optional[str] = None):
        """
        Radar chart comparing multiple models across key dimensions.
        
        Each model has a unique 'shape' - like a constellation.
        Overlapping models look similar; divergent models are clearly different.
        """
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Key dimensions for comparison
        dimensions = [
            'Mean Std', 'Mean Spectral Entropy', 'Mean Sparsity',
            'Mean Kurtosis', 'Mean Effective Rank', 'Parameter Count (norm)'
        ]
        
        # Normalize across models
        all_values = []
        for fp in fingerprints:
            values = [
                fp.global_stats['mean_std'],
                fp.global_stats['mean_spectral_entropy'],
                fp.global_stats['total_sparsity'],
                abs(fp.global_stats.get('mean_of_means', 0)),
                fp.global_stats['mean_effective_rank'],
                fp.total_parameters / 1e9,  # Billions
            ]
            all_values.append(values)
        
        all_values = np.array(all_values)
        # Normalize to [0, 1]
        for i in range(all_values.shape[1]):
            col = all_values[:, i]
            if col.max() > col.min():
                all_values[:, i] = (col - col.min()) / (col.max() - col.min() + 1e-10)
        
        # Plot each model
        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        cmap = self._get_cmap('Set2')
        colors = cmap(np.linspace(0, 1, len(fingerprints)))
        
        for idx, (fp, values) in enumerate(zip(fingerprints, all_values)):
            vals = values.tolist()
            vals += vals[:1]
            ax.plot(angles, vals, 'o-', linewidth=2, label=fp.model_name, color=colors[idx])
            ax.fill(angles, vals, alpha=0.1, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, fontsize=10)
        ax.set_title('Model Fingerprint Comparison\n(Radar Chart)',
                    fontsize=14, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='black', edgecolor='none')
            print(f"Radar chart saved to {output_path}")
        
        return fig
    
    def create_diff_map(self, fingerprint_a, fingerprint_b, output_path: Optional[str] = None):
        """
        Show what changed between two models (e.g., before/after abliteration).
        
        This is the most practical visualization - it literally shows
        where the model was modified, like highlighting tumor removal in MRI.
        """
        fig, axes = plt.subplots(2, 2, figsize=self.fig_size)
        fig.suptitle(f"Model Diff Atlas: {fingerprint_a.model_name} -> {fingerprint_b.model_name}\n"
                    f"(Red = increased, Blue = decreased)",
                    fontsize=14, fontweight='bold', color='white')
        
        # Get common layers
        common_layers = set(fingerprint_a.layers.keys()) & set(fingerprint_b.layers.keys())
        layer_names = sorted(common_layers)
        x = np.arange(len(layer_names))
        
        # 1. Std difference
        ax1 = axes[0, 0]
        std_a = np.array([fingerprint_a.layers[n].std for n in layer_names])
        std_b = np.array([fingerprint_b.layers[n].std for n in layer_names])
        diff = std_b - std_a
        
        colors = ['red' if d > 0 else 'blue' for d in diff]
        ax1.bar(x, diff, color=colors, alpha=0.7)
        ax1.set_title('Weight Std Difference', fontweight='bold')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Delta Std')
        ax1.axhline(y=0, color='white', linewidth=0.5)
        
        # 2. Spectral entropy difference
        ax2 = axes[0, 1]
        se_a = np.array([fingerprint_a.layers[n].spectral_entropy for n in layer_names])
        se_b = np.array([fingerprint_b.layers[n].spectral_entropy for n in layer_names])
        diff_se = se_b - se_a
        
        ax2.fill_between(x, 0, diff_se, where=diff_se > 0, color='red', alpha=0.3)
        ax2.fill_between(x, 0, diff_se, where=diff_se < 0, color='blue', alpha=0.3)
        ax2.plot(x, diff_se, color='white', linewidth=1)
        ax2.set_title('Spectral Entropy Difference', fontweight='bold')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Delta Spectral Entropy')
        ax2.axhline(y=0, color='white', linewidth=0.5)
        
        # 3. Absolute change heatmap
        ax3 = axes[1, 0]
        properties = ['std', 'mean', 'spectral_entropy', 'sparsity', 'effective_rank']
        change_matrix = np.zeros((len(layer_names), len(properties)))
        
        for j, prop in enumerate(properties):
            vals_a = np.array([getattr(fingerprint_a.layers[n], prop) for n in layer_names])
            vals_b = np.array([getattr(fingerprint_b.layers[n], prop) for n in layer_names])
            # Normalize change
            range_vals = max(vals_a.max() - vals_a.min(), 1e-10)
            change_matrix[:, j] = np.abs(vals_b - vals_a) / range_vals
        
        im = ax3.imshow(change_matrix.T, aspect='auto', cmap='hot',
                       interpolation='nearest')
        ax3.set_yticks(range(len(properties)))
        ax3.set_yticklabels(properties)
        ax3.set_xlabel('Layer')
        ax3.set_title('Relative Change Intensity', fontweight='bold')
        plt.colorbar(im, ax=ax3, shrink=0.8)
        
        # 4. Change distribution
        ax4 = axes[1, 1]
        total_change = np.mean(change_matrix, axis=1)
        ax4.hist(total_change, bins=30, color='purple', alpha=0.7, edgecolor='white')
        ax4.axvline(x=np.mean(total_change), color='red', linestyle='--',
                   label=f'Mean: {np.mean(total_change):.3f}')
        ax4.set_title('Distribution of Changes\nacross all layers',
                     fontweight='bold')
        ax4.set_xlabel('Average Relative Change')
        ax4.set_ylabel('Count')
        ax4.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='black', edgecolor='none')
            print(f"Diff map saved to {output_path}")
        
        return fig
    
    def create_complete_atlas(self, fingerprint, output_dir: str):
        """Generate all visualizations for a model."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        name = fingerprint.model_name
        
        print(f"\nGenerating atlas for: {name}")
        print("=" * 50)
        
        # Sagittal slice (signature view)
        print("  Creating sagittal slice...")
        self.create_sagittal_slice(
            fingerprint,
            output_dir / f"{name}_sagittal_slice.png"
        )
        plt.close()
        
        # Activation heatmap
        print("  Creating activation heatmap...")
        self.create_activation_heatmap(
            fingerprint,
            output_dir / f"{name}_activation_map.png"
        )
        plt.close()
        
        # 3D topography
        print("  Creating 3D topography...")
        self.create_weight_topography_3d(
            fingerprint,
            output_dir / f"{name}_topography_3d.png"
        )
        plt.close()
        
        print(f"\nAtlas complete! Saved to {output_dir}")
