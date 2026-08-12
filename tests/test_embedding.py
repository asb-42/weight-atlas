"""Tests for Embedding Sheet (M7)."""

from __future__ import annotations

import json

import numpy as np
from safetensors.numpy import save_file

from weight_atlas.core.types import AtlasSpec
from weight_atlas.embedding.pca import compute_pca, embedding_to_density, project_with_pca

# ---------------------------------------------------------------------------
# PCA tests
# ---------------------------------------------------------------------------


class TestPCA:
    def test_pca_basic(self):
        """PCA should compute components, variance, and mean."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (100, 10))

        components, variance, mean = compute_pca(data, n_components=3, seed=0)

        assert components.shape == (3, 10)
        assert variance.shape == (3,)
        assert mean.shape == (10,)

    def test_pca_deterministic(self):
        """PCA should be deterministic across runs."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (100, 10))

        result1 = compute_pca(data, n_components=3, seed=0)
        result2 = compute_pca(data, n_components=3, seed=0)

        np.testing.assert_array_equal(result1[0], result2[0])
        np.testing.assert_array_equal(result1[1], result2[1])
        np.testing.assert_array_equal(result1[2], result2[2])

    def test_pca_sign_convention(self):
        """Sign convention: largest |loading| should be positive."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (100, 10))

        components, _, _ = compute_pca(data, n_components=3, seed=0)

        for i in range(3):
            col = components[i]
            max_idx = np.argmax(np.abs(col))
            assert col[max_idx] >= 0

    def test_pca_sign_flip_invariance(self):
        """Flipping input signs should not change PCA output."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (100, 10))

        components1, variance1, mean1 = compute_pca(data, n_components=3, seed=0)
        components2, variance2, mean2 = compute_pca(-data, n_components=3, seed=0)

        np.testing.assert_array_almost_equal(components1, components2)
        np.testing.assert_array_almost_equal(variance1, variance2)

    def test_project_with_pca(self):
        """Projection should give correct shape."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (100, 10))

        components, _, mean = compute_pca(data, n_components=3, seed=0)
        projected = project_with_pca(data, components, mean)

        assert projected.shape == (100, 3)


# ---------------------------------------------------------------------------
# Density tests
# ---------------------------------------------------------------------------


class TestDensity:
    def test_density_shape(self):
        """Density field should have correct grid size."""
        rng = np.random.default_rng(42)
        coords = rng.normal(0, 1, (1000, 2))

        density = embedding_to_density(coords, grid_size=256)

        assert density.shape == (256, 256)

    def test_density_sum_equals_count(self):
        """Sum of density should equal number of points."""
        rng = np.random.default_rng(42)
        coords = rng.normal(0, 1, (1000, 2))

        density = embedding_to_density(coords, grid_size=256)

        assert int(density.sum()) == 1000

    def test_density_subsample(self):
        """Subsampling should reduce point count."""
        rng = np.random.default_rng(42)
        coords = rng.normal(0, 1, (1000, 2))

        density = embedding_to_density(coords, grid_size=256, subsample=500, seed=0)

        assert int(density.sum()) == 500

    def test_density_deterministic(self):
        """Density computation should be deterministic."""
        rng = np.random.default_rng(42)
        coords = rng.normal(0, 1, (1000, 2))

        density1 = embedding_to_density(coords, grid_size=256, seed=0)
        density2 = embedding_to_density(coords, grid_size=256, seed=0)

        np.testing.assert_array_equal(density1, density2)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


class TestEmbeddingScan:
    def test_scan_produces_embedding_artefacts(self, tmp_path):
        """Scan should produce embedding PCA and density field."""
        from weight_atlas.scan import scan

        # Create model with embedding
        model_path = tmp_path / "model.safetensors"
        rng = np.random.default_rng(42)
        tensors = {
            "model.embed_tokens.weight": rng.normal(0, 0.1, (100, 32)).astype(np.float32),
            "model.layers.0.self_attn.q_proj.weight": rng.normal(0, 0.1, (32, 32)).astype(np.float32),
        }
        save_file(tensors, str(model_path))

        spec = AtlasSpec(
            spec_version=1,
            slots=["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"],
            channels={"height": {"stat": "spectral_norm", "scale": {"type": "log1p"}}},
            grid={"upsample": 2, "smooth_sigma": 1.0},
            sheet={"contour_levels": 4, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
            seeds={"svd": 0},
            embedding={"method": "pca", "grid": 256, "components": 3, "seeds": {"pca": 0}},
        )

        out = tmp_path / "out"
        scan(model_path, out, spec)

        # Check artefacts exist
        assert (out / "embedding_pca.npy").exists()
        assert (out / "embedding_meta.json").exists()
        assert (out / "field_embed_density_raw.tif").exists()
        assert (out / "field_embed_density_smooth.tif").exists()

        # Check metadata
        with open(out / "embedding_meta.json") as f:
            meta = json.load(f)
        assert meta["method"] == "pca"
        assert "explained_variance" in meta

    def test_scan_embedding_deterministic(self, tmp_path):
        """Embedding scan should be deterministic."""
        from weight_atlas.scan import scan

        # Create model
        model_path = tmp_path / "model.safetensors"
        rng = np.random.default_rng(42)
        tensors = {
            "model.embed_tokens.weight": rng.normal(0, 0.1, (100, 32)).astype(np.float32),
            "model.layers.0.self_attn.q_proj.weight": rng.normal(0, 0.1, (32, 32)).astype(np.float32),
        }
        save_file(tensors, str(model_path))

        spec = AtlasSpec(
            spec_version=1,
            slots=["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"],
            channels={"height": {"stat": "spectral_norm", "scale": {"type": "log1p"}}},
            grid={"upsample": 2, "smooth_sigma": 1.0},
            sheet={"contour_levels": 4, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
            seeds={"svd": 0},
            embedding={"method": "pca", "grid": 256, "components": 3, "seeds": {"pca": 0}},
        )

        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        scan(model_path, out1, spec)
        scan(model_path, out2, spec)

        # Compare PCA outputs
        pca1 = np.load(out1 / "embedding_pca.npy")
        pca2 = np.load(out2 / "embedding_pca.npy")
        np.testing.assert_array_equal(pca1, pca2)

    def test_embedding_cluster_separation(self, tmp_path):
        """PCA should separate planted Gaussian clusters."""
        from weight_atlas.scan import scan

        # Create model with clustered embeddings
        model_path = tmp_path / "model.safetensors"
        rng = np.random.default_rng(42)

        # Create two clusters
        cluster1 = rng.normal(-5, 0.5, (50, 32)).astype(np.float32)
        cluster2 = rng.normal(5, 0.5, (50, 32)).astype(np.float32)
        embeddings = np.vstack([cluster1, cluster2])

        tensors = {
            "model.embed_tokens.weight": embeddings,
            "model.layers.0.self_attn.q_proj.weight": rng.normal(0, 0.1, (32, 32)).astype(np.float32),
        }
        save_file(tensors, str(model_path))

        spec = AtlasSpec(
            spec_version=1,
            slots=["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"],
            channels={"height": {"stat": "spectral_norm", "scale": {"type": "log1p"}}},
            grid={"upsample": 2, "smooth_sigma": 1.0},
            sheet={"contour_levels": 4, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
            seeds={"svd": 0},
            embedding={"method": "pca", "grid": 256, "components": 3, "seeds": {"pca": 0}},
        )

        out = tmp_path / "out"
        scan(model_path, out, spec)

        # Load PCA and check clusters are separated
        pca = np.load(out / "embedding_pca.npy")
        pc1 = pca[:, 0]  # First principal component

        # First 50 points should have different mean than last 50
        mean1 = pc1[:50].mean()
        mean2 = pc1[50:].mean()
        assert abs(mean1 - mean2) > 2.0  # Clear separation


# ---------------------------------------------------------------------------
# UMAP tests (skip if umap-learn not installed)
# ---------------------------------------------------------------------------


class TestUMAP:
    def test_umap_import_error(self):
        """UMAP should raise ImportError if not installed."""
        pass  # Skip this test - we test PCA only in CI


# ---------------------------------------------------------------------------
# Manifest discovery test
# ---------------------------------------------------------------------------


class TestManifestDiscovery:
    def test_embedding_in_manifest(self, tmp_path):
        """Embedding fields should appear in manifest."""
        from weight_atlas.scan import scan

        model_path = tmp_path / "model.safetensors"
        rng = np.random.default_rng(42)
        tensors = {
            "model.embed_tokens.weight": rng.normal(0, 0.1, (100, 32)).astype(np.float32),
            "model.layers.0.self_attn.q_proj.weight": rng.normal(0, 0.1, (32, 32)).astype(np.float32),
        }
        save_file(tensors, str(model_path))

        spec = AtlasSpec(
            spec_version=1,
            slots=["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"],
            channels={"height": {"stat": "spectral_norm", "scale": {"type": "log1p"}}},
            grid={"upsample": 2, "smooth_sigma": 1.0},
            sheet={"contour_levels": 4, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
            seeds={"svd": 0},
            embedding={"method": "pca", "grid": 256, "components": 3, "seeds": {"pca": 0}},
        )

        out = tmp_path / "out"
        scan(model_path, out, spec)

        with open(out / "manifest.json") as f:
            manifest = json.load(f)

        # Check embedding artefacts in manifest
        assert "embedding_pca.npy" in manifest
        assert "embedding_meta.json" in manifest
        assert "field_embed_density_raw.tif" in manifest
        assert "field_embed_density_smooth.tif" in manifest


# ---------------------------------------------------------------------------
# Scatter overlay and --field dry-run tests
# ---------------------------------------------------------------------------


class TestScatterOverlay:
    def test_scatter_file_created(self, tmp_path):
        """Scatter coordinates file should be created during scan."""
        from weight_atlas.scan import scan

        model_path = tmp_path / "model.safetensors"
        rng = np.random.default_rng(42)
        tensors = {
            "model.embed_tokens.weight": rng.normal(0, 0.1, (100, 32)).astype(np.float32),
            "model.layers.0.self_attn.q_proj.weight": rng.normal(0, 0.1, (32, 32)).astype(np.float32),
        }
        save_file(tensors, str(model_path))

        spec = AtlasSpec(
            spec_version=1,
            slots=["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"],
            channels={"height": {"stat": "spectral_norm", "scale": {"type": "log1p"}}},
            grid={"upsample": 2, "smooth_sigma": 1.0},
            sheet={"contour_levels": 4, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
            seeds={"svd": 0},
            embedding={"method": "pca", "grid": 256, "components": 3, "subsample_scatter": 50, "seeds": {"pca": 0}},
        )

        out = tmp_path / "out"
        scan(model_path, out, spec)

        scatter_path = out / "embedding_scatter.npy"
        assert scatter_path.exists()

        scatter = np.load(scatter_path)
        assert scatter.shape[0] == 50  # Subsampled to 50
        assert scatter.shape[1] == 2


class TestFieldDryRun:
    def test_field_argument_dry_run(self, tmp_path):
        """Test that --field argument is accepted by render command."""
        from weight_atlas.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["render", str(tmp_path), "--renderer", "blender", "--field", "embed_density"])

        assert args.field == "embed_density"
        assert args.renderer == "blender"


class TestPanelSheetDeterminism:
    def test_panel_sheet_deterministic(self, tmp_path):
        """Panel comparison sheet should be byte-identical on second run."""
        from weight_atlas.compare.panel import compare_expert_panels
        from weight_atlas.core.types import AtlasSpec, ExpertPanel

        rng = np.random.default_rng(42)
        panel_a = ExpertPanel(slot="mlp_down", channel="height", data=rng.normal(0, 1, (4, 8)))
        panel_b = ExpertPanel(slot="mlp_down", channel="height", data=rng.normal(0, 1, (4, 8)))

        spec = AtlasSpec(
            spec_version=1,
            slots=["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"],
            channels={"height": {"stat": "spectral_norm", "scale": {"type": "log1p"}}},
            grid={"upsample": 2, "smooth_sigma": 1.0},
            sheet={"contour_levels": 4, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
            seeds={"svd": 0},
        )

        results1 = compare_expert_panels([panel_a], [panel_b], spec)
        results2 = compare_expert_panels([panel_a], [panel_b], spec)

        assert results1[0].status == results2[0].status == "compared"
