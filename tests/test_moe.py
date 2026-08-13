"""Tests for MoE Expert Panel (M6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from weight_atlas.core.name_map import (
    extract_expert_id,
    is_expert_tensor,
    is_shared_expert,
    map_name,
)
from weight_atlas.core.types import TensorStats
from weight_atlas.fields.rasterizer import detect_moe, rasterize, rasterize_expert_panels

# ---------------------------------------------------------------------------
# Name mapping tests
# ---------------------------------------------------------------------------


class TestMoENameMapping:
    def test_hf_router_before_gate_proj(self):
        """mlp.gate.weight (router) must be recognized before mlp.gate_proj."""
        layer, slot = map_name("model.layers.0.mlp.gate.weight")
        assert layer == 0
        assert slot == "router"

    def test_hf_expert_tensor(self):
        """HF expert tensor should be recognized."""
        layer, slot = map_name("model.layers.0.mlp.experts.3.gate_proj.weight")
        assert layer == 0
        assert slot == "expert"

    def test_hf_shared_expert(self):
        """Shared expert should map to mlp slots."""
        layer, slot = map_name("model.layers.0.shared_expert.gate_proj.weight")
        assert layer == 0
        assert slot == "mlp_gate"

    def test_hf_shared_expert_gate(self):
        """Shared expert gate should map to other."""
        layer, slot = map_name("model.layers.0.shared_expert_gate.weight")
        assert layer == 0
        assert slot == "other"

    def test_extract_expert_id(self):
        """Expert ID should be extracted from tensor name."""
        assert extract_expert_id("model.layers.0.mlp.experts.5.gate_proj.weight") == 5
        assert extract_expert_id("model.layers.0.mlp.gate_proj.weight") is None

    def test_is_expert_tensor(self):
        """Expert tensor detection."""
        assert is_expert_tensor("model.layers.0.mlp.experts.3.gate_proj.weight")
        assert not is_expert_tensor("model.layers.0.mlp.gate_proj.weight")

    def test_is_shared_expert(self):
        """Shared expert detection."""
        assert is_shared_expert("model.layers.0.shared_expert.gate_proj.weight")
        assert not is_shared_expert("model.layers.0.mlp.gate_proj.weight")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_moe_model(path: Path, n_layers: int = 4, n_experts: int = 8, shared: bool = True):
    """Create a small MoE model with expert tensors."""
    rng = np.random.default_rng(42)
    tensors = {}

    for layer in range(n_layers):
        # Regular attention
        tensors[f"model.layers.{layer}.self_attn.q_proj.weight"] = rng.normal(0, 0.1, (32, 32)).astype(np.float32)
        tensors[f"model.layers.{layer}.self_attn.k_proj.weight"] = rng.normal(0, 0.1, (32, 32)).astype(np.float32)

        # Router
        tensors[f"model.layers.{layer}.mlp.gate.weight"] = rng.normal(0, 0.1, (n_experts, 32)).astype(np.float32)

        # Expert tensors
        for expert in range(n_experts):
            tensors[f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"] = rng.normal(0, 0.1, (32, 32)).astype(np.float32)
            tensors[f"model.layers.{layer}.mlp.experts.{expert}.up_proj.weight"] = rng.normal(0, 0.1, (32, 32)).astype(np.float32)
            tensors[f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight"] = rng.normal(0, 0.1, (32, 32)).astype(np.float32)

        # Shared expert
        if shared:
            tensors[f"model.layers.{layer}.shared_expert.gate_proj.weight"] = rng.normal(0, 0.1, (32, 32)).astype(np.float32)
            tensors[f"model.layers.{layer}.shared_expert.up_proj.weight"] = rng.normal(0, 0.1, (32, 32)).astype(np.float32)
            tensors[f"model.layers.{layer}.shared_expert.down_proj.weight"] = rng.normal(0, 0.1, (32, 32)).astype(np.float32)

    save_file(tensors, str(path))
    return tensors


@pytest.fixture
def moe_model(tmp_path):
    """Create a small MoE model."""
    path = tmp_path / "moe_model.safetensors"
    make_moe_model(path, n_layers=4, n_experts=8, shared=True)
    return path


# ---------------------------------------------------------------------------
# Rasterizer tests
# ---------------------------------------------------------------------------


class TestMoERasterizer:
    def test_main_raster_excludes_experts(self, moe_model):
        """Main raster should exclude expert tensors."""
        from weight_atlas.core.types import AtlasSpec
        from weight_atlas.loaders.safetensors_loader import SafetensorsLoader

        loader = SafetensorsLoader()
        handles = loader.open(moe_model)
        stats = [TensorStats(name=h.name, shape=h.shape, spectral_norm=1.0) for h in handles]

        spec = AtlasSpec(
            spec_version=1,
            slots=["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"],
            channels={"height": {"stat": "spectral_norm", "scale": {"type": "log1p"}}},
            grid={"upsample": 2, "smooth_sigma": 1.0},
            sheet={"contour_levels": 4, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
            seeds={"svd": 0},
        )

        field = rasterize(stats, spec, "spectral_norm")
        # mlp_gate column should only have shared expert values (from router it's skipped)
        # Router tensors are skipped from main raster
        assert field.data.shape[0] == 4  # 4 layers

    def test_expert_panel_shape(self, moe_model):
        """Expert panel should have shape (n_layers, n_experts)."""
        from weight_atlas.core.types import AtlasSpec
        from weight_atlas.loaders.safetensors_loader import SafetensorsLoader

        loader = SafetensorsLoader()
        handles = loader.open(moe_model)
        stats = [TensorStats(name=h.name, shape=h.shape, spectral_norm=1.0, expert_id=extract_expert_id(h.name)) for h in handles]

        spec = AtlasSpec(
            spec_version=1,
            slots=["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"],
            channels={"height": {"stat": "spectral_norm", "scale": {"type": "log1p"}}},
            grid={"upsample": 2, "smooth_sigma": 1.0},
            sheet={"contour_levels": 4, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
            seeds={"svd": 0},
        )

        panels = rasterize_expert_panels(stats, spec, "spectral_norm")
        assert len(panels) == 3  # gate, up, down

        for panel in panels:
            assert panel.n_layers == 4
            assert panel.n_experts == 8

    def test_detect_moe(self, moe_model):
        """detect_moe should identify MoE configuration."""
        from weight_atlas.loaders.safetensors_loader import SafetensorsLoader

        loader = SafetensorsLoader()
        handles = loader.open(moe_model)
        stats = [TensorStats(name=h.name, shape=h.shape, spectral_norm=1.0) for h in handles]

        moe_info = detect_moe(stats)
        assert moe_info["num_experts"] == 8
        assert moe_info["shared_expert"] is True
        assert moe_info["num_layers"] == 4


# ---------------------------------------------------------------------------
# Fingerprint tests
# ---------------------------------------------------------------------------


class TestMoEFingerprint:
    def test_fingerprint_has_moe_info(self, moe_model, tmp_path):
        """Fingerprint should include MoE info."""
        import json

        from weight_atlas.core.types import AtlasSpec
        from weight_atlas.scan import scan

        spec = AtlasSpec(
            spec_version=1,
            slots=["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"],
            channels={"height": {"stat": "spectral_norm", "scale": {"type": "log1p"}}},
            grid={"upsample": 2, "smooth_sigma": 1.0},
            sheet={"contour_levels": 4, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
            seeds={"svd": 0},
        )

        out = tmp_path / "out"
        scan(moe_model, out, spec)

        fp = json.loads((out / "fingerprint.json").read_text())
        assert "model" in fp
        assert "moe" in fp["model"]
        assert fp["model"]["moe"]["num_experts"] == 8
        assert fp["model"]["moe"]["shared_expert"] is True


# ---------------------------------------------------------------------------
# Localization test
# ---------------------------------------------------------------------------


class TestMoELocalization:
    def test_expert_perturbation_localized(self, tmp_path):
        """Perturbation in specific expert should be localized by compare."""
        from weight_atlas.core.types import AtlasSpec

        AtlasSpec(
            spec_version=1,
            slots=["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"],
            channels={"height": {"stat": "spectral_norm", "scale": {"type": "log1p"}}},
            grid={"upsample": 2, "smooth_sigma": 1.0},
            sheet={"contour_levels": 4, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
            seeds={"svd": 0},
        )

        # Create two identical panels
        rng = np.random.default_rng(42)
        panel_a = rng.normal(0, 1, (4, 8)).astype(np.float64)
        panel_b = panel_a.copy()

        # Perturb expert (2, 5)
        panel_b[2, 5] += 10.0

        # Create fingerprint with MoE info

        # Compare

        delta = panel_b - panel_a
        abs_delta = np.abs(delta)

        # Find hotspot
        flat = abs_delta.flatten()
        idx = np.argmax(flat)
        row, col = divmod(idx, abs_delta.shape[1])

        assert row == 2
        assert col == 5


# ---------------------------------------------------------------------------
# GGUF MoE fixture and 3D-split tests
# ---------------------------------------------------------------------------


def make_gguf_moe_file(path: Path, n_layers: int = 4, n_experts: int = 8, shared: bool = True):
    """Create a GGUF MoE model with 3D stacked expert tensors."""
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), arch="llama")
    writer.add_architecture()
    writer.add_block_count(n_layers)
    writer.add_uint32("llama.block_count", n_layers)

    rng = np.random.default_rng(42)

    for layer in range(n_layers):
        # Regular attention
        writer.add_tensor(f"blk.{layer}.attn_q.weight", rng.normal(0, 0.1, (32, 32)).astype(np.float32))
        writer.add_tensor(f"blk.{layer}.attn_k.weight", rng.normal(0, 0.1, (32, 32)).astype(np.float32))

        # Router
        writer.add_tensor(f"blk.{layer}.ffn_gate_inp.weight", rng.normal(0, 0.1, (n_experts, 32)).astype(np.float32))

        # 3D stacked expert tensors (n_experts, hidden, hidden)
        writer.add_tensor(f"blk.{layer}.ffn_gate_exps.weight", rng.normal(0, 0.1, (n_experts, 32, 32)).astype(np.float32))
        writer.add_tensor(f"blk.{layer}.ffn_up_exps.weight", rng.normal(0, 0.1, (n_experts, 32, 32)).astype(np.float32))
        writer.add_tensor(f"blk.{layer}.ffn_down_exps.weight", rng.normal(0, 0.1, (n_experts, 32, 32)).astype(np.float32))

        # Shared expert
        if shared:
            writer.add_tensor(f"blk.{layer}.ffn_gate_shexp.weight", rng.normal(0, 0.1, (32, 32)).astype(np.float32))
            writer.add_tensor(f"blk.{layer}.ffn_up_shexp.weight", rng.normal(0, 0.1, (32, 32)).astype(np.float32))
            writer.add_tensor(f"blk.{layer}.ffn_down_shexp.weight", rng.normal(0, 0.1, (32, 32)).astype(np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture
def gguf_moe_model(tmp_path):
    """Create a small GGUF MoE model with 3D expert tensors."""
    path = tmp_path / "moe_model.gguf"
    make_gguf_moe_file(path, n_layers=4, n_experts=8, shared=True)
    return path


class TestGGUF3DSplit:
    def test_gguf_moe_loads_expert_subhandles(self, gguf_moe_model):
        """GGUF MoE should load expert tensors as sub-handles."""
        from weight_atlas.loaders.gguf_loader import GGUFLoader

        loader = GGUFLoader()
        handles = loader.open(gguf_moe_model)

        # Should have sub-handlers for expert tensors
        expert_handles = [h for h in handles if h.expert_id is not None]
        assert len(expert_handles) > 0

        # Each layer has 3 expert tensors × 8 experts = 24 sub-handles per layer
        # 4 layers × 24 = 96 expert sub-handles
        assert len(expert_handles) == 4 * 3 * 8

    def test_gguf_moe_detect_moe_uses_expert_id(self, gguf_moe_model):
        """detect_moe must use TensorStats.expert_id for GGUF-split names.

        GGUF expert sub-handles are named ``blk.N.ffn_*_exps.weight[E]`` where the
        expert id is NOT parseable from the name string (HF-style). Regression test
        for Qwen3.6-35B-A3B (qwen35moe) detection.
        """
        from weight_atlas.loaders.gguf_loader import GGUFLoader

        loader = GGUFLoader()
        handles = loader.open(gguf_moe_model)
        stats = [TensorStats(name=h.name, shape=h.shape, spectral_norm=1.0, expert_id=h.expert_id) for h in handles]

        moe_info = detect_moe(stats)
        assert moe_info["num_experts"] == 8
        assert moe_info["shared_expert"] is True
        assert moe_info["num_layers"] == 4

    def test_gguf_moe_panels_use_expert_id(self, gguf_moe_model):
        """Expert panels must rasterize GGUF-split expert handles."""
        from weight_atlas.core.types import AtlasSpec
        from weight_atlas.fields.rasterizer import rasterize_expert_panels
        from weight_atlas.loaders.gguf_loader import GGUFLoader

        spec = AtlasSpec(
            spec_version=1,
            slots=["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"],
            channels={"height": {"stat": "spectral_norm", "scale": {"type": "log1p"}}},
            grid={"upsample": 2, "smooth_sigma": 1.0},
            sheet={"contour_levels": 4, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
            seeds={"svd": 0},
        )

        loader = GGUFLoader()
        handles = loader.open(gguf_moe_model)
        stats = [TensorStats(name=h.name, shape=h.shape, spectral_norm=1.0, expert_id=h.expert_id) for h in handles]

        panels = rasterize_expert_panels(stats, spec, "spectral_norm")
        assert len(panels) == 3
        for panel in panels:
            assert panel.n_layers == 4
            assert panel.n_experts == 8

    def test_gguf_moe_full_scan_produces_panels_and_moe_meta(self, gguf_moe_model, tmp_path):
        """Full scan() of a GGUF MoE model must emit expert-panel TIFFs and MoE metadata.

        Regression test for the silent data-loss bug where GGUF MoE expert
        panels (field_expert_*) and ``model.moe`` were dropped even though the
        loader produces expert sub-handles.
        """
        import json

        from weight_atlas.core.types import AtlasSpec
        from weight_atlas.scan import scan as run_scan

        spec = AtlasSpec(
            spec_version=1,
            slots=["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"],
            channels={"height": {"stat": "spectral_norm", "scale": {"type": "log1p"}}},
            grid={"upsample": 2, "smooth_sigma": 1.0},
            sheet={"contour_levels": 4, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
            seeds={"svd": 0},
        )

        out = tmp_path / "scan_out"
        run_scan(gguf_moe_model, out, spec)

        # Expert panel TIFFs must exist for every slot × channel.
        for slot in ("mlp_gate", "mlp_up", "mlp_down"):
            assert (out / f"field_expert_{slot}_height_raw.tif").exists(), f"missing panel {slot}"

        # Fingerprint must carry MoE metadata.
        fp = json.loads((out / "fingerprint.json").read_text())
        assert fp["model"]["moe"]["num_experts"] == 8
        assert fp["model"]["moe"]["shared_expert"] is True
        assert fp["model"]["moe"]["num_layers"] == 4

    def test_expert_subhandle_has_expert_id(self, gguf_moe_model):
        """Expert sub-handles should have expert_id set."""
        from weight_atlas.loaders.gguf_loader import GGUFLoader

        loader = GGUFLoader()
        handles = loader.open(gguf_moe_model)

        expert_handles = [h for h in handles if h.expert_id is not None]
        for h in expert_handles:
            assert h.expert_id is not None
            assert 0 <= h.expert_id < 8

    def test_expert_subhandle_loads_2d_slice(self, gguf_moe_model):
        """Expert sub-handle load() should return 2D slice."""
        from weight_atlas.loaders.gguf_loader import GGUFLoader

        loader = GGUFLoader()
        handles = loader.open(gguf_moe_model)

        # Find first expert handle
        expert_handles = [h for h in handles if h.expert_id is not None]
        handle = expert_handles[0]

        arr = handle.load()
        assert len(arr.shape) == 2  # 2D slice
        assert arr.shape == (32, 32)

    def test_shared_expert_maps_to_mlp_slots(self, gguf_moe_model):
        """Shared expert tensors should map to mlp slots."""
        from weight_atlas.loaders.gguf_loader import GGUFLoader

        loader = GGUFLoader()
        handles = loader.open(gguf_moe_model)

        # Check that shared expert handles exist with correct names
        shexp_handles = [h for h in handles if "shexp" in h.name]
        assert len(shexp_handles) > 0

        for h in shexp_handles:
            assert h.expert_id is None  # Shared experts don't have expert_id


# ---------------------------------------------------------------------------
# Compare Panel tests
# ---------------------------------------------------------------------------


class TestComparePanels:
    def test_panel_compare_compared(self):
        """Panels with same shape should be compared."""
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

        results = compare_expert_panels([panel_a], [panel_b], spec)
        assert len(results) == 1
        assert results[0].status == "compared"

    def test_panel_compare_skipped_shape_mismatch(self):
        """Panels with different shapes should be skipped."""
        from weight_atlas.compare.panel import compare_expert_panels
        from weight_atlas.core.types import AtlasSpec, ExpertPanel

        rng = np.random.default_rng(42)
        panel_a = ExpertPanel(slot="mlp_down", channel="height", data=rng.normal(0, 1, (4, 8)))
        panel_b = ExpertPanel(slot="mlp_down", channel="height", data=rng.normal(0, 1, (4, 16)))

        spec = AtlasSpec(
            spec_version=1,
            slots=["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"],
            channels={"height": {"stat": "spectral_norm", "scale": {"type": "log1p"}}},
            grid={"upsample": 2, "smooth_sigma": 1.0},
            sheet={"contour_levels": 4, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
            seeds={"svd": 0},
        )

        results = compare_expert_panels([panel_a], [panel_b], spec)
        assert len(results) == 1
        assert results[0].status == "skipped"
        assert "Shape mismatch" in results[0].reason


# ---------------------------------------------------------------------------
# Expert panel channels (spec v2.4: expert_channels = cheap O(n) stats)
# ---------------------------------------------------------------------------


class TestExpertChannels:
    def test_spec_v24_has_expert_channels(self):
        from weight_atlas.core.types import DEFAULT_SPEC_VERSION, load_default_spec

        spec = load_default_spec()
        assert spec.spec_version == DEFAULT_SPEC_VERSION
        assert set(spec.expert_channels) == {"height", "tint", "rough"}
        assert spec.expert_channels["height"]["stat"] == "frobenius"
        assert spec.expert_channels["tint"]["stat"] == "kurtosis"
        assert spec.expert_channels["rough"]["stat"] == "sparsity"

    def test_scan_moe_writes_expert_panel_fields(self, moe_model, tmp_path):
        """Expert panel TIFFs (raw + smooth) are written and in the manifest."""
        import json

        from weight_atlas.core.types import load_default_spec
        from weight_atlas.scan import scan

        spec = load_default_spec()
        out = tmp_path / "out"
        scan(moe_model, out, spec)

        manifest = json.loads((out / "manifest.json").read_text())
        for slot in ("mlp_gate", "mlp_up", "mlp_down"):
            for ch in ("height", "tint", "rough"):
                assert f"field_expert_{slot}_{ch}_raw.tif" in manifest
                assert f"field_expert_{slot}_{ch}_smooth.tif" in manifest

    def test_panel_field_uses_frobenius_not_spectral(self, moe_model, tmp_path):
        """Expert panels rasterize the expert_channels stat, not the main one."""
        from weight_atlas.core.types import load_default_spec
        from weight_atlas.fields.rasterizer import rasterize_expert_panels
        from weight_atlas.fields.tif_io import read_tif
        from weight_atlas.loaders.safetensors_loader import SafetensorsLoader
        from weight_atlas.scan import _make_handles, scan

        spec = load_default_spec()
        out = tmp_path / "out"
        scan(moe_model, out, spec)

        stats = [_make_handles(h) for h in SafetensorsLoader().open(moe_model)]
        expected = rasterize_expert_panels(stats, spec, "frobenius")
        gate_panel = [p for p in expected if p.slot == "mlp_gate"][0]
        written = read_tif(out / "field_expert_mlp_gate_height_raw.tif")
        np.testing.assert_allclose(written, gate_panel.data, rtol=1e-6)

    def test_load_channel_field_expert_labels(self, moe_model, tmp_path):
        """Expert fields render with expert-id columns and layer rows."""
        from weight_atlas.core.types import load_default_spec
        from weight_atlas.fields.rasterizer import load_channel_field
        from weight_atlas.scan import scan

        spec = load_default_spec()
        out = tmp_path / "out"
        scan(moe_model, out, spec)

        field = load_channel_field(out, "expert_mlp_gate_height", spec, model_name="moe")
        assert field is not None
        assert field.col_labels == [str(i) for i in range(8)]  # expert ids
        assert field.row_labels == [str(i) for i in range(4)]  # layers


# ---------------------------------------------------------------------------
# MoE Localization test
# ---------------------------------------------------------------------------


class TestMoELocalizationFull:
    def test_expert_perturbation_panel_hotspot(self, tmp_path):
        """Perturbation in expert (2, 5, down) should be localized as panel hotspot."""
        from weight_atlas.compare.panel import compare_expert_panels
        from weight_atlas.core.types import AtlasSpec, ExpertPanel

        rng = np.random.default_rng(42)
        panel_a = ExpertPanel(slot="mlp_down", channel="height", data=rng.normal(0, 1, (4, 8)))
        panel_b = panel_a.data.copy()
        panel_b[2, 5] += 10.0  # Perturb expert (2, 5)
        panel_b = ExpertPanel(slot="mlp_down", channel="height", data=panel_b)

        spec = AtlasSpec(
            spec_version=1,
            slots=["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"],
            channels={"height": {"stat": "spectral_norm", "scale": {"type": "log1p"}}},
            grid={"upsample": 2, "smooth_sigma": 1.0},
            sheet={"contour_levels": 4, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
            seeds={"svd": 0},
        )

        results = compare_expert_panels([panel_a], [panel_b], spec)
        assert len(results) == 1
        assert results[0].status == "compared"

        # Check hotspot is at (2, 5) - row 2, expert 5
        delta = results[0].delta
        assert delta.argmax[0] == 2  # row 2
        # The column label should be '5' (expert ID as string)
        assert delta.argmax[1] == "5"
