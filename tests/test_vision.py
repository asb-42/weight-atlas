"""VLM vision-tower support: name mapping, kernel norm, vision raster, scan e2e."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from weight_atlas.core.name_map import (
    is_vision_tensor,
    map_name,
    map_vision,
)
from weight_atlas.core.types import (
    TensorHandle,
    TensorStats,
    load_default_spec,
)
from weight_atlas.fields.rasterizer import (
    detect_vision,
    load_channel_field,
    rasterize_vision,
)
from weight_atlas.stats.norms import KernelNorm


def _handle(arr: np.ndarray) -> TensorHandle:
    return TensorHandle(name="t", shape=arr.shape, dtype="F32", loader=lambda: arr)


# ---------------------------------------------------------------------------
# Name mapping
# ---------------------------------------------------------------------------


class TestMapVision:
    def test_gguf_v_blk_slots(self):
        assert map_vision("v.blk.3.attn_q.weight") == (3, "v_attn_q")
        assert map_vision("v.blk.3.attn_k.weight") == (3, "v_attn_k")
        assert map_vision("v.blk.3.attn_v.weight") == (3, "v_attn_v")
        assert map_vision("v.blk.3.attn_out.weight") == (3, "v_attn_o")
        assert map_vision("v.blk.3.ln1.weight") == (3, "v_attn_norm")
        assert map_vision("v.blk.3.ln2.weight") == (3, "v_mlp_norm")
        assert map_vision("v.blk.3.mlp.ffn_gate.weight") == (3, "v_mlp_gate")
        assert map_vision("v.blk.3.mlp.ffn_up.weight") == (3, "v_mlp_up")
        assert map_vision("v.blk.3.mlp.ffn_down.weight") == (3, "v_mlp_down")
        assert map_vision("v.blk.3.something.else.weight") == (3, "v_other")

    def test_gguf_global_tensors(self):
        assert map_vision("v.patch_embed.weight") == (None, "v_patch_embed")
        assert map_vision("v.pos_embed") == (None, "v_pos_emb")
        assert map_vision("v.cls") == (None, "v_cls")

    def test_projector(self):
        assert map_vision("mm.model.mlp.0.weight") == (None, "mm_projector")
        assert map_vision("mm.2.weight") == (None, "mm_projector")
        assert map_vision("mm_projector.proj.0.weight") == (None, "mm_projector")
        assert map_vision("model.multi_modal_projector.layers.0.linear_1.weight") == (None, "mm_projector")

    def test_hf_vision_model(self):
        assert map_vision("model.vision_model.encoder.layers.5.self_attn.q_proj.weight") == (5, "v_attn_q")
        assert map_vision("model.vision_model.encoder.layers.5.self_attn.o_proj.weight") == (5, "v_attn_o")
        assert map_vision("model.vision_model.encoder.layers.5.layer_norm1.weight") == (5, "v_attn_norm")
        assert map_vision("model.vision_model.encoder.layers.5.mlp.fc1.weight") == (5, "v_mlp")
        assert map_vision("model.vision_model.embeddings.patch_embedding.proj.weight") == (None, "v_patch_embed")
        assert map_vision("model.vision_model.embeddings.position_embedding.weight") == (None, "v_pos_emb")
        assert map_vision("model.vision_model.post_layernorm.weight") == (None, "v_mlp_norm")

    def test_hf_qwen2_visual(self):
        assert map_vision("model.visual.blocks.2.attn.qkv.weight") == (2, "v_attn_qkv")
        assert map_vision("model.visual.blocks.2.attn.proj.weight") == (2, "v_attn_o")
        assert map_vision("model.visual.blocks.2.norm1.weight") == (2, "v_attn_norm")
        assert map_vision("model.visual.blocks.2.mlp.fc2.weight") == (2, "v_mlp")
        assert map_vision("model.visual.patch_embed.proj.weight") == (None, "v_patch_embed")
        assert map_vision("model.visual.merger.0.weight") == (None, "mm_projector")

    def test_kimi_vision_tower(self):
        assert map_vision("vision_tower.encoder.blocks.1.wqkv.weight") == (1, "v_attn_qkv")
        assert map_vision("vision_tower.encoder.blocks.1.wo.weight") == (1, "v_attn_o")
        assert map_vision("vision_tower.encoder.blocks.1.norm0.weight") == (1, "v_attn_norm")
        assert map_vision("vision_tower.encoder.blocks.1.mlp.fc0.weight") == (1, "v_mlp")
        assert map_vision("vision_tower.encoder.final_layernorm.weight") == (None, "v_mlp_norm")
        assert map_vision("vision_tower.patch_embed.pos_emb.weight") == (None, "v_pos_emb")

    def test_text_tensors_are_not_vision(self):
        assert map_vision("model.layers.0.self_attn.q_proj.weight") is None
        assert map_vision("blk.0.attn_q.weight") is None
        assert map_vision("token_embd.weight") is None
        assert map_vision("model.embed_tokens.weight") is None

    def test_is_vision_tensor(self):
        assert is_vision_tensor("v.blk.0.attn_q.weight")
        assert is_vision_tensor("mm.model.mlp.0.weight")
        assert not is_vision_tensor("blk.0.attn_q.weight")
        assert not is_vision_tensor("model.layers.0.self_attn.q_proj.weight")

    def test_map_name_delegates(self):
        """map_name keeps vision tensors out of the transformer raster."""
        assert map_name("v.blk.3.attn_q.weight") == (None, "v_attn_q")
        assert map_name("mm.model.mlp.0.weight") == (None, "mm_projector")
        assert map_name("blk.3.attn_q.weight") == (3, "attn_q")
        assert map_name("model.layers.0.self_attn.q_proj.weight") == (0, "attn_q")


# ---------------------------------------------------------------------------
# KernelNorm statistic
# ---------------------------------------------------------------------------


class TestKernelNorm:
    def test_conv_kernel_per_channel_mean(self):
        x = np.array(
            [
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 2.0], [3.0, 4.0]]],
            ],
            dtype=np.float32,
        )
        expected = (1.0 + np.sqrt(30.0)) / 2.0
        assert KernelNorm().compute(_handle(x)) == pytest.approx(expected)

    def test_2d_falls_back_to_frobenius(self):
        x = np.array([[3.0, 4.0]], dtype=np.float32)
        assert KernelNorm().compute(_handle(x)) == pytest.approx(5.0)

    def test_1d_frobenius(self):
        x = np.array([3.0, 4.0], dtype=np.float32)
        assert KernelNorm().compute(_handle(x)) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Vision rasterization
# ---------------------------------------------------------------------------


def _vision_stats() -> list[TensorStats]:
    return [
        TensorStats(name="v.blk.0.attn_q.weight", shape=(4, 4), kernel_norm=1.0),
        TensorStats(name="v.blk.0.attn_out.weight", shape=(4, 4), kernel_norm=2.0),
        TensorStats(name="v.blk.2.attn_q.weight", shape=(4, 4), kernel_norm=3.0),
        TensorStats(name="v.patch_embed.weight", shape=(4, 1, 3, 3), kernel_norm=4.0),
        TensorStats(name="mm.model.mlp.0.weight", shape=(4, 4), kernel_norm=5.0),
        TensorStats(name="mm.model.mlp.1.weight", shape=(4, 4), kernel_norm=9.0),
        # Text tensor — must be excluded from the vision raster.
        TensorStats(name="model.layers.0.self_attn.q_proj.weight", shape=(4, 4), kernel_norm=99.0),
    ]


class TestRasterizeVision:
    @pytest.fixture
    def spec(self):
        return load_default_spec()

    def test_field_shape_and_labels(self, spec):
        field = rasterize_vision(_vision_stats(), spec, "kernel_norm")
        assert field is not None
        # blocks 0 and 2 + one "global" row
        assert field.data.shape == (3, len(spec.vision_slots))
        assert field.row_labels == ["0", "2", "global"]
        assert field.col_labels == list(spec.vision_slots)

    def test_cell_placement(self, spec):
        field = rasterize_vision(_vision_stats(), spec, "kernel_norm")
        q_col = spec.vision_slots.index("v_attn_q")
        o_col = spec.vision_slots.index("v_attn_o")
        mp_col = spec.vision_slots.index("mm_projector")
        pe_col = spec.vision_slots.index("v_patch_embed")
        assert field.data[0, q_col] == 1.0
        assert field.data[0, o_col] == 2.0
        assert field.data[1, q_col] == 3.0
        assert np.isnan(field.data[1, o_col])  # block 2 has no attn_o
        # global row: projector tensors mean-aggregated, patch embed placed
        assert field.data[2, mp_col] == pytest.approx(7.0)
        assert field.data[2, pe_col] == 4.0

    def test_text_only_model_returns_none(self, spec):
        stats = [
            TensorStats(name="model.layers.0.self_attn.q_proj.weight", shape=(4, 4), kernel_norm=1.0),
            TensorStats(name="model.embed_tokens.weight", shape=(4, 4), kernel_norm=2.0),
        ]
        assert rasterize_vision(stats, spec, "kernel_norm") is None

    def test_detect_vision(self, spec):
        info = detect_vision(_vision_stats())
        assert info == {"present": True, "n_tensors": 6, "n_blocks": 2, "n_global": 3}

    def test_detect_vision_text_only(self, spec):
        stats = [TensorStats(name="model.layers.0.self_attn.q_proj.weight", shape=(4, 4))]
        assert detect_vision(stats) is None


# ---------------------------------------------------------------------------
# End-to-end scan of a VLM model
# ---------------------------------------------------------------------------


def make_vlm_model(path: Path, n_layers: int = 2, n_vision: int = 2, hidden: int = 16):
    """Write a fake safetensors VLM model: text tower + vision tower + projector."""
    rng = np.random.default_rng(7)
    tensors = {}
    for layer in range(n_layers):
        for slot in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            "input_layernorm", "post_attention_layernorm",
        ):
            tensors[f"model.layers.{layer}.{slot}.weight"] = rng.normal(0, 0.1, (hidden, hidden)).astype(np.float32)
    tensors["model.embed_tokens.weight"] = rng.normal(0, 0.1, (hidden, hidden)).astype(np.float32)
    tensors["lm_head.weight"] = rng.normal(0, 0.1, (hidden, hidden)).astype(np.float32)
    for b in range(n_vision):
        for slot in ("attn_q", "attn_k", "attn_v", "attn_out"):
            tensors[f"v.blk.{b}.{slot}.weight"] = rng.normal(0, 0.2, (hidden, hidden)).astype(np.float32)
        tensors[f"v.blk.{b}.ln1.weight"] = rng.normal(0, 0.2, (hidden,)).astype(np.float32)
        tensors[f"v.blk.{b}.ln2.weight"] = rng.normal(0, 0.2, (hidden,)).astype(np.float32)
        for slot in ("ffn_gate", "ffn_up", "ffn_down"):
            tensors[f"v.blk.{b}.mlp.{slot}.weight"] = rng.normal(0, 0.2, (hidden, hidden)).astype(np.float32)
    # Conv patch embedding (C_out, C_in, 3, 3) and multimodal projector.
    tensors["v.patch_embed.weight"] = rng.normal(0, 0.1, (hidden, 3, 3, 3)).astype(np.float32)
    for i in range(3):
        tensors[f"mm.model.mlp.{i}.weight"] = rng.normal(0, 0.1, (hidden, hidden)).astype(np.float32)
    save_file(tensors, str(path))
    return tensors


@pytest.fixture
def vlm_model(tmp_path):
    path = tmp_path / "vlm.safetensors"
    make_vlm_model(path)
    return path


class TestScanVLM:
    def test_scan_produces_vision_artefacts(self, vlm_model, tmp_path):
        from weight_atlas.scan import scan

        spec = load_default_spec()
        out = tmp_path / "out"
        scan(vlm_model, out, spec)

        fp = json.loads((out / "fingerprint.json").read_text())
        vision = fp["model"]["vision"]
        assert vision["present"] is True
        assert vision["n_blocks"] == 2
        assert vision["n_global"] == 4  # patch_embed + 3 projector tensors
        assert vision["n_tensors"] == 2 * 9 + 4

        # Vision tensors must count as mapped → coverage is complete.
        assert fp["mapping_coverage"]["vision_tensors"] == vision["n_tensors"]
        assert fp["mapping_coverage"]["in_slots"] == pytest.approx(1.0)

        # Vision fields written (raw + smooth) and in the manifest.
        from weight_atlas.fields.tif_io import read_tif

        raw = read_tif(out / "field_vision_height_raw.tif")
        assert raw.shape == (3, len(spec.vision_slots))  # blocks + global row
        smooth = read_tif(out / "field_vision_height_smooth.tif")
        assert smooth.shape == (3 * 8, len(spec.vision_slots) * 8)
        manifest = json.loads((out / "manifest.json").read_text())
        assert "field_vision_height_raw.tif" in manifest
        assert "field_vision_height_smooth.tif" in manifest

    def test_scan_vlm_renders_vision_sheet(self, vlm_model, tmp_path):
        """The vision sheet renders and carries the vision slot labels."""
        from weight_atlas.cli import main

        spec = load_default_spec()
        out = tmp_path / "out"
        assert main(["scan", str(vlm_model), "--out", str(out)]) == 0
        assert main(["render", str(out)]) == 0

        field = load_channel_field(out, "vision_height", spec, model_name="vlm")
        assert field is not None
        assert field.col_labels == list(spec.vision_slots)
        assert field.row_labels == ["0", "1", "global"]
        assert field.model_name == "vlm"

        pngs = sorted((out / "render").glob("vision_*.png"))
        assert any(p.name == "vision_height_raw.png" for p in pngs)

    def test_text_only_scan_has_no_vision_artefacts(self, tmp_path):
        from tests.fixtures import make_fake_model
        from weight_atlas.scan import scan

        model = tmp_path / "text.safetensors"
        make_fake_model(model)
        spec = load_default_spec()
        out = tmp_path / "out"
        scan(model, out, spec)

        fp = json.loads((out / "fingerprint.json").read_text())
        assert "vision" not in fp["model"]
        assert not list(out.glob("field_vision_*.tif"))
