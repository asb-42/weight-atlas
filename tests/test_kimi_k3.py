"""Tests for Kimi K3 support: MXFP4 dequant, BF16 loader, name mapping."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from weight_atlas.core.name_map import (
    extract_expert_id,
    get_moe_slot,
    is_expert_tensor,
    is_shared_expert,
    map_name,
)
from weight_atlas.loaders.mxfp4 import (
    dequantize_mxfp4,
    e8m0_to_float,
    scale_name_for_packed,
    unpack_e2m1,
    weight_name_for_packed,
)
from weight_atlas.loaders.safetensors_loader import SafetensorsLoader


# --------------------------------------------------------------------------
# MXFP4 dequantizer
# --------------------------------------------------------------------------
def test_unpack_e2m1_interleaves_low_high_nibbles():
    """Element at col 2j = low nibble, col 2j+1 = high nibble (reference layout)."""
    # nibbles (E2M1 indices): [2,3] -> bytes 0x32, ... (low=2, high=3)
    packed = np.array([[0x32, 0x32, 0x32, 0x32]], dtype=np.uint8)  # (1, 4)
    vals = unpack_e2m1(packed)
    # E2M1: idx 2 -> 1.0, idx 3 -> 1.5 (LUT [0,.5,1,1.5,2,3,4,6])
    expected = np.array([[1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5]], dtype=np.float32)
    np.testing.assert_allclose(vals, expected)
    assert vals.shape == (1, 8)


def test_unpack_e2m1_sign_bit_and_lut():
    nibbles = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.uint8)
    packed = np.zeros((1, 8), dtype=np.uint8)
    packed[0, :] = nibbles[0::2] | (nibbles[1::2] << 4)
    vals = unpack_e2m1(packed)
    lut = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    expected = lut[nibbles & 0x07] * np.where((nibbles & 0x08) != 0, -1.0, 1.0)
    np.testing.assert_allclose(vals[0], expected.astype(np.float32))


def test_e8m0_to_float():
    np.testing.assert_allclose(e8m0_to_float(np.array([127, 128, 126], dtype=np.uint8)),
                               [1.0, 2.0, 0.5])


def test_dequantize_mxfp4_broadcasts_group_scale():
    """All-1.0 FP4 (index 2) with E8M0 scale 2.0 -> all 2.0, shape (m, k)."""
    m, k, g = 1, 32, 32
    nibble = 2  # LUT value 1.0
    byte = nibble | (nibble << 4)
    packed = np.full((m, k // 2), byte, dtype=np.uint8)
    scale = np.full((m, k // g), 128, dtype=np.uint8)  # 2 ** (128 - 127) = 2.0
    out = dequantize_mxfp4(packed, scale, group_size=g)
    assert out.shape == (m, k)
    np.testing.assert_allclose(out, np.full((m, k), 2.0, dtype=np.float32))


def test_dequantize_mxfp4_known_row():
    """Known nibble pattern x E8M0 scale -> exact dequantized row."""
    m, k, g = 2, 32, 32
    rng = np.random.default_rng(0)
    nibbles = rng.integers(0, 16, size=(m, k)).astype(np.uint8)
    packed = np.zeros((m, k // 2), dtype=np.uint8)
    packed[:, :] = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
    scale_bytes = rng.integers(100, 140, size=(m, k // g)).astype(np.uint8)
    out = dequantize_mxfp4(packed, scale_bytes, group_size=g)
    lut = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    expected = lut[nibbles & 0x07] * np.where((nibbles & 0x08) != 0, -1.0, 1.0)
    scales = 2.0 ** (scale_bytes.astype(np.float64) - 127.0)
    expected = (expected.reshape(m, -1, g) * scales[:, :, None]).reshape(m, k)
    np.testing.assert_allclose(out, expected.astype(np.float32), rtol=1e-6)


def test_mxfp4_name_helpers():
    p = "language_model.model.layers.3.block_sparse_moe.experts.12.w1.weight_packed"
    assert weight_name_for_packed(p) == "language_model.model.layers.3.block_sparse_moe.experts.12.w1.weight"
    assert scale_name_for_packed(p) == "language_model.model.layers.3.block_sparse_moe.experts.12.w1.weight_scale"


# --------------------------------------------------------------------------
# Kimi K3 name mapping
# --------------------------------------------------------------------------
def test_kimi_moe_mapping():
    assert map_name("language_model.model.layers.3.block_sparse_moe.gate.weight") == (3, "router")
    assert map_name("language_model.model.layers.3.block_sparse_moe.experts.895.w1.weight") == (3, "expert")
    assert map_name("language_model.model.layers.3.block_sparse_moe.experts.895.w2.weight") == (3, "expert")
    assert map_name("language_model.model.layers.3.block_sparse_moe.experts.895.w3.weight") == (3, "expert")
    assert map_name("language_model.model.layers.3.block_sparse_moe.shared_experts.w1.weight") == (3, "shared_expert")
    assert map_name("language_model.model.layers.3.block_sparse_moe.routed_expert_norm.weight") == (3, "moe_routed_norm")


def test_kimi_moe_helpers():
    name = "language_model.model.layers.3.block_sparse_moe.experts.42.w1.weight"
    assert extract_expert_id(name) == 42
    assert get_moe_slot(name) == "gate"
    assert get_moe_slot(name.replace("w1", "w2")) == "up"
    assert get_moe_slot(name.replace("w1", "w3")) == "down"
    assert is_expert_tensor(name)
    assert is_shared_expert("language_model.model.layers.3.block_sparse_moe.shared_experts.w1.weight")


def test_kimi_attention_and_mla_mapping():
    prefix = "language_model.model.layers.7"
    assert map_name(f"{prefix}.self_attn.q_a_proj.weight") == (7, "attn_q_a")
    assert map_name(f"{prefix}.self_attn.q_a_layernorm.weight") == (7, "attn_q_a_norm")
    assert map_name(f"{prefix}.self_attn.kv_a_proj_with_mqa.weight") == (7, "attn_kv_a")
    assert map_name(f"{prefix}.self_attn.kv_b_proj.weight") == (7, "attn_kv_b")
    assert map_name(f"{prefix}.self_attn.q_conv1d.weight") == (7, "attn_q_conv")
    assert map_name(f"{prefix}.self_attn.f_a_proj.weight") == (7, "attn_f_a")
    assert map_name(f"{prefix}.self_attn.b_proj.weight") == (7, "attn_b")
    assert map_name(f"{prefix}.self_attn.A_log") == (7, "attn_a_log")
    assert map_name(f"{prefix}.self_attn.g_proj.weight") == (7, "attn_gate")
    assert map_name(f"{prefix}.self_attention_res_proj.weight") == (7, "attn_res_proj")
    assert map_name(f"{prefix}.mlp_res_norm.weight") == (7, "mlp_res_norm")


def test_kimi_vision_and_projector_mapping():
    assert map_name("vision_tower.encoder.blocks.5.wqkv.weight") == (None, "vision_qkv")
    assert map_name("vision_tower.encoder.blocks.5.wo.weight") == (None, "vision_o")
    assert map_name("vision_tower.encoder.blocks.5.norm0.weight") == (None, "vision_norm")
    assert map_name("vision_tower.encoder.blocks.5.mlp.fc1.weight") == (None, "vision_mlp")
    assert map_name("vision_tower.patch_embed.pos_emb.weight") == (None, "vision_pos_emb")
    assert map_name("mm_projector.proj.0.weight") == (None, "mm_projector")


def test_kimi_non_layer_tensors():
    assert map_name("language_model.model.embed_tokens.weight") == (None, "embed")
    assert map_name("language_model.model.norm.weight") == (None, "norm_mlp")
    assert map_name("language_model.model.lm_head.weight") == (None, "lm_head")
    assert map_name("language_model.model.output_attn_res_norm.weight") == (None, "attn_res_norm")


# --------------------------------------------------------------------------
# Loader: BF16 reading + MXFP4 pair merging
# --------------------------------------------------------------------------
def _bf16_bytes(values: list[float]) -> bytes:
    out = bytearray()
    for v in values:
        f32 = struct.unpack("<I", struct.pack("<f", float(v)))[0]
        out += struct.pack("<H", (f32 >> 16) & 0xFFFF)
    return bytes(out)


def _write_st_file(path: Path, tensors: dict[str, tuple[str, list, bytes]]) -> None:
    """Write a safetensors file from raw tensor payloads.

    ``tensors`` maps name -> (dtype_str, shape_list, raw_bytes).
    """
    header: dict = {}
    data = b""
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    hb = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + data)


def test_loader_bf16_and_mxfp4_pair(tmp_path):
    base = "language_model.model.layers.0"
    tensors = {
        "language_model.model.norm.weight": ("F32", [4], np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tobytes()),
        f"{base}.input_layernorm.weight": ("BF16", [4], _bf16_bytes([0.5, -1.5, 2.0, 0.25])),
        f"{base}.block_sparse_moe.experts.3.w1.weight_packed": ("U8", [1, 16], np.full((1, 16), 0x22, dtype=np.uint8).tobytes()),
        f"{base}.block_sparse_moe.experts.3.w1.weight_scale": ("U8", [1, 1], np.array([[127]], dtype=np.uint8).tobytes()),
    }
    _write_st_file(tmp_path / "model.safetensors", tensors)

    handles = SafetensorsLoader().open(tmp_path)
    by_name = {h.name: h for h in handles}

    # No raw packed/scale handles survive as standalone tensors.
    assert not any(n.endswith("weight_packed") or n.endswith("weight_scale") for n in by_name)

    # BF16 norm loads as float32 with correct values.
    np.testing.assert_allclose(
        by_name["language_model.model.norm.weight"].load(), [1.0, 2.0, 3.0, 4.0], rtol=1e-6
    )
    np.testing.assert_allclose(
        by_name[f"{base}.input_layernorm.weight"].load(), [0.5, -1.5, 2.0, 0.25], atol=1e-3
    )

    # MXFP4 pair merged into a single dequantized handle with real geometry.
    deq_name = f"{base}.block_sparse_moe.experts.3.w1.weight"
    h = by_name[deq_name]
    assert h.shape == (1, 32)         # packed (1,16) -> real (1,32)
    assert h.expert_id == 3
    np.testing.assert_allclose(h.load(), np.ones((1, 32), dtype=np.float32), rtol=1e-6)
    assert by_name[deq_name].dtype == "FP4_MXFP4"
