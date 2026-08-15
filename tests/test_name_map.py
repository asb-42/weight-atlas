"""Tests for core/name_map.py against real tensor naming patterns."""

from __future__ import annotations

import pytest

from weight_atlas.core.name_map import map_name


@pytest.mark.parametrize(
    "name,expected_slot",
    [
        ("model.layers.0.self_attn.q_proj.weight", "attn_q"),
        ("model.layers.1.self_attn.k_proj.weight", "attn_k"),
        ("model.layers.2.self_attn.v_proj.weight", "attn_v"),
        ("model.layers.3.self_attn.o_proj.weight", "attn_o"),
        ("model.layers.0.mlp.gate_proj.weight", "mlp_gate"),
        ("model.layers.1.mlp.up_proj.weight", "mlp_up"),
        ("model.layers.2.mlp.down_proj.weight", "mlp_down"),
        ("model.layers.0.input_layernorm.weight", "norm_attn"),
        ("model.layers.0.post_attention_layernorm.weight", "norm_mlp"),
        ("model.embed_tokens.weight", "embed"),
        ("lm_head.weight", "lm_head"),
        ("blk.0.attn_q_norm.weight", "attn_q_norm"),
        ("blk.0.attn_k_norm.weight", "attn_k_norm"),
        ("blk.0.attn_q.weight", "attn_q"),
        ("blk.0.attn_k.weight", "attn_k"),
        ("blk.0.attn_v.weight", "attn_v"),
        ("blk.0.attn_output.weight", "attn_o"),
        ("blk.0.attn_sinks.weight", "attn_sinks"),
        ("blk.0.ffn_gate.weight", "mlp_gate"),
        ("blk.0.ffn_up.weight", "mlp_up"),
        ("blk.0.ffn_down.weight", "mlp_down"),
        ("blk.0.attn_norm.weight", "norm_attn"),
        ("blk.0.ffn_norm.weight", "norm_mlp"),
        ("token_embd.weight", "embed"),
        ("output.weight", "lm_head"),
        ("blk.0.attn_gate.weight", "attn_gate"),
        ("blk.0.post_attention_norm.weight", "norm_mlp"),
        ("blk.0.ssm_a", "ssm_a"),
        ("blk.0.ssm_ba", "ssm_ba"),
        ("blk.0.ssm_alpha.weight", "ssm_alpha"),
        ("blk.0.ssm_beta.weight", "ssm_beta"),
        ("blk.0.ssm_conv1d.weight", "ssm_conv1d"),
        ("blk.0.ssm_dt.bias", "ssm_dt"),
        ("blk.0.ssm_norm.weight", "ssm_norm"),
        ("blk.0.ssm_out.weight", "ssm_out"),
        # Qwen3-Next MTP draft head (blk.N.nextn.*)
        ("blk.64.nextn.eh_proj.weight", "mtp_eh_proj"),
        ("blk.64.nextn.enorm.weight", "mtp_enorm"),
        ("blk.64.nextn.hnorm.weight", "mtp_hnorm"),
        ("blk.64.nextn.shared_head_norm.weight", "mtp_shared_head_norm"),
        # Gemma-4 "ultra"/heretic extra per-layer tensors
        ("blk.0.layer_output_scale.weight", "layer_output_scale"),
        ("blk.0.post_ffw_norm.weight", "post_ffw_norm"),
        ("blk.0.post_ffw_norm_1.weight", "post_ffw_norm_1"),
        ("blk.0.post_ffw_norm_2.weight", "post_ffw_norm_2"),
        ("blk.0.pre_ffw_norm_2.weight", "pre_ffw_norm_2"),
        ("blk.29.post_ffw_norm.weight", "post_ffw_norm"),
        # Gemma-4 RoPE frequency table (global)
        ("rope_freqs.weight", "rope_freqs"),
    ],
)
def test_slot_mapping(name, expected_slot):
    layer, slot = map_name(name)
    assert slot == expected_slot


@pytest.mark.parametrize(
    "name,expected_slot",
    [
        ("v.blk.0.attn_out.weight", "v_attn_o"),
        ("v.blk.0.attn_out.bias", "v_attn_o"),
        ("v.blk.0.ln1.weight", "v_attn_norm"),
        ("v.blk.0.ln2.bias", "v_mlp_norm"),
        ("v.blk.5.mlp.fc0.weight", "v_mlp"),
        ("v.blk.5.other.weight", "v_other"),
        ("v.patch_embed.weight", "v_patch_embed"),
        ("v.pos_embed.weight", "v_pos_emb"),
        ("mm.0.weight", "mm_projector"),
        ("mm.2.weight", "mm_projector"),
        ("mm.model.mlp.1.weight", "mm_projector"),
        ("mm.input_projection.weight", "mm_projector"),  # Gemma-4 mmproj
        ("model.vision_model.encoder.layers.5.self_attn.q_proj.weight", "v_attn_q"),
        ("model.visual.blocks.2.attn.qkv.weight", "v_attn_qkv"),
        ("model.multi_modal_projector.layers.0.linear_1.weight", "mm_projector"),
    ],
)
def test_vlm_mapping(name, expected_slot):
    """VLM (vision-language) tensors map to vision/projector slots, non-layer."""
    layer, slot = map_name(name)
    assert slot == expected_slot
    assert layer is None, "VLM tensors must not collide with language-model layers"


def test_layer_index_extracted():
    layer, _ = map_name("model.layers.7.self_attn.q_proj.weight")
    assert layer == 7


def test_ssm_ba_hf_maps_to_ssm_ba():
    """Qwen3-Next HF: the Mamba B-matrix maps to its own ssm_ba slot.

    Regression for the 36 unmapped ``blk.*.ssm_ba.weight`` tensors in the
    Qwen3-Next fingerprint: the B-matrix must not fall through to ``other``.
    """
    layer, slot = map_name("model.layers.3.ssm.ba.weight")
    assert slot == "ssm_ba"
    assert layer == 3


def test_ssm_ba_gguf_maps_to_ssm_ba():
    layer, slot = map_name("blk.3.ssm_ba")
    assert slot == "ssm_ba"
    assert layer == 3


def test_non_layer_has_none_index():
    layer, slot = map_name("model.embed_tokens.weight")
    assert layer is None
    assert slot == "embed"


def test_unknown_maps_to_other():
    layer, slot = map_name("model.layers.0.weird_tensor.weight")
    assert slot == "other"
    assert layer == 0
