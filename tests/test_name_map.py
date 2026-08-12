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
    ],
)
def test_slot_mapping(name, expected_slot):
    layer, slot = map_name(name)
    assert slot == expected_slot


def test_layer_index_extracted():
    layer, _ = map_name("model.layers.7.self_attn.q_proj.weight")
    assert layer == 7


def test_non_layer_has_none_index():
    layer, slot = map_name("model.embed_tokens.weight")
    assert layer is None
    assert slot == "embed"


def test_unknown_maps_to_other():
    layer, slot = map_name("model.layers.0.weird_tensor.weight")
    assert slot == "other"
    assert layer == 0
