"""Name audit tests: Bonsai-8B tensor name mapping, rule ordering, and mapping coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from weight_atlas.core.name_map import map_name

# Load Bonsai fixture
_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "names_bonsai_8b.json"
_BONSAI = json.loads(_FIXTURE_PATH.read_text())

# Load Qwen3-Next (hybrid) fixture
_QQWEN3_PATH = Path(__file__).parent / "fixtures" / "names_qwen3_next_gguf.json"
_QWEN3_NEXT = json.loads(_QQWEN3_PATH.read_text())


@pytest.mark.parametrize("name,expected", _BONSAI["expected_mapping"].items())
def test_bonsai_hf_mapping(name, expected):
    """Every Bonsai HF tensor name maps to the expected (layer, slot)."""
    layer, slot = map_name(name)
    exp_layer, exp_slot = expected
    assert layer == exp_layer, f"{name}: layer {layer} != {exp_layer}"
    assert slot == exp_slot, f"{name}: slot {slot} != {exp_slot}"


@pytest.mark.parametrize("name,expected", _BONSAI["gguf_expected_mapping"].items())
def test_bonsai_gguf_mapping(name, expected):
    """Every Bonsai GGUF tensor name maps to the expected (layer, slot)."""
    layer, slot = map_name(name)
    exp_layer, exp_slot = expected
    assert layer == exp_layer, f"{name}: layer {layer} != {exp_layer}"
    assert slot == exp_slot, f"{name}: slot {slot} != {exp_slot}"


@pytest.mark.parametrize("name,expected", _QWEN3_NEXT["gguf_expected_mapping"].items())
def test_qwen3_next_gguf_mapping(name, expected):
    """Every Qwen3-Next GGUF tensor name maps to the expected (layer, slot).

    Covers the hybrid attention+Mamba branch: attn_gate, post_attention_norm,
    and the full ssm_* Mamba family.
    """
    layer, slot = map_name(name)
    exp_layer, exp_slot = expected
    assert layer == exp_layer, f"{name}: layer {layer} != {exp_layer}"
    assert slot == exp_slot, f"{name}: slot {slot} != {exp_slot}"


def test_mapping_coverage_qwen3_next():
    """All Qwen3-Next tensors must map to a non-'other' slot."""
    all_names = _QWEN3_NEXT["gguf_tensor_names"]
    unmapped = [name for name in all_names if map_name(name)[1] == "other"]
    assert not unmapped, f"Unmapped Qwen3-Next tensors: {unmapped}"


def test_no_duplicate_layer_slot_mapping():
    """No two tensor names map to the same (layer, slot) — would indicate rule ambiguity.

    Checks within each naming convention separately since HF and GGUF
    legitimately map to the same slots (they represent the same tensors
    in different file formats).
    """
    for convention, names in [("HF", _BONSAI["tensor_names"]), ("GGUF", _BONSAI["gguf_tensor_names"])]:
        seen: dict[tuple[int, str], str] = {}
        for name in names:
            layer, slot = map_name(name)
            if layer is None:
                continue  # skip non-layer tensors (embed, lm_head, norms)
            key = (layer, slot)
            if key in seen:
                pytest.fail(
                    f"[{convention}] Duplicate (layer, slot) {key}: '{name}' and '{seen[key]}'"
                )
            seen[key] = name


def test_attn_q_norm_before_attn_q():
    """q_norm must map to attn_q_norm, not attn_q (rule ordering)."""
    layer, slot = map_name("model.layers.0.self_attn.q_norm.weight")
    assert slot == "attn_q_norm", f"q_norm mapped to {slot}, expected attn_q_norm"
    assert layer == 0


def test_attn_k_norm_before_attn_k():
    """k_norm must map to attn_k_norm, not attn_k (rule ordering)."""
    layer, slot = map_name("model.layers.0.self_attn.k_norm.weight")
    assert slot == "attn_k_norm", f"k_norm mapped to {slot}, expected attn_k_norm"
    assert layer == 0


def test_gguf_attn_q_norm_before_attn_q():
    """GGUF blk.N.attn_q_norm must map to attn_q_norm, not attn_q."""
    layer, slot = map_name("blk.5.attn_q_norm.weight")
    assert slot == "attn_q_norm", f"blk.5.attn_q_norm mapped to {slot}, expected attn_q_norm"
    assert layer == 5


def test_gguf_attn_k_norm_before_attn_k():
    """GGUF blk.N.attn_k_norm must map to attn_k_norm, not attn_k."""
    layer, slot = map_name("blk.5.attn_k_norm.weight")
    assert slot == "attn_k_norm", f"blk.5.attn_k_norm mapped to {slot}, expected attn_k_norm"
    assert layer == 5


def test_mapping_coverage_bonsai():
    """All Bonsai tensor names must map to a non-'other' slot."""
    all_names = _BONSAI["tensor_names"] + _BONSAI["gguf_tensor_names"]
    unmapped = []
    for name in all_names:
        _, slot = map_name(name)
        if slot == "other":
            unmapped.append(name)
    assert not unmapped, f"Unmapped Bonsai tensors: {unmapped}"


def test_in_slots_calculation():
    """Calculate in_slots ratio for Bonsai-8B (should be >= 80%)."""
    all_names = _BONSAI["tensor_names"] + _BONSAI["gguf_tensor_names"]
    total = len(all_names)
    in_slots = sum(1 for name in all_names if map_name(name)[1] != "other")
    ratio = in_slots / total if total > 0 else 0.0
    assert ratio >= 0.8, f"in_slots ratio {ratio:.1%} < 80%"
