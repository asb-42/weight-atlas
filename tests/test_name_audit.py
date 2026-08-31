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

# Load Gemma-4 ultra/heretic (MoE + mmproj) fixture
_GEMMA4_PATH = Path(__file__).parent / "fixtures" / "names_gemma4_heretic_gguf.json"
_GEMMA4 = json.loads(_GEMMA4_PATH.read_text())

# Load Qwen3.8-27B fixture (hybrid full/GDN linear attention; real names from
# the alesha-pro/atlas shipped scan — the Qwen3.8-Flash-Next naming family)
_QWEN38_PATH = Path(__file__).parent / "fixtures" / "names_qwen38_27b_hf.json"
_QWEN38 = json.loads(_QWEN38_PATH.read_text())

# Qwen3.8 MTP-head globals with no dedicated slot yet (known mapping gap,
# documented in docs/2026-08-31_atlas-alesha-pro-analysis.md §P0):
# mapping them needs an MTP slot design (GGUF solves this via blk.N.nextn.*).
_QWEN38_MTP_GLOBAL_OTHER = {
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.fc.weight",
}


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


@pytest.mark.parametrize("name,expected", _GEMMA4["gguf_expected_mapping"].items())
def test_gemma4_gguf_mapping(name, expected):
    """Every Gemma-4 ultra/heretic GGUF tensor name maps to the expected (layer, slot).

    Covers the extra per-layer tensors (pre/post_ffw_norm variants,
    layer_output_scale) and the global RoPE frequency table.
    """
    layer, slot = map_name(name)
    exp_layer, exp_slot = expected
    assert layer == exp_layer, f"{name}: layer {layer} != {exp_layer}"
    assert slot == exp_slot, f"{name}: slot {slot} != {exp_slot}"


@pytest.mark.parametrize("name,expected", _GEMMA4["mmproj_expected_mapping"].items())
def test_gemma4_mmproj_mapping(name, expected):
    """Every Gemma-4 mmproj tensor name maps to the expected vision slot."""
    layer, slot = map_name(name)
    exp_layer, exp_slot = expected
    assert layer == exp_layer, f"{name}: layer {layer} != {exp_layer}"
    assert slot == exp_slot, f"{name}: slot {slot} != {exp_slot}"


def test_mapping_coverage_gemma4():
    """All Gemma-4 GGUF + mmproj tensors must map to a non-'other' slot."""
    all_names = _GEMMA4["gguf_tensor_names"] + _GEMMA4["mmproj_tensor_names"]
    unmapped = [name for name in all_names if map_name(name)[1] == "other"]
    assert not unmapped, f"Unmapped Gemma-4 tensors: {unmapped}"


def test_in_slots_calculation():
    """Calculate in_slots ratio for Bonsai-8B (should be >= 80%)."""
    all_names = _BONSAI["tensor_names"] + _BONSAI["gguf_tensor_names"]
    total = len(all_names)
    in_slots = sum(1 for name in all_names if map_name(name)[1] != "other")
    ratio = in_slots / total if total > 0 else 0.0
    assert ratio >= 0.8, f"in_slots ratio {ratio:.1%} < 80%"


# ── Qwen3.8-27B family (GDN linear attention, HF naming) ──────────────────


@pytest.mark.parametrize(
    "suffix,slot",
    [
        ("in_proj_qkv.weight", "ssm_in_qkv"),
        ("in_proj_z.weight", "ssm_in_z"),
        ("in_proj_b.weight", "ssm_in_b"),
        ("in_proj_a.weight", "ssm_in_a"),
        ("conv1d.weight", "ssm_conv1d"),
        ("dt_bias", "ssm_dt"),
        ("A_log", "ssm_a"),
        ("norm.weight", "ssm_norm"),
        ("out_proj.weight", "ssm_out"),
    ],
)
def test_qwen38_gdn_linear_attn_mapping(suffix, slot):
    """GDN linear_attn.* names (Qwen3.8 family) map to their ssm_* slots."""
    layer, got = map_name(f"model.language_model.layers.7.linear_attn.{suffix}")
    assert (layer, got) == (7, slot), f"linear_attn.{suffix} -> {(layer, got)}, expected (7, {slot})"


def test_qwen38_gdn_anchoring():
    """The linear_attn rules must not steal names they are not about."""
    # in_proj_a must not match a hypothetical in_proj_ab / in_proj_a_x
    layer, slot = map_name("model.layers.0.linear_attn.in_proj_a_extra.weight")
    assert slot == "other", f"anchored rule stolen in_proj_a_extra -> {slot}"
    # ssm.* Mamba-branch naming must still map through its own rules
    layer, slot = map_name("model.layers.0.ssm.conv1d.weight")
    assert (layer, slot) == (0, "ssm_conv1d")


def test_mapping_coverage_qwen38():
    """All Qwen3.8-27B tensors map to a non-'other' slot, except the four
    known MTP-head globals (see _QWEN38_MTP_GLOBAL_OTHER)."""
    unmapped = [
        name
        for name in _QWEN38["hf_tensor_names"]
        if map_name(name)[1] == "other" and name not in _QWEN38_MTP_GLOBAL_OTHER
    ]
    assert not unmapped, f"Unmapped Qwen3.8 tensors: {unmapped}"


def test_mapping_coverage_qwen38_ratio():
    """in_slots ratio for Qwen3.8-27B (real shipped names) — was 63.6% before
    the GDN rules, must stay >= 99% after."""
    names = _QWEN38["hf_tensor_names"]
    in_slots = sum(1 for name in names if map_name(name)[1] != "other")
    ratio = in_slots / len(names)
    assert ratio >= 0.99, f"in_slots ratio {ratio:.1%} < 99%"


def test_qwen38_mtp_layer_collision_known():
    """Documented known issue: mtp.layers.N.* shares the main-stack layer
    index space, so e.g. mtp.layers.0.self_attn.q_proj collides with
    model.language_model.layers.0.self_attn.q_proj at (0, attn_q). Fixing
    this needs an MTP slot design (see analysis doc); here we pin the
    current behaviour so the gap stays visible."""
    main_layer, main_slot = map_name("model.language_model.layers.0.self_attn.q_proj.weight")
    mtp_layer, mtp_slot = map_name("mtp.layers.0.self_attn.q_proj.weight")
    assert (mtp_layer, mtp_slot) == (main_layer, main_slot), (
        "collision behaviour changed — revisit _QWEN38_MTP_GLOBAL_OTHER / analysis doc"
    )
