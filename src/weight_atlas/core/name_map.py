"""Tensor name → (layer_index, slot) mapping for model families.

Convention: regex rules applied in order; first match wins. Unknown names
map to slot ``"other"`` (never silently dropped).

Supports several naming conventions:
- HuggingFace: model.layers.N.self_attn.q_proj.weight
- GGUF: blk.N.attn_q.weight
- MoE (HF): model.layers.N.mlp.experts.{e}.gate_proj.weight
- VLM vision towers (GGUF v.blk.N.* / HF vision_model / visual / vision_tower)
  and multimodal projectors, via :func:`map_vision` (own slot taxonomy).
"""

from __future__ import annotations

import re

# Ordered regex rules: (pattern, slot). First match wins.
# HuggingFace rules (original)
_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"embed_tokens"), "embed"),
    (re.compile(r"model\.norm"), "norm_mlp"),
    (re.compile(r"lm_head"), "lm_head"),
    (re.compile(r"self_attn\.q_norm"), "attn_q_norm"),
    (re.compile(r"self_attn\.k_norm"), "attn_k_norm"),
    (re.compile(r"self_attn\.q_proj"), "attn_q"),
    (re.compile(r"self_attn\.k_proj"), "attn_k"),
    (re.compile(r"self_attn\.v_proj"), "attn_v"),
    (re.compile(r"self_attn\.o_proj"), "attn_o"),
    (re.compile(r"mlp\.gate_proj"), "mlp_gate"),
    (re.compile(r"mlp\.up_proj"), "mlp_up"),
    (re.compile(r"mlp\.down_proj"), "mlp_down"),
    (re.compile(r"input_layernorm"), "norm_attn"),
    (re.compile(r"post_attention_layernorm"), "norm_mlp"),
    (re.compile(r"router"), "router"),
]

# MoE-specific rules (HF)
# IMPORTANT: Order matters! mlp.gate.weight (router) must come before mlp.gate_proj
_MOE_RULES: list[tuple[re.Pattern[str], str | None]] = [
    (re.compile(r"mlp\.gate\.weight"), "router"),  # MoE router (before mlp_gate)
    (re.compile(r"shared_expert_gate"), "other"),  # Shared expert gate → other
    (re.compile(r"shared_expert\.(gate|up|down)_proj"), None),  # Shared expert → mlp slots (handled specially)
    (re.compile(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj"), None),  # Expert tensors (handled specially)
    # Kimi K3 / DeepSeek-style block_sparse_moe MoE
    (re.compile(r"block_sparse_moe\.shared_experts"), None),  # → shared_expert slot (handled specially)
    (re.compile(r"block_sparse_moe\.experts\.\d+\.w[123]"), None),  # → expert slot (handled specially)
    (re.compile(r"block_sparse_moe\.gate"), "router"),  # Kimi router (before generic fallbacks)
]

# Qwen3-Next / hybrid Mamba-aware HF rules (applied after _RULES, before fallback).
# The Mamba/SSM branch and the attention gate use distinct sub-block names.
_HF_HYBRID_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"self_attn\.attn_gate"), "attn_gate"),
    (re.compile(r"self_attn\.gating"), "attn_gate"),
    (re.compile(r"post_attention_norm"), "norm_mlp"),
    (re.compile(r"ssm\.a\b"), "ssm_a"),
    (re.compile(r"ssm\.alpha"), "ssm_alpha"),
    (re.compile(r"ssm\.beta"), "ssm_beta"),
    (re.compile(r"ssm\.conv1d"), "ssm_conv1d"),
    (re.compile(r"ssm\.dt"), "ssm_dt"),
    (re.compile(r"ssm\.norm"), "ssm_norm"),
    (re.compile(r"ssm\.out_proj"), "ssm_out"),
]

# Kimi K3 (language_model.model.layers.N.*) rules — applied after base HF rules.
# Covers MLA (q_a/q_b/kv_a/kv_b), linear-attention/KDA (conv1d, f_a/f_b, b, A_log,
# dt) and hybrid residual branches. The vision tower and multimodal projector
# are handled by ``_VISION_RULES``/``map_vision`` (checked first).
_KIMI_RULES: list[tuple[re.Pattern[str], str]] = [
    # MLA (multi-head latent attention) projections
    (re.compile(r"self_attn\.q_a_proj"), "attn_q_a"),
    (re.compile(r"self_attn\.q_a_layernorm"), "attn_q_a_norm"),
    (re.compile(r"self_attn\.q_b_proj"), "attn_q_b"),
    (re.compile(r"self_attn\.kv_a_proj_with_mqa"), "attn_kv_a"),
    (re.compile(r"self_attn\.kv_a_layernorm"), "attn_kv_a_norm"),
    (re.compile(r"self_attn\.kv_b_proj"), "attn_kv_b"),
    (re.compile(r"self_attn\.o_norm"), "attn_o_norm"),
    # Latent-MoE routed-expert projections
    (re.compile(r"block_sparse_moe\.routed_expert_norm"), "moe_routed_norm"),
    (re.compile(r"block_sparse_moe\.routed_expert_up_proj"), "moe_routed_up"),
    (re.compile(r"block_sparse_moe\.routed_expert_down_proj"), "moe_routed_down"),
    # Linear-attention (KDA) convolution filters and projections
    (re.compile(r"self_attn\.q_conv1d"), "attn_q_conv"),
    (re.compile(r"self_attn\.k_conv1d"), "attn_k_conv"),
    (re.compile(r"self_attn\.v_conv1d"), "attn_v_conv"),
    (re.compile(r"self_attn\.f_a_proj"), "attn_f_a"),
    (re.compile(r"self_attn\.f_b_proj"), "attn_f_b"),
    (re.compile(r"self_attn\.b_proj"), "attn_b"),
    (re.compile(r"self_attn\.g_proj"), "attn_gate"),
    (re.compile(r"self_attn\.A_log"), "attn_a_log"),
    (re.compile(r"self_attn\.dt_bias"), "attn_dt"),
    # Hybrid linear-attention residual branches
    (re.compile(r"self_attention_res_proj"), "attn_res_proj"),
    (re.compile(r"self_attention_res_norm"), "attn_res_norm"),
    (re.compile(r"mlp_res_proj"), "mlp_res_proj"),
    (re.compile(r"mlp_res_norm"), "mlp_res_norm"),
    (re.compile(r"output_attn_res_proj"), "attn_res_proj"),
    (re.compile(r"output_attn_res_norm"), "attn_res_norm"),
]

# GGUF-specific rules (applied after HF rules, before fallback)
_GGUF_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"blk\.\d+\.attn_q_norm"), "attn_q_norm"),
    (re.compile(r"blk\.\d+\.attn_k_norm"), "attn_k_norm"),
    (re.compile(r"blk\.\d+\.attn_q"), "attn_q"),
    (re.compile(r"blk\.\d+\.attn_k"), "attn_k"),
    (re.compile(r"blk\.\d+\.attn_v"), "attn_v"),
    (re.compile(r"blk\.\d+\.attn_output"), "attn_o"),
    (re.compile(r"blk\.\d+\.ffn_gate"), "mlp_gate"),
    (re.compile(r"blk\.\d+\.ffn_up"), "mlp_up"),
    (re.compile(r"blk\.\d+\.ffn_down"), "mlp_down"),
    (re.compile(r"blk\.\d+\.attn_norm"), "norm_attn"),
    (re.compile(r"blk\.\d+\.ffn_norm"), "norm_mlp"),
    # Gemma-4 "ultra"/heretic: extra FFW norms and per-layer output scale.
    # Ordered most-specific first: post_ffw_norm_1/_2 before post_ffw_norm.
    (re.compile(r"blk\.\d+\.post_ffw_norm_1"), "post_ffw_norm_1"),
    (re.compile(r"blk\.\d+\.post_ffw_norm_2"), "post_ffw_norm_2"),
    (re.compile(r"blk\.\d+\.post_ffw_norm"), "post_ffw_norm"),
    (re.compile(r"blk\.\d+\.pre_ffw_norm_2"), "pre_ffw_norm_2"),
    (re.compile(r"blk\.\d+\.layer_output_scale"), "layer_output_scale"),
    (re.compile(r"rope_freqs"), "rope_freqs"),
    (re.compile(r"token_embd"), "embed"),
    (re.compile(r"output\.weight"), "lm_head"),
    (re.compile(r"output_norm"), "norm_mlp"),
]

# Qwen3-Next / hybrid Mamba-aware GGUF rules (applied after base GGUF rules).
# Order matters within this group: more specific ssm patterns first,
# and ``ssm_a`` (a plain suffix) must come AFTER ``ssm_alpha``/``ssm_beta``.
_GGUF_HYBRID_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"blk\.\d+\.attn_gate"), "attn_gate"),
    (re.compile(r"blk\.\d+\.post_attention_norm"), "norm_mlp"),
    (re.compile(r"blk\.\d+\.ssm_alpha"), "ssm_alpha"),
    (re.compile(r"blk\.\d+\.ssm_beta"), "ssm_beta"),
    (re.compile(r"blk\.\d+\.ssm_conv1d"), "ssm_conv1d"),
    (re.compile(r"blk\.\d+\.ssm_dt"), "ssm_dt"),
    (re.compile(r"blk\.\d+\.ssm_norm"), "ssm_norm"),
    (re.compile(r"blk\.\d+\.ssm_out"), "ssm_out"),
    (re.compile(r"blk\.\d+\.ssm_a"), "ssm_a"),
]

# GGUF MoE-specific rules
_GGUF_MOE_RULES: list[tuple[re.Pattern[str], str | None]] = [
    (re.compile(r"blk\.\d+\.ffn_gate_inp"), "router"),  # MoE router
    (re.compile(r"blk\.\d+\.ffn_gate_exps"), None),  # Expert gate (3D stacked)
    (re.compile(r"blk\.\d+\.ffn_up_exps"), None),  # Expert up (3D stacked)
    (re.compile(r"blk\.\d+\.ffn_down_exps"), None),  # Expert down (3D stacked)
    (re.compile(r"blk\.\d+\.ffn_gate_shexp"), "mlp_gate"),  # Shared expert gate
    (re.compile(r"blk\.\d+\.ffn_up_shexp"), "mlp_up"),  # Shared expert up
    (re.compile(r"blk\.\d+\.ffn_down_shexp"), "mlp_down"),  # Shared expert down
]

# VLM (vision-language) rules. Covers the major vision-tower naming families:
# - GGUF llama.cpp (``v.blk.N.attn_q``, ``v.patch_embed``, ``mm.model.mlp.N``)
# - HF Qwen3-VL / CLIP-style (``vision_model.encoder.layers.N.self_attn.*``)
# - HF Qwen2-VL (``visual.blocks.N.attn.qkv``, ``visual.patch_embed``)
# - Kimi K3 (``vision_tower.encoder.blocks.N.wqkv``, ``vision_tower.patch_embed``)
# Each pattern may capture the vision block index in group 1 (then it becomes a
# row in the vision sheet); patterns without a capture group map to global
# tensors (patch_embed, pos_embed, projector) that land in the "global" row.
# Rules are ordered most-specific first; first match wins.
_VISION_RULES: list[tuple[re.Pattern[str], str]] = [
    # GGUF llama.cpp vision tower (v.blk.N.*)
    (re.compile(r"^v\.blk\.(\d+)\.attn_q\."), "v_attn_q"),
    (re.compile(r"^v\.blk\.(\d+)\.attn_k\."), "v_attn_k"),
    (re.compile(r"^v\.blk\.(\d+)\.attn_v\."), "v_attn_v"),
    (re.compile(r"^v\.blk\.(\d+)\.attn_out\."), "v_attn_o"),
    (re.compile(r"^v\.blk\.(\d+)\.attn_o\."), "v_attn_o"),
    (re.compile(r"^v\.blk\.(\d+)\.ln1\."), "v_attn_norm"),
    (re.compile(r"^v\.blk\.(\d+)\.ln2\."), "v_mlp_norm"),
    (re.compile(r"^v\.blk\.(\d+)\.mlp\.ffn_gate\."), "v_mlp_gate"),
    (re.compile(r"^v\.blk\.(\d+)\.mlp\.ffn_up\."), "v_mlp_up"),
    (re.compile(r"^v\.blk\.(\d+)\.mlp\.ffn_down\."), "v_mlp_down"),
    (re.compile(r"^v\.blk\.(\d+)\.mlp\.fc[12]\."), "v_mlp"),
    (re.compile(r"^v\.blk\.(\d+)\.mlp\."), "v_mlp"),
    (re.compile(r"^v\.blk\.(\d+)\.conv"), "v_conv"),
    (re.compile(r"^v\.blk\.(\d+)\."), "v_other"),
    (re.compile(r"^v\.patch_embed"), "v_patch_embed"),
    (re.compile(r"^v\.pos_embed"), "v_pos_emb"),
    (re.compile(r"^v\.cls"), "v_cls"),
    (re.compile(r"^v\.conv"), "v_conv"),
    (re.compile(r"^v\."), "v_other"),
    # Multimodal projector (GGUF mm.*, HF mm_projector / multi_modal_projector)
    (re.compile(r"mm\.model\.mlp\.\d+"), "mm_projector"),
    (re.compile(r"mm\.\d+"), "mm_projector"),
    (re.compile(r"mm\.input_projection"), "mm_projector"),  # Gemma-4 mmproj
    (re.compile(r"mm_projector"), "mm_projector"),
    (re.compile(r"multi_modal_projector"), "mm_projector"),
    # HF Qwen3-VL / CLIP-style vision_model
    (re.compile(r"vision_model\.encoder\.layers\.(\d+)\.self_attn\.q_proj"), "v_attn_q"),
    (re.compile(r"vision_model\.encoder\.layers\.(\d+)\.self_attn\.k_proj"), "v_attn_k"),
    (re.compile(r"vision_model\.encoder\.layers\.(\d+)\.self_attn\.v_proj"), "v_attn_v"),
    (re.compile(r"vision_model\.encoder\.layers\.(\d+)\.self_attn\.o_proj"), "v_attn_o"),
    (re.compile(r"vision_model\.encoder\.layers\.(\d+)\.self_attn\.qkv"), "v_attn_qkv"),
    (re.compile(r"vision_model\.encoder\.layers\.(\d+)\.layer_norm1"), "v_attn_norm"),
    (re.compile(r"vision_model\.encoder\.layers\.(\d+)\.layer_norm2"), "v_mlp_norm"),
    (re.compile(r"vision_model\.encoder\.layers\.(\d+)\.mlp\.fc[12]"), "v_mlp"),
    (re.compile(r"vision_model\.encoder\.layers\.(\d+)\.mlp\."), "v_mlp"),
    (re.compile(r"vision_model\.encoder\.layers\.(\d+)\."), "v_other"),
    (re.compile(r"vision_model\.embeddings\.patch_embedding"), "v_patch_embed"),
    (re.compile(r"vision_model\.embeddings\.position_embedding"), "v_pos_emb"),
    (re.compile(r"vision_model\.embeddings\.class_embedding"), "v_cls"),
    (re.compile(r"vision_model\.post_layernorm"), "v_mlp_norm"),
    (re.compile(r"vision_model\."), "v_other"),
    # HF Qwen2-VL visual tower
    (re.compile(r"visual\.blocks\.(\d+)\.attn\.qkv"), "v_attn_qkv"),
    (re.compile(r"visual\.blocks\.(\d+)\.attn\.proj"), "v_attn_o"),
    (re.compile(r"visual\.blocks\.(\d+)\.norm1"), "v_attn_norm"),
    (re.compile(r"visual\.blocks\.(\d+)\.norm2"), "v_mlp_norm"),
    (re.compile(r"visual\.blocks\.(\d+)\.mlp\.fc[12]"), "v_mlp"),
    (re.compile(r"visual\.blocks\.(\d+)\.mlp\."), "v_mlp"),
    (re.compile(r"visual\.blocks\.(\d+)\."), "v_other"),
    (re.compile(r"visual\.patch_embed"), "v_patch_embed"),
    (re.compile(r"visual\.rot_pos_emb"), "v_pos_emb"),
    (re.compile(r"visual\.merger\.\d+"), "mm_projector"),
    (re.compile(r"visual\."), "v_other"),
    # Kimi K3 vision tower
    (re.compile(r"vision_tower\.encoder\.blocks\.(\d+)\.wqkv"), "v_attn_qkv"),
    (re.compile(r"vision_tower\.encoder\.blocks\.(\d+)\.wo"), "v_attn_o"),
    (re.compile(r"vision_tower\.encoder\.blocks\.(\d+)\.norm0"), "v_attn_norm"),
    (re.compile(r"vision_tower\.encoder\.blocks\.(\d+)\.norm1"), "v_mlp_norm"),
    (re.compile(r"vision_tower\.encoder\.blocks\.(\d+)\.mlp\.fc[01]"), "v_mlp"),
    (re.compile(r"vision_tower\.encoder\.blocks\.(\d+)\.mlp\."), "v_mlp"),
    (re.compile(r"vision_tower\.encoder\.blocks\.(\d+)\."), "v_other"),
    (re.compile(r"vision_tower\.encoder\.final_layernorm"), "v_mlp_norm"),
    (re.compile(r"vision_tower\.patch_embed\.pos_emb"), "v_pos_emb"),
    (re.compile(r"vision_tower\.patch_embed"), "v_patch_embed"),
    (re.compile(r"vision_tower\."), "v_other"),
]

# Layer index extraction patterns
_LAYER_RE = re.compile(r"layers?\.(\d+)")
_GGUF_LAYER_RE = re.compile(r"blk\.(\d+)")

# Expert ID extraction
_EXPERT_RE = re.compile(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj")


def map_vision(name: str) -> tuple[int | None, str] | None:
    """Map a VLM tensor to ``(vision_block_index, vision_slot)``.

    ``vision_block_index`` is the block index for per-block tensors (e.g.
    ``v.blk.3.attn_q`` → 3) and ``None`` for global tensors (patch_embed,
    pos_embed, multimodal projector). Returns ``None`` for tensors that are
    not part of a vision tower / multimodal projector.
    """
    for pat, slot in _VISION_RULES:
        m = pat.search(name)
        if m:
            block: int | None = None
            if m.lastindex and m.group(1) is not None:
                block = int(m.group(1))
            return block, slot
    return None


def is_vision_tensor(name: str) -> bool:
    """True if the tensor belongs to a vision tower or multimodal projector."""
    return map_vision(name) is not None


def map_name(name: str) -> tuple[int | None, str]:
    """Return (layer_index, slot) for a tensor name.

    ``layer_index`` is ``None`` for non-layer tensors (embed, lm_head) and
    for vision-tower/projector tensors, which are mapped to their vision
    slots (``v_*``/``mm_*``) but never populate the transformer raster.
    Supports HuggingFace, GGUF, MoE, and VLM naming conventions.
    """
    # VLM (vision-language) tensors first: they are non-layer, so the vision
    # tower (v.blk.N.*) never collides with language-model layers in the raster.
    vision = map_vision(name)
    if vision is not None:
        return None, vision[1]

    # Try HuggingFace layer pattern first
    m = _LAYER_RE.search(name)
    if m:
        layer = int(m.group(1))
        # Check MoE rules first (order matters!)
        for pat, moe_slot in _MOE_RULES:
            if pat.search(name):
                if moe_slot is None:
                    # Special handling for shared expert and expert tensors
                    return _handle_moe_hf(name, layer)
                return layer, moe_slot
        # Then regular HF rules
        for pat, slot in _RULES:
            if pat.search(name):
                return layer, slot
        # Then Qwen3-Next hybrid rules
        for pat, slot in _HF_HYBRID_RULES:
            if pat.search(name):
                return layer, slot
        # Then Kimi K3 rules
        for pat, slot in _KIMI_RULES:
            if pat.search(name):
                return layer, slot
        return layer, "other"

    # Try GGUF layer pattern
    m = _GGUF_LAYER_RE.search(name)
    if m:
        layer = int(m.group(1))
        # Check GGUF MoE rules first
        for pat, moe_slot in _GGUF_MOE_RULES:
            if pat.search(name):
                if moe_slot is None:
                    # Expert tensor (3D stacked)
                    return layer, "expert"
                return layer, moe_slot
        # Then regular GGUF rules
        for pat, slot in _GGUF_RULES:
            if pat.search(name):
                return layer, slot
        # Then Qwen3-Next hybrid rules
        for pat, slot in _GGUF_HYBRID_RULES:
            if pat.search(name):
                return layer, slot
        return layer, "other"

    # Non-layer tensors (embed, lm_head)
    for pat, slot in _RULES:
        if pat.search(name):
            return None, slot
    for pat, slot in _HF_HYBRID_RULES:
        if pat.search(name):
            return None, slot
    for pat, slot in _KIMI_RULES:
        if pat.search(name):
            return None, slot
    for pat, slot in _GGUF_RULES:
        if pat.search(name):
            return None, slot
    for pat, slot in _GGUF_HYBRID_RULES:
        if pat.search(name):
            return None, slot
    return None, "other"


def _handle_moe_hf(name: str, layer: int) -> tuple[int, str]:
    """Handle MoE HF tensor names."""
    # Shared expert → mlp slots
    m = re.search(r"shared_expert\.(gate|up|down)_proj", name)
    if m:
        slot_map = {"gate": "mlp_gate", "up": "mlp_up", "down": "mlp_down"}
        return layer, slot_map[m.group(1)]
    # Kimi K3 shared experts (block_sparse_moe.shared_experts.*) → shared_expert
    if re.search(r"block_sparse_moe\.shared_experts", name):
        return layer, "shared_expert"
    # Expert tensors → expert slot (both HF mlp.experts and Kimi block_sparse_moe)
    if re.search(r"mlp\.experts\.\d+\.(gate|up|down)_proj", name):
        return layer, "expert"
    if re.search(r"block_sparse_moe\.experts\.\d+\.w[123]", name):
        return layer, "expert"
    return layer, "other"


def extract_expert_id(name: str) -> int | None:
    """Extract expert ID from tensor name.

    Returns:
        Expert ID if the tensor is an expert tensor, None otherwise.
    """
    # HF: mlp.experts.{e}.(gate|up|down)_proj
    m = re.search(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj", name)
    if m:
        return int(m.group(1))
    # Kimi K3: block_sparse_moe.experts.{e}.w[123]
    m = re.search(r"block_sparse_moe\.experts\.(\d+)\.w[123]", name)
    if m:
        return int(m.group(1))
    # GGUF expert sub-handles: blk.N.ffn_gate_exps.weight[3] → 3
    m = re.search(r"\[(\d+)\]$", name)
    if m:
        return int(m.group(1))
    # GGUF: handled differently (3D stacked tensor name doesn't contain expert ID)
    return None


def is_expert_tensor(name: str) -> bool:
    """Check if a tensor is an MoE expert tensor."""
    # HF: mlp.experts.{e}.(gate|up|down)_proj
    if re.search(r"mlp\.experts\.\d+\.(gate|up|down)_proj", name):
        return True
    # Kimi K3: block_sparse_moe.experts.{e}.w[123]
    if re.search(r"block_sparse_moe\.experts\.\d+\.w[123]", name):
        return True
    # GGUF: ffn_(gate|up|down)_exps
    return bool(re.search(r"blk\.\d+\.ffn_(gate|up|down)_exps", name))


def is_shared_expert(name: str) -> bool:
    """Check if a tensor is a shared expert tensor."""
    if re.search(r"shared_expert", name):
        return True
    # Kimi K3: block_sparse_moe.shared_experts.*
    if re.search(r"block_sparse_moe\.shared_experts", name):
        return True
    return bool(re.search(r"blk\.\d+\.ffn_(gate|up|down)_shexp", name))


def get_moe_slot(name: str) -> str | None:
    """Get the MoE slot (gate/up/down) from an expert tensor name."""
    # HF
    m = re.search(r"mlp\.experts\.\d+\.(gate|up|down)_proj", name)
    if m:
        return m.group(1)
    # Kimi K3: w1=gate, w2=up, w3=down
    m = re.search(r"block_sparse_moe\.experts\.\d+\.w([123])", name)
    if m:
        return {"1": "gate", "2": "up", "3": "down"}[m.group(1)]
    # GGUF
    m = re.search(r"blk\.\d+\.ffn_(gate|up|down)_exps", name)
    if m:
        return m.group(1)
    return None
