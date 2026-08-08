"""Tensor name → (layer_index, slot) mapping for model families.

Convention: regex rules applied in order; first match wins. Unknown names
map to slot ``"other"`` (never silently dropped).

Supports three naming conventions:
- HuggingFace: model.layers.N.self_attn.q_proj.weight
- GGUF: blk.N.attn_q.weight
- MoE (HF): model.layers.N.mlp.experts.{e}.gate_proj.weight
"""

from __future__ import annotations

import re

# Ordered regex rules: (pattern, slot). First match wins.
# HuggingFace rules (original)
_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"embed_tokens"), "embed"),
    (re.compile(r"lm_head"), "lm_head"),
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
]

# GGUF-specific rules (applied after HF rules, before fallback)
_GGUF_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"blk\.\d+\.attn_q"), "attn_q"),
    (re.compile(r"blk\.\d+\.attn_k"), "attn_k"),
    (re.compile(r"blk\.\d+\.attn_v"), "attn_v"),
    (re.compile(r"blk\.\d+\.attn_output"), "attn_o"),
    (re.compile(r"blk\.\d+\.ffn_gate"), "mlp_gate"),
    (re.compile(r"blk\.\d+\.ffn_up"), "mlp_up"),
    (re.compile(r"blk\.\d+\.ffn_down"), "mlp_down"),
    (re.compile(r"blk\.\d+\.attn_norm"), "norm_attn"),
    (re.compile(r"blk\.\d+\.ffn_norm"), "norm_mlp"),
    (re.compile(r"token_embd"), "embed"),
    (re.compile(r"output\.weight"), "lm_head"),
    (re.compile(r"output_norm"), "norm_mlp"),
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

# Layer index extraction patterns
_LAYER_RE = re.compile(r"layers?\.(\d+)")
_GGUF_LAYER_RE = re.compile(r"blk\.(\d+)")

# Expert ID extraction
_EXPERT_RE = re.compile(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj")


def map_name(name: str) -> tuple[int | None, str]:
    """Return (layer_index, slot) for a tensor name.

    ``layer_index`` is ``None`` for non-layer tensors (embed, lm_head).
    Supports HuggingFace, GGUF, and MoE naming conventions.
    """
    # Try HuggingFace layer pattern first
    m = _LAYER_RE.search(name)
    if m:
        layer = int(m.group(1))
        # Check MoE rules first (order matters!)
        for pat, slot in _MOE_RULES:
            if pat.search(name):
                if slot is None:
                    # Special handling for shared expert and expert tensors
                    return _handle_moe_hf(name, layer)
                return layer, slot
        # Then regular HF rules
        for pat, slot in _RULES:
            if pat.search(name):
                return layer, slot
        return layer, "other"

    # Try GGUF layer pattern
    m = _GGUF_LAYER_RE.search(name)
    if m:
        layer = int(m.group(1))
        # Check GGUF MoE rules first
        for pat, slot in _GGUF_MOE_RULES:
            if pat.search(name):
                if slot is None:
                    # Expert tensor (3D stacked)
                    return layer, "expert"
                return layer, slot
        # Then regular GGUF rules
        for pat, slot in _GGUF_RULES:
            if pat.search(name):
                return layer, slot
        return layer, "other"

    # Non-layer tensors (embed, lm_head)
    for pat, slot in _RULES:
        if pat.search(name):
            return None, slot
    for pat, slot in _GGUF_RULES:
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
    # Expert tensors → expert slot
    if re.search(r"mlp\.experts\.\d+\.(gate|up|down)_proj", name):
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
    # GGUF: handled differently (3D stacked tensor name doesn't contain expert ID)
    return None


def is_expert_tensor(name: str) -> bool:
    """Check if a tensor is an MoE expert tensor."""
    # HF: mlp.experts.{e}.(gate|up|down)_proj
    if re.search(r"mlp\.experts\.\d+\.(gate|up|down)_proj", name):
        return True
    # GGUF: ffn_(gate|up|down)_exps
    return bool(re.search(r"blk\.\d+\.ffn_(gate|up|down)_exps", name))


def is_shared_expert(name: str) -> bool:
    """Check if a tensor is a shared expert tensor."""
    if re.search(r"shared_expert", name):
        return True
    return bool(re.search(r"blk\.\d+\.ffn_(gate|up|down)_shexp", name))


def get_moe_slot(name: str) -> str | None:
    """Get the MoE slot (gate/up/down) from an expert tensor name."""
    # HF
    m = re.search(r"mlp\.experts\.\d+\.(gate|up|down)_proj", name)
    if m:
        return m.group(1)
    # GGUF
    m = re.search(r"blk\.\d+\.ffn_(gate|up|down)_exps", name)
    if m:
        return m.group(1)
    return None
