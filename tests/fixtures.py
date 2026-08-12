"""Seeded fake-model fixtures for deterministic testing."""

from __future__ import annotations

import numpy as np
from safetensors.numpy import save_file

SLOTS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "input_layernorm",
    "post_attention_layernorm",
]


def make_fake_model(path, n_layers: int = 4, hidden: int = 32, seed: int = 42):
    """Write a fake safetensors model with known structure."""
    rng = np.random.default_rng(seed)
    tensors = {}
    for layer in range(n_layers):
        for slot in SLOTS:
            # scale per slot so stats differ between slots
            scale = (SLOTS.index(slot) + 1) * 0.1
            tensors[f"model.layers.{layer}.{slot}.weight"] = rng.normal(
                0.0, scale, size=(hidden, hidden)
            ).astype(np.float32)
    tensors["model.embed_tokens.weight"] = rng.normal(0, 0.1, (hidden, hidden)).astype(np.float32)
    tensors["lm_head.weight"] = rng.normal(0, 0.1, (hidden, hidden)).astype(np.float32)
    save_file(tensors, str(path))
    return tensors


def make_diag_tensor(path, values):
    """Write a single tensor (2-D diagonal) for hand-computed stat tests."""
    arr = np.diag(values).astype(np.float32)
    save_file({"t": arr}, str(path))
    return arr
