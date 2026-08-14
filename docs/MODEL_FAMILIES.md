# Model Families

This document describes known model families and their tensor naming conventions as verified by the name audit system.

## Bonsai-8B

**Architecture**: Transformer with QK-Norm  
**Convention**: HuggingFace (safetensors) + GGUF  
**Layers**: 32  
**Hidden Size**: 4096  
**Heads**: 32 (head_dim=128)  
**Vocab**: 32000  
**Intermediate Size**: 11008  

### Tensor Names (HuggingFace)

| Tensor Name | Shape | Slot |
|-------------|-------|------|
| `model.embed_tokens.weight` | [32000, 4096] | embed |
| `model.layers.N.self_attn.q_proj.weight` | [4096, 4096] | attn_q |
| `model.layers.N.self_attn.k_proj.weight` | [4096, 4096] | attn_k |
| `model.layers.N.self_attn.v_proj.weight` | [4096, 4096] | attn_v |
| `model.layers.N.self_attn.o_proj.weight` | [4096, 4096] | attn_o |
| `model.layers.N.self_attn.q_norm.weight` | [128] | attn_q_norm |
| `model.layers.N.self_attn.k_norm.weight` | [128] | attn_k_norm |
| `model.layers.N.mlp.gate_proj.weight` | [11008, 4096] | mlp_gate |
| `model.layers.N.mlp.up_proj.weight` | [11008, 4096] | mlp_up |
| `model.layers.N.mlp.down_proj.weight` | [4096, 11008] | mlp_down |
| `model.layers.N.input_layernorm.weight` | [4096] | norm_attn |
| `model.layers.N.post_attention_layernorm.weight` | [4096] | norm_mlp |
| `model.norm.weight` | [4096] | norm_mlp |
| `lm_head.weight` | [32000, 4096] | lm_head |

### Tensor Names (GGUF)

| Tensor Name | Shape | Slot |
|-------------|-------|------|
| `token_embd.weight` | [32000, 4096] | embed |
| `blk.N.attn_q.weight` | [4096, 4096] | attn_q |
| `blk.N.attn_k.weight` | [4096, 4096] | attn_k |
| `blk.N.attn_v.weight` | [4096, 4096] | attn_v |
| `blk.N.attn_output.weight` | [4096, 4096] | attn_o |
| `blk.N.attn_q_norm.weight` | [128] | attn_q_norm |
| `blk.N.attn_k_norm.weight` | [128] | attn_k_norm |
| `blk.N.ffn_gate.weight` | [11008, 4096] | mlp_gate |
| `blk.N.ffn_up.weight` | [11008, 4096] | mlp_up |
| `blk.N.ffn_down.weight` | [4096, 11008] | mlp_down |
| `blk.N.attn_norm.weight` | [4096] | norm_attn |
| `blk.N.ffn_norm.weight` | [4096] | norm_mlp |
| `output_norm.weight` | [4096] | norm_mlp |
| `output.weight` | [32000, 4096] | lm_head |

### Special Features

- **QK-Norm**: Bonsai-8B applies normalization to query and key vectors after projection, using dedicated `q_norm` and `k_norm` tensors (head_dim=128). These are 1-D tensors.

### Mapping Coverage

Verified by `tests/test_name_audit.py` using `tests/fixtures/names_bonsai_8b.json`.

## Gemma-4-26B-A4B (ultra/heretic, MoE)

**Architecture**: Gemma-4 MoE with extra per-layer normalization and output scaling
**Convention**: GGUF
**Layers**: 30
**MoE**: Yes (router + `ffn_*_exps` expert tensors)
**Vision**: Separate mmproj GGUF (`mm.input_projection` + `v.blk.N.*` vision tower)

### Tensor Names (GGUF, per layer `N`)

| Tensor Name | Slot |
|-------------|------|
| `token_embd.weight` | embed |
| `rope_freqs.weight` | rope_freqs |
| `blk.N.attn_q.weight` | attn_q |
| `blk.N.attn_k.weight` | attn_k |
| `blk.N.attn_v.weight` | attn_v |
| `blk.N.attn_output.weight` | attn_o |
| `blk.N.attn_q_norm.weight` | attn_q_norm |
| `blk.N.attn_k_norm.weight` | attn_k_norm |
| `blk.N.attn_norm.weight` | norm_attn |
| `blk.N.ffn_gate_inp.weight` | router |
| `blk.N.ffn_gate_exps.weight` / `ffn_up_exps` / `ffn_down_exps` | expert |
| `blk.N.ffn_gate.weight` / `ffn_up` / `ffn_down` | mlp_gate / mlp_up / mlp_down |
| `blk.N.ffn_norm.weight` | norm_mlp |
| `blk.N.post_attention_norm.weight` | norm_mlp |
| `blk.N.pre_ffw_norm_2.weight` | pre_ffw_norm_2 |
| `blk.N.post_ffw_norm.weight` | post_ffw_norm |
| `blk.N.post_ffw_norm_1.weight` | post_ffw_norm_1 |
| `blk.N.post_ffw_norm_2.weight` | post_ffw_norm_2 |
| `blk.N.layer_output_scale.weight` | layer_output_scale |
| `output_norm.weight` | norm_mlp |
| `output.weight` | lm_head |

### Tensor Names (mmproj GGUF)

| Tensor Name | Slot |
|-------------|------|
| `mm.input_projection.weight` | mm_projector |
| `v.blk.N.attn_q.weight` | v_attn_q |
| `v.blk.N.attn_k.weight` | v_attn_k |
| `v.blk.N.attn_v.weight` | v_attn_v |
| `v.blk.N.attn_out.weight` | v_attn_o |
| `v.blk.N.ln1.weight` | v_attn_norm |
| `v.blk.N.ln2.weight` | v_mlp_norm |

### Special Features

- **Extra FFW norms**: the "ultra/heretic" variant adds pre/post feed-forward
  layernorms beyond the standard `ffn_norm`/`post_attention_norm`, including
  numbered variants (`post_ffw_norm_1`, `post_ffw_norm_2`). Each is a distinct
  per-layer tensor, so they map to dedicated slots rather than collapsing into
  `norm_mlp`.
- **Layer output scale**: `blk.N.layer_output_scale.weight` is a per-layer gain
  on the block output; it gets its own slot.
- **RoPE frequency table**: `rope_freqs.weight` is a global (non-layer) tensor,
  mapped to its own global slot.

### Mapping Coverage

Verified by `tests/test_name_audit.py` using `tests/fixtures/names_gemma4_heretic_gguf.json`.

## Adding a New Model Family

1. Create a JSON fixture in `tests/fixtures/names_<family>.json` with:
   - `tensor_names`: list of HF tensor names
   - `gguf_tensor_names`: list of GGUF tensor names
   - `expected_mapping`: expected (layer, slot) for HF names
   - `gguf_expected_mapping`: expected (layer, slot) for GGUF names
2. Add tests in `tests/test_name_audit.py` (or a new test file)
3. Run `weight-atlas diagnose <model_path>` to verify
4. Document the family in this file

## Known Families

| Family | Slots | QK-Norm | MoE | Fixture |
|--------|-------|---------|-----|---------|
| Bonsai-8B | 15 | Yes | No | `names_bonsai_8b.json` |
| LLaMA-2/3 | 13 | No | No | (built-in) |
| Mixtral | 13 | No | Yes | (built-in) |
| DeepSeek | 13 | No | Yes | (built-in) |
| Gemma-4-26B-A4B (ultra/heretic) | 24+ | Yes | Yes | `names_gemma4_heretic_gguf.json` |
