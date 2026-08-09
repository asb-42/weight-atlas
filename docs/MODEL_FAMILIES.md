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
