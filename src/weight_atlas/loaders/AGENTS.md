# AGENTS.md — loaders

## Purpose

Load model weights from disk into per-slot statistics inputs: safetensors,
GGUF, and PyTorch loaders plus GGUF dequantisation (including MXFP4 block
formats).

## Ownership

- `base.py` (loader base contract), `safetensors_loader.py`,
  `gguf_loader.py`, `pytorch_loader.py`, `gguf_dequant.py`, `mxfp4.py`,
  `nvfp4.py`.
- Registered with `core.registry.register_loader`; loader ids appear in
  `fingerprint.json` (`loader` field) and drive compare compatibility checks.

## Local Contracts

- **Registry**: every loader registers a string id; duplicate ids raise
  ValueError. `cli.py`/`scan.py` resolve loaders by id or auto-detect.
- **Determinism**: loading + dequantisation must be deterministic (same file
  → same tensors). No parallel-order-dependent aggregation.
- **Name mapping**: tensor-name → slot mapping is centralised in
  `core/name_map.py` (owned by parent doc); loaders must not embed their own
  mapping tables.
- **Dequant correctness**: GGUF/MXFP4/NVFP4 dequant is numerically pinned by
  tests (`tests/test_gguf.py`, `tests/test_kimi_k3.py`,
  `tests/test_nvfp4.py`); do not change
  dequantisation without updating those fixtures. Q4_0 uses the canonical
  layout (first 16 values in the low nibbles, last 16 in the high nibbles of
  the 16 qs bytes) — pinned by `test_q4_0_canonical_layout`. Q8_K uses the
  canonical 292-byte block `[f32 d][256 x int8 qs][16 x int16 bsums]` (the
  gguf library does not implement it) — pinned by `TestDequantQ8K`.
- **Block decoders**: vectorized over whole payloads, accept bytes OR uint8
  ndarray payloads (GGUFReader hands over `(rows, block_bytes)` arrays), and
  raise `ValueError` on payloads that are not an exact multiple of the block
  size — never floor-divide away trailing bytes.
- **IQ family (IQ1/IQ2/IQ3/IQ4)**: delegated to the official gguf library
  (`_GGUF_ONLY`, authoritative) — block layouts are too intricate for
  self-contained decoders. Needed for Unsloth UD dynamic quants
  (Qwen3.8-Flash-Next UD-IQ4_XS mixes IQ4_XS experts with IQ2/IQ3 layers).
- **Plain fixed-width types** (I8/I16/I32/I64/F64): `np.frombuffer` →
  `astype(np.float32)`, no block encoding; empty payloads raise ValueError.
  Pinned by `TestPlainTypes`.
- **Safetensors header validation**: header length capped at 512 MB and every
  tensor's `data_offsets` must satisfy `0 <= start <= end <= data_len`
  (`_validate_offsets`) before any payload read.
- **NVFP4 (safetensors, compressed-tensors `nvfp4-pack-quantized`)**:
  `nvfp4.py` decodes FP4 E2M1 weights (nibble-packed, low nibble = even
  column) x per-group-16 **E4M3 scales stored as full bytes** (tensor
  `weight_scale`, dtype F8_E4M3) x per-tensor fp32 `weight_global_scale`.
  Verified against compressed-tensors v0.18 source (`compressors/nvfp4/
  base.py`); E4M3 decode is a pure-numpy bitfield table, byte-exact vs
  ml_dtypes over all 256 bytes (pinned). The loader discriminates NVFP4
  from MXFP4 by the **global-scale sibling** (both formats share
  `.weight_packed` + `.weight_scale` names; MXFP4 scale is E8M0 uint8,
  group 32). Sibling tensors are consumed in a pre-scan — dict iteration
  order must never leak a stray handle. Merged handle dtype `FP4_NVFP4`.
  Pinned by `tests/test_nvfp4.py` (round-trip + loader E2E + scan E2E +
  MXFP4-still-works).
- **NVFP4 HF naming (Unsloth plefp8)**: same decode, different names —
  packed weights ARE ``<base>.weight`` (U8, 2-D), scales in
  ``<base>.weight_scale`` (F8), global scale in ``<base>.weight_scale_2``
  (F32 scalar). A U8 2-D `.weight` is unambiguous (no dense format uses
  uint8). Per-expert `input_scale` scalars (F32, shape `()`) are
  standalone records — they must NEVER resolve expert machinery
  (`is_expert_tensor`/`get_moe_slot`/`extract_expert_id` are weight-only;
  scale siblings would otherwise collide with weight cells in expert
  panels, last-wins). Standalone F8_E4M3 tensors decode via the nvfp4
  bitfield table in `_from_raw` (a scan must never crash on them).
- **0-D scalar tensors scan**: `to_matrix` maps `()` → `(1, 1)`; 1-D/0-D
  stable_rank = 1.0 with the zero-signal guard first (all-zero → 0.0).
  Real models store scalars (global scales, temperatures); pinned by
  scalar stats tests.
- **Q4_0 synthesis for fixtures**: `GGUFWriter.add_tensor(raw_dtype=...)`
  only tags the type (writes raw float32 bytes, no quantization). Real
  quantized fixtures use `gguf.quants.quantize(data, qtype)` and pass the
  packed bytes with `raw_shape=list(q.shape)` + `raw_dtype` (see
  `tests/test_paired.py`, `tests/test_gguf.py`).
- **PyTorch loader** (`pytorch_loader.py`): pure-python unpickler for `.pt`
  ZIP checkpoints (no torch dependency); reads only model weights (optimizer
  state discarded). BDH-layout checkpoints (cfg `n_head`/`n_embd`/
  `mlp_internal_dim_multiplier` + 3D `encoder`/`encoder_v` + head-major
  `decoder`) are expanded into three granularities: monolithic names,
  per-head `blk.{h}.{name}` (ride the GGUF layer pattern + gguf/base bdh
  rules), and per-lattice-unit `{name}.u{u}.h{h}` with `expert_id=u`
  (unit = `n_embd // n_head` neurons/head, the route lattice). encoder/
  encoder_v unit slices are column slices of the head block (neuron axis
  last); decoder unit slices are contiguous ranges (head-major). Per-unit
  and per-head handles share an instance-level float32 storage cache —
  peak RAM is the full model in float32; deterministic (single-ZIP-read
  per storage). Loader ids in `cli.py` choices come from `list_loaders()`.

## Work Guidance

- New formats: implement the loader base, register it, add a mapping coverage
  test (`tests/test_mapping_coverage.py`) and a fixture.

## Verification

- `tests/test_loader.py`, `tests/test_gguf.py`, `tests/test_kimi_k3.py`,
  `tests/test_mapping_coverage.py`, `tests/test_pytorch_loader.py`. Run via
  `cd /media/data/coding/weight-atlas && .venv/bin/python -m pytest tests/test_loader.py tests/test_gguf.py tests/test_pytorch_loader.py`.