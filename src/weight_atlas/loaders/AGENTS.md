# AGENTS.md — loaders

## Purpose

Load model weights from disk into per-slot statistics inputs: safetensors and
GGUF loaders plus GGUF dequantisation (including MXFP4 block formats).

## Ownership

- `base.py` (loader base contract), `safetensors_loader.py`,
  `gguf_loader.py`, `gguf_dequant.py`, `mxfp4.py`.
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
- **Dequant correctness**: GGUF/MXFP4 dequant is numerically pinned by tests
  (`tests/test_gguf.py`, `tests/test_kimi_k3.py`); do not change
  dequantisation without updating those fixtures. Q4_0 uses the canonical
  layout (first 16 values in the low nibbles, last 16 in the high nibbles of
  the 16 qs bytes) — pinned by `test_q4_0_canonical_layout`. Q8_K uses the
  canonical 292-byte block `[f32 d][256 x int8 qs][16 x int16 bsums]` (the
  gguf library does not implement it) — pinned by `TestDequantQ8K`.
- **Block decoders**: vectorized over whole payloads, accept bytes OR uint8
  ndarray payloads (GGUFReader hands over `(rows, block_bytes)` arrays), and
  raise `ValueError` on payloads that are not an exact multiple of the block
  size — never floor-divide away trailing bytes.
- **Safetensors header validation**: header length capped at 512 MB and every
  tensor's `data_offsets` must satisfy `0 <= start <= end <= data_len`
  (`_validate_offsets`) before any payload read.
- **Q4_0 synthesis for fixtures**: `GGUFWriter.add_tensor(raw_dtype=...)`
  only tags the type (writes raw float32 bytes, no quantization). Real
  quantized fixtures use `gguf.quants.quantize(data, qtype)` and pass the
  packed bytes with `raw_shape=list(q.shape)` + `raw_dtype` (see
  `tests/test_paired.py`, `tests/test_gguf.py`).

## Work Guidance

- New formats: implement the loader base, register it, add a mapping coverage
  test (`tests/test_mapping_coverage.py`) and a fixture.

## Verification

- `tests/test_loader.py`, `tests/test_gguf.py`, `tests/test_kimi_k3.py`,
  `tests/test_mapping_coverage.py`. Run via
  `cd /media/data/coding/weight-atlas && .venv/bin/python -m pytest tests/test_loader.py tests/test_gguf.py`.