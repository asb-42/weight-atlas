"""GGUF dequantization: convert quantized tensors to canonical float32.

Uses the official gguf library for dequantization when available,
with fallback implementations for supported types.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

# GGUF quantization type IDs
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23
GGML_TYPE_I8 = 24
GGML_TYPE_I16 = 25
GGML_TYPE_I32 = 26
GGML_TYPE_I64 = 27
GGML_TYPE_F64 = 28
GGML_TYPE_IQ1_M = 29
GGML_TYPE_BF16 = 30
GGML_TYPE_Q4_0_4_4 = 31
GGML_TYPE_Q4_0_4_8 = 32
GGML_TYPE_Q4_0_8_8 = 33
GGML_TYPE_TQ1_0 = 34
GGML_TYPE_TQ2_0 = 35
GGML_TYPE_MXFP4 = 39
GGML_TYPE_NVFP4 = 40
GGML_TYPE_Q1_0 = 41

# K-quant block size (256 weights per block)
QK_K = 256

# Supported types (M5 + k-quants + common types)
SUPPORTED_TYPES = {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_1,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_K,
    GGML_TYPE_TQ1_0,
    GGML_TYPE_TQ2_0,
    GGML_TYPE_MXFP4,
    GGML_TYPE_NVFP4,
    GGML_TYPE_Q1_0,
}

# Type names for error messages
TYPE_NAMES = {
    GGML_TYPE_F32: "F32",
    GGML_TYPE_F16: "F16",
    GGML_TYPE_Q4_0: "Q4_0",
    GGML_TYPE_Q4_1: "Q4_1",
    GGML_TYPE_Q5_0: "Q5_0",
    GGML_TYPE_Q5_1: "Q5_1",
    GGML_TYPE_Q8_0: "Q8_0",
    GGML_TYPE_Q8_1: "Q8_1",
    GGML_TYPE_Q2_K: "Q2_K",
    GGML_TYPE_Q3_K: "Q3_K",
    GGML_TYPE_Q4_K: "Q4_K",
    GGML_TYPE_Q5_K: "Q5_K",
    GGML_TYPE_Q6_K: "Q6_K",
    GGML_TYPE_Q8_K: "Q8_K",
    GGML_TYPE_IQ2_XXS: "IQ2_XXS",
    GGML_TYPE_IQ2_XS: "IQ2_XS",
    GGML_TYPE_IQ3_XXS: "IQ3_XXS",
    GGML_TYPE_IQ1_S: "IQ1_S",
    GGML_TYPE_IQ4_NL: "IQ4_NL",
    GGML_TYPE_IQ3_S: "IQ3_S",
    GGML_TYPE_IQ2_S: "IQ2_S",
    GGML_TYPE_IQ4_XS: "IQ4_XS",
    GGML_TYPE_I8: "I8",
    GGML_TYPE_I16: "I16",
    GGML_TYPE_I32: "I32",
    GGML_TYPE_I64: "I64",
    GGML_TYPE_F64: "F64",
    GGML_TYPE_IQ1_M: "IQ1_M",
    GGML_TYPE_BF16: "BF16",
    GGML_TYPE_Q4_0_4_4: "Q4_0_4_4",
    GGML_TYPE_Q4_0_4_8: "Q4_0_4_8",
    GGML_TYPE_Q4_0_8_8: "Q4_0_8_8",
    GGML_TYPE_TQ1_0: "TQ1_0",
    GGML_TYPE_TQ2_0: "TQ2_0",
    GGML_TYPE_MXFP4: "MXFP4",
    GGML_TYPE_NVFP4: "NVFP4",
    GGML_TYPE_Q1_0: "Q1_0",
}


def get_type_name(ggml_type: int) -> str:
    """Get human-readable name for a GGML type."""
    return TYPE_NAMES.get(ggml_type, f"UNKNOWN({ggml_type})")


def check_supported(ggml_type: int) -> None:
    """Raise ValueError if the type is not supported."""
    if ggml_type not in SUPPORTED_TYPES:
        name = get_type_name(ggml_type)
        supported = sorted([TYPE_NAMES[t] for t in SUPPORTED_TYPES])
        raise ValueError(
            f"Unsupported GGUF quantization type: {name}. "
            f"Supported types: {', '.join(supported)}. "
            f"Full support for IQ variants is on the backlog."
        )


# Optional official gguf library — imported once at module load. Used only for
# quant types without a self-contained pure-numpy implementation. None when the
# package is not installed.
try:
    import gguf

    _gguf: ModuleType | None = gguf
except ImportError:  # pragma: no cover - optional dependency
    _gguf = None

# Quant types that require the official gguf library (no pure-numpy fallback).
_GGUF_ONLY = {
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_1,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_TQ1_0,
    GGML_TYPE_TQ2_0,
    GGML_TYPE_MXFP4,
    GGML_TYPE_NVFP4,
}


def dequantize(tensor_data: bytes, ggml_type: int) -> np.ndarray:
    """Dequantize a tensor to float32.

    Uses the official gguf library for quant types that need it (``_GGUF_ONLY``)
    and self-contained pure-numpy implementations for the rest (F32/F16/BF16/
    Q8_0/Q4_0/Q8_K/Q1_0), so the module works without the ``gguf`` package.
    Genuine dequantization errors are re-raised, never silently swallowed.
    """
    check_supported(ggml_type)

    if ggml_type in _GGUF_ONLY:
        if _gguf is None:
            name = get_type_name(ggml_type)
            raise ImportError(
                f"GGUF quantization type {name} requires the 'gguf' package. "
                "Install it with: pip install gguf"
            )
        return _dequant_with_gguf(tensor_data, ggml_type)

    # Self-contained pure-numpy path (never requires gguf).
    if ggml_type == GGML_TYPE_F32:
        return _dequant_f32(tensor_data)
    if ggml_type == GGML_TYPE_F16:
        return _dequant_f16(tensor_data)
    if ggml_type == GGML_TYPE_BF16:
        return _dequant_bf16(tensor_data)
    if ggml_type == GGML_TYPE_Q8_0:
        return _dequant_q8_0(tensor_data)
    if ggml_type == GGML_TYPE_Q4_0:
        return _dequant_q4_0(tensor_data)
    if ggml_type == GGML_TYPE_Q8_K:
        return _dequant_q8_k(tensor_data)
    if ggml_type == GGML_TYPE_Q1_0:
        return _dequant_q1_0(tensor_data)

    name = get_type_name(ggml_type)
    raise ValueError(f"Unsupported GGUF quantization type: {name}")


def _dequant_with_gguf(tensor_data: bytes, ggml_type: int) -> np.ndarray:
    """Dequantize via the official gguf library.

    Block size is taken from ``gguf.GGML_QUANT_SIZES`` (authoritative) rather
    than hardcoded per-type constants, so a gguf library upgrade cannot silently
    change the layout. Raises if the type is not implemented by gguf.
    """
    assert _gguf is not None
    qtype = _gguf.GGMLQuantizationType(ggml_type)
    quant_sizes = _gguf.GGML_QUANT_SIZES.get(qtype)
    if quant_sizes is None:
        raise ValueError(
            f"gguf library has no block layout for {get_type_name(ggml_type)}"
        )
    _, block_bytes = quant_sizes
    arr = np.frombuffer(tensor_data, dtype=np.uint8)
    try:
        if block_bytes > 1:
            blocks = arr.reshape(-1, block_bytes)
            dequantized = _gguf.dequantize(blocks, qtype)
        else:
            dequantized = _gguf.dequantize(arr, qtype)
        return np.ascontiguousarray(dequantized, dtype=np.float32).flatten()
    except NotImplementedError as exc:
        # The installed gguf library may not implement every quant type; surface
        # a clear error instead of silently producing garbage.
        raise ValueError(
            f"gguf library does not implement dequantization for "
            f"{get_type_name(ggml_type)}"
        ) from exc


def _dequant_q1_0(data: bytes) -> np.ndarray:
    """Q1_0: 1-bit quantization (128 weights per block, 18 bytes).

    Block layout: [scale: f16] [qs: 16 bytes = 128 x 1-bit]
    Dequantization: value = scale * (2 * quant - 1)
    """
    block_size = 18
    n_blocks = len(data) // block_size
    result = np.empty(n_blocks * 128, dtype=np.float32)

    # Vectorized bit unpacking
    data_arr = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_size)
    scales = np.frombuffer(data_arr[:, :2].tobytes(), dtype=np.float16).astype(np.float32)
    qs = data_arr[:, 2:]  # (n_blocks, 16)

    # Unpack bits using broadcasting
    bits = np.unpackbits(qs, axis=1)  # (n_blocks, 128)
    quants = bits.astype(np.float32)

    # Apply scales
    result = (scales[:, np.newaxis] * (2.0 * quants - 1.0)).flatten()
    return result


def _dequant_f32(data: bytes) -> np.ndarray:
    """F32: direct reinterpretation."""
    return np.frombuffer(data, dtype=np.float32).copy()


def _dequant_f16(data: bytes) -> np.ndarray:
    """F16: convert half-precision to float32."""
    return np.frombuffer(data, dtype=np.float16).astype(np.float32).copy()


def _dequant_bf16(data: bytes) -> np.ndarray:
    """BF16: bit-shift to float32 (uint16 → uint32 view)."""
    bf16_arr = np.frombuffer(data, dtype=np.uint16)
    f32_arr = bf16_arr.astype(np.uint32) << 16
    return f32_arr.view(np.float32).copy()


def _dequant_q8_0(data: bytes) -> np.ndarray:
    """Q8_0: block-wise dequantization (32-element blocks, f16 scale)."""
    block_size = 34
    n_blocks = len(data) // block_size
    result = np.empty(n_blocks * 32, dtype=np.float32)
    for i in range(n_blocks):
        offset = i * block_size
        scale = np.frombuffer(data[offset:offset + 2], dtype=np.float16)[0].astype(np.float32)
        quants = np.frombuffer(data[offset + 2:offset + 34], dtype=np.int8)
        result[i * 32:(i + 1) * 32] = quants.astype(np.float32) * scale
    return result


def _dequant_q4_0(data: bytes) -> np.ndarray:
    """Q4_0: block-wise dequantization (32-element blocks, f16 scale, 4-bit quants)."""
    block_size = 18
    n_blocks = len(data) // block_size
    result = np.empty(n_blocks * 32, dtype=np.float32)
    for i in range(n_blocks):
        offset = i * block_size
        scale = np.frombuffer(data[offset:offset + 2], dtype=np.float16)[0].astype(np.float32)
        packed = np.frombuffer(data[offset + 2:offset + 18], dtype=np.uint8)
        quants = np.empty(32, dtype=np.float32)
        for j in range(16):
            low = packed[j] & 0x0F
            high = (packed[j] >> 4) & 0x0F
            quants[j * 2] = float(low) - 8.0
            quants[j * 2 + 1] = float(high) - 8.0
        result[i * 32:(i + 1) * 32] = quants * scale
    return result


def _dequant_q8_k(data: bytes) -> np.ndarray:
    """Q8_K: block-wise dequantization (256 weights per block, 258 bytes).

    Block layout: [scale: f16] [quants: 256 x int8]
    Note: gguf library doesn't implement Q8_K, so we do it manually.
    """
    block_size = 258  # 2 bytes scale + 256 bytes quants
    n_blocks = len(data) // block_size
    result = np.empty(n_blocks * 256, dtype=np.float32)
    for i in range(n_blocks):
        offset = i * block_size
        scale = np.frombuffer(data[offset:offset + 2], dtype=np.float16)[0].astype(np.float32)
        quants = np.frombuffer(data[offset + 2:offset + 258], dtype=np.int8)
        result[i * 256:(i + 1) * 256] = quants.astype(np.float32) * scale
    return result
