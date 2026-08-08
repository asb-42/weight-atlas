"""GGUF dequantization: convert quantized tensors to canonical float32."""

from __future__ import annotations

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

# Supported types for M5
SUPPORTED_TYPES = {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
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
}


def get_type_name(ggml_type: int) -> str:
    """Get human-readable name for a GGML type."""
    return TYPE_NAMES.get(ggml_type, f"UNKNOWN({ggml_type})")


def check_supported(ggml_type: int) -> None:
    """Raise ValueError if the type is not supported in M5 scope."""
    if ggml_type not in SUPPORTED_TYPES:
        name = get_type_name(ggml_type)
        raise ValueError(
            f"Unsupported GGUF quantization type: {name}. "
            f"M5 supports: F32, F16, BF16, Q8_0, Q4_0. "
            f"Full k-quant support is on the backlog."
        )


def dequantize(tensor_data: bytes, ggml_type: int) -> np.ndarray:
    """Dequantize a tensor to float32.

    Args:
        tensor_data: raw bytes of the tensor
        ggml_type: GGML quantization type ID

    Returns:
        Dequantized tensor as float32 numpy array
    """
    check_supported(ggml_type)

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

    # Should not reach here due to check_supported
    name = get_type_name(ggml_type)
    raise ValueError(f"Unsupported GGUF quantization type: {name}")


def _dequant_f32(data: bytes) -> np.ndarray:
    """F32: direct reinterpretation."""
    return np.frombuffer(data, dtype=np.float32).copy()


def _dequant_f16(data: bytes) -> np.ndarray:
    """F16: convert half-precision to float32."""
    return np.frombuffer(data, dtype=np.float16).astype(np.float32).copy()


def _dequant_bf16(data: bytes) -> np.ndarray:
    """BF16: bit-shift to float32 (uint16 → uint32 view)."""
    # BF16 is the lower 16 bits of a float32 (with truncated mantissa)
    bf16_arr = np.frombuffer(data, dtype=np.uint16)
    # Shift left by 16 bits to get float32 representation
    f32_arr = bf16_arr.astype(np.uint32) << 16
    return f32_arr.view(np.float32).copy()


def _dequant_q8_0(data: bytes) -> np.ndarray:
    """Q8_0: block-wise dequantization (32-element blocks, f16 scale).

    Block layout: [scale: f16] [quants: 32 x int8]
    """
    # Each block is 2 bytes (scale) + 32 bytes (quants) = 34 bytes
    block_size = 34
    n_blocks = len(data) // block_size

    result = np.empty(n_blocks * 32, dtype=np.float32)
    for i in range(n_blocks):
        offset = i * block_size
        # Read f16 scale
        scale = np.frombuffer(data[offset:offset + 2], dtype=np.float16)[0].astype(np.float32)
        # Read 32 int8 values
        quants = np.frombuffer(data[offset + 2:offset + 32 + 2], dtype=np.int8)
        # Dequantize
        result[i * 32:(i + 1) * 32] = quants.astype(np.float32) * scale

    return result


def _dequant_q4_0(data: bytes) -> np.ndarray:
    """Q4_0: block-wise dequantization (32-element blocks, f16 scale, 4-bit quants).

    Block layout: [scale: f16] [quants: 16 bytes (32 x 4-bit packed)]
    Each 4-bit value is in [-8, 7], stored as nibbles (low nibble first).
    """
    # Each block is 2 bytes (scale) + 16 bytes (32 x 4-bit) = 18 bytes
    block_size = 18
    n_blocks = len(data) // block_size

    result = np.empty(n_blocks * 32, dtype=np.float32)
    for i in range(n_blocks):
        offset = i * block_size
        # Read f16 scale
        scale = np.frombuffer(data[offset:offset + 2], dtype=np.float16)[0].astype(np.float32)
        # Read 16 bytes, each containing two 4-bit values
        packed = np.frombuffer(data[offset + 2:offset + 16 + 2], dtype=np.uint8)
        # Unpack nibbles: low nibble first, then high nibble
        quants = np.empty(32, dtype=np.float32)
        for j in range(16):
            low = packed[j] & 0x0F
            high = (packed[j] >> 4) & 0x0F
            # Convert from unsigned to signed [-8, 7]
            quants[j * 2] = float(low) - 8.0
            quants[j * 2 + 1] = float(high) - 8.0
        # Dequantize
        result[i * 32:(i + 1) * 32] = quants * scale

    return result
