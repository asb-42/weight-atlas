"""Pure-numpy dequantizer for the compressed-tensors ``mxfp4-pack-quantized`` format.

Mirrors the reference implementation in `compressed-tensors`
(``compressed_tensors/compressors/nvfp4/helpers.py`` and
``compressed_tensors/compressors/mx_utils.py``):

- 4-bit weights are stored as **FP4 E2M1** values (1 sign bit, 2 exponent bits,
  1 mantissa bit), packed two-per-uint8.  Element at column ``2j`` lives in the
  low nibble, element at column ``2j+1`` in the high nibble.
- Per-group scales (group_size=32, along the input dim) are stored as **E8M0**
  biased exponents (uint8): ``scale = 2 ** (byte - 127)``.
- Dequantized weight = FP4 value * per-group scale.

Layout example (Kimi K3 expert ``w1``):
- ``w1.weight_packed`` shape ``(m, k/2)`` uint8
- ``w1.weight_scale``   shape ``(m, k/32)`` uint8
- dequantized weight   shape ``(m, k)`` float32
"""

from __future__ import annotations

from typing import cast

import numpy as np

# FP4 E2M1 magnitude lookup table (index = lower 3 bits of the nibble).
# 0, 0.5, 1, 1.5, 2, 3, 4, 6
_E2M1_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

# Default MX block/group size for MXFP4.
DEFAULT_GROUP_SIZE = 32

# Packed/scale tensor suffixes used by the mxfp4-pack-quantized format.
PACKED_SUFFIX = ".weight_packed"
SCALE_SUFFIX = ".weight_scale"

def is_packed_tensor(name: str) -> bool:
    """True if ``name`` is an MXFP4 ``*.weight_packed`` tensor."""
    return name.endswith(PACKED_SUFFIX)


def weight_name_for_packed(name: str) -> str:
    """Canonical dense weight name for a packed tensor.

    ``...w1.weight_packed`` -> ``...w1.weight`` (drop the ``_packed`` suffix).
    """
    if not name.endswith(PACKED_SUFFIX):
        raise ValueError(f"not an MXFP4 packed tensor name: {name!r}")
    return name[: -len("_packed")]


def scale_name_for_packed(name: str) -> str:
    """Sibling ``*.weight_scale`` tensor name for a packed tensor.

    ``...w1.weight_packed`` -> ``...w1.weight_scale``.
    """
    if not name.endswith(PACKED_SUFFIX):
        raise ValueError(f"not an MXFP4 packed tensor name: {name!r}")
    return name[: -len(PACKED_SUFFIX)] + SCALE_SUFFIX


def unpack_e2m1(packed: np.ndarray) -> np.ndarray:
    """Unpack ``(m, k/2)`` uint8 packed FP4 into ``(m, k)`` float32 E2M1 values.

    Element at column ``2j`` comes from the low nibble of byte ``j``; element
    at column ``2j+1`` from the high nibble (matches
    ``unpack_fp4_from_uint8`` in compressed-tensors).
    """
    packed = np.asarray(packed, dtype=np.uint8)
    m, k2 = packed.shape
    k = k2 * 2
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = np.empty((m, k), dtype=np.uint8)
    nibbles[:, 0::2] = low
    nibbles[:, 1::2] = high
    sign = (nibbles & 0x08) != 0
    magnitude = nibbles & 0x07
    values = _E2M1_LUT[magnitude]
    return cast(np.ndarray, np.where(sign, -values, values))


def e8m0_to_float(scale: np.ndarray) -> np.ndarray:
    """Convert E8M0 biased exponents (uint8) to float power-of-2 scales."""
    scale = np.asarray(scale, dtype=np.uint8)
    return np.power(2.0, scale.astype(np.float64) - 127.0)


def dequantize_mxfp4(
    packed: np.ndarray,
    scale: np.ndarray,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> np.ndarray:
    """Dequantize an MXFP4 packed weight to a float32 dense weight.

    Args:
        packed: ``(m, k/2)`` uint8 tensor of packed FP4 E2M1 values.
        scale: ``(m, k/group_size)`` uint8 tensor of E8M0 scales.
        group_size: number of columns per scale group (MXFP4 uses 32).

    Returns:
        ``(m, k)`` float32 dequantized weight.
    """
    values = unpack_e2m1(packed)
    m, k = values.shape
    scales = e8m0_to_float(scale)
    if scales.shape != (m, k // group_size):
        raise ValueError(
            f"scale shape {scales.shape} does not match packed "
            f"{(m, k)} with group_size={group_size}; expected {(m, k // group_size)}"
        )
    grouped = values.reshape(m, k // group_size, group_size)
    grouped *= scales[:, :, None]
    return grouped.reshape(m, k).astype(np.float32, copy=False)
