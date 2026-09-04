"""Pure-numpy dequantizer for the compressed-tensors ``nvfp4-pack-quantized`` format.

Mirrors the reference implementation in `compressed-tensors`
(``compressed_tensors/compressors/nvfp4/base.py`` + ``helpers.py``,
v0.18.0 — verified against source):

- 4-bit weights are stored as **FP4 E2M1** values, packed two-per-uint8:
  element at column ``2j`` in the LOW nibble, element ``2j+1`` in the HIGH
  nibble (``pack_fp4_to_uint8``: ``packed = idx0 | (idx1 << 4)``).
- Per-group scales are **FP8 E4M3** — stored as the tensor
  ``weight_scale`` with dtype F8_E4M3 (one FULL byte per group, 8 bits —
  NOT nibble-packed), group_size=**16** along the input dim.
- A per-tensor ``weight_global_scale`` (scalar fp32) is applied on top:
  ``dequant = fp4_value * e4m3(weight_scale) * global_scale``
  (``base.py: dequantize(x_q, scale_float, global_scale, ...)``).

Layout example:
- ``model.layer.weight_packed``      shape ``(m, k/2)``   uint8 (FP4 pairs)
- ``model.layer.weight_scale``       shape ``(m, k/16)``   F8_E4M3 bytes
- ``model.layer.weight_global_scale``      scalar fp32
- dequantized weight shape ``(m, k)`` float32

No ml_dtypes dependency at decode time: the E4M3 decode is an explicit
bitfield table (byte-exact vs ml_dtypes, pinned by test).
"""

from __future__ import annotations

import numpy as np

from weight_atlas.loaders.mxfp4 import unpack_e2m1

# Packed/scale tensor suffixes used by the nvfp4-pack-quantized format.
PACKED_SUFFIX = ".weight_packed"
SCALE_SUFFIX = ".weight_scale"
GLOBAL_SCALE_SUFFIX = ".weight_global_scale"

# NVFP4 group size (weights per E4M3 scale group).
NVFP4_GROUP_SIZE = 16


# ── E4M3 decode (explicit bitfields — no float8 dtype dependency) ───────────
# E4M3: 1 sign, 4 exp (bias 7), 3 mantissa. Finite max = 448. The only NaN
# encodings are 0x7F/0xFF (S=1, E=15, M=7); every other byte is finite
# (E4M3 has no infinities). Decoded exactly in fp64.
_E4M3_MAGNITUDE = np.array(
    [
        (float(m) / 8.0 + 1.0) * 2.0 ** (e - 7) if e > 0 else
        float(m) / 8.0 * 2.0 ** -6
        for e in range(16)
        for m in range(8)
    ],
    dtype=np.float64,
)  # index = exponent*8 + mantissa
_E4M3_MAGNITUDE[0x7F] = np.nan  # magnitude 0x7F = the NaN encoding


def e4m3_to_float(scale: np.ndarray) -> np.ndarray:
    """Decode uint8 E4M3 bytes to float64 (NaN for 0x7F/0xFF, sign preserved)."""
    scale = np.asarray(scale, dtype=np.uint8)
    sign = np.where(scale & 0x80, -1.0, 1.0)
    magnitude_index = (scale & 0x7F).astype(np.int64)
    return sign * _E4M3_MAGNITUDE[magnitude_index]


def is_nvfp4_packed_tensor(name: str) -> bool:
    """True if ``name`` is an NVFP4 ``*.weight_packed`` tensor."""
    return name.endswith(PACKED_SUFFIX)


def weight_name_for_packed(name: str) -> str:
    """Canonical dense weight name: ``...w.weight_packed`` -> ``...w.weight``."""
    if not name.endswith(PACKED_SUFFIX):
        raise ValueError(f"not an NVFP4 packed tensor name: {name!r}")
    return name[: -len("_packed")]


def scale_name_for_packed(name: str) -> str:
    """``...w.weight_packed`` -> ``...w.weight_scale``."""
    if not name.endswith(PACKED_SUFFIX):
        raise ValueError(f"not an NVFP4 packed tensor name: {name!r}")
    return name[: -len(PACKED_SUFFIX)] + SCALE_SUFFIX


def global_scale_name_for_packed(name: str) -> str:
    """``...w.weight_packed`` -> ``...w.weight_global_scale``."""
    if not name.endswith(PACKED_SUFFIX):
        raise ValueError(f"not an NVFP4 packed tensor name: {name!r}")
    return name[: -len(PACKED_SUFFIX)] + GLOBAL_SCALE_SUFFIX


def dequantize_nvfp4(
    packed: np.ndarray,
    scale: np.ndarray,
    global_scale: np.ndarray | float,
    group_size: int = NVFP4_GROUP_SIZE,
) -> np.ndarray:
    """Dequantize an NVFP4 packed weight to a float32 dense weight.

    Args:
        packed: ``(m, k/2)`` uint8 tensor of packed FP4 E2M1 values.
        scale: ``(m, k/group_size)`` uint8 tensor of full E4M3 scale bytes
            (dtype F8_E4M3 in the file — read as raw uint8).
        global_scale: scalar (or 1-element array) fp32 per-tensor scale.
        group_size: weights per scale group (NVFP4 uses 16).

    Returns:
        ``(m, k)`` float32 dequantized weight.
    """
    values = unpack_e2m1(packed)  # (m, k) fp32 E2M1
    m, k = values.shape
    if k % group_size != 0:
        raise ValueError(
            f"input dim {k} not a multiple of group_size {group_size}"
        )
    n_groups = k // group_size
    scale = np.asarray(scale)
    if scale.shape != (m, n_groups):
        raise ValueError(
            f"scale shape {scale.shape} does not match packed "
            f"{(m, k)} with group_size={group_size}; expected {(m, n_groups)}"
        )
    gs = float(np.asarray(global_scale).reshape(-1)[0])
    scales = e4m3_to_float(scale) * gs  # (m, n_groups)
    grouped = values.reshape(m, n_groups, group_size)
    grouped *= scales[:, :, None]
    return grouped.reshape(m, k).astype(np.float32, copy=False)
