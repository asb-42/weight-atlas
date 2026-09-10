"""EXL3 dequantization: packed trellis + procedural codebook -> float32 weights.

Pure-NumPy transcription of the reference CUDA implementation in exllamav3
(https://github.com/turboderp-org/exllamav3), specifically:

- ``exllamav3_ext/quant/pack.cu``       trellis bit packing (SWAP16 word pairs)
- ``exllamav3_ext/quant/exl3_dq.cuh``   16-bit decode windows on a tail-biting ring
- ``exllamav3_ext/quant/codebook.cuh``   procedural codebooks (plain / mcg / mul1)
- ``exllamav3_ext/quant/reconstruct.cu`` tile scatter (tensor-core permutation)
- ``exllamav3/modules/quant/exl3.py``    Hadamard/scale epilogue (get_weight_tensor)

Format facts, each verified against the reference sources (encoder and decoder
side):

- One 16x16 tile of the transformed weight W_hat packs 256 K-bit trellis
  indices MSB-first into 16*K uint16 words; the file stores each adjacent
  word pair swapped (the packer's SWAP16 on 32-bit groups).
- Index t is decoded through the 16-bit window of the bitstream ring that
  ENDS at ring bit (t+1)*K -- i.e. the low 16 bits of
  ``v[t] | v[t-1]<<K | v[t-2]<<(2K) | ...`` with indices wrapping the ring.
- The window feeds a procedural codebook: mcg multiplies by 0xCBAC1FED,
  plain by 89226354 (+ 64248484), mul1 by 0x83DCD12D with a dp4a byte sum.
  mcg/plain then apply the LOP3 ``((x & 0x8FFF8FFF) ^ 0x3B603B60)`` and add
  the two halves of the result reinterpreted as IEEE float16.
- Stream position p scatters to tile element ``tensor_core_perm()[p]``
  (flat index row*16 + col), matching both the quantizer's tensor_core_perm()
  and the store side of reconstruct.cu.
- Tile (kb, nb) covers rows kb*16..kb*16+15 and columns nb*16..nb*16+15 of
  W_hat (shape in_features x out_features).
- Original basis (exl3.py get_weight_tensor):
  ``A = (H128 on rows of W_hat) * suh[:, None]; W = (A * H128 on cols) * svh``
  with H128 the orthonormal Sylvester Hadamard (+-1/sqrt(128)); float32 math
  with float16 rounding after each of the four steps (torch half semantics).
- EXL3 stores linears as (in_features, out_features); the HuggingFace
  nn.Linear convention is (out, in), so callers transpose the result.
"""

from __future__ import annotations

import numpy as np

# --- codebook.cuh constants --------------------------------------------------

CODEBOOK_MCG_MULT = 0xCBAC1FED
CODEBOOK_MUL1_MULT = 0x83DCD12D
CODEBOOK_PLAIN_MULT = 89226354
CODEBOOK_PLAIN_ADD = 64248484
LOP3_AND = 0x8FFF8FFF
LOP3_XOR = 0x3B603B60
MUL1_ACC = 0x6400
MUL1_INV_BITS = 0x1EEE  # float16 0.00677.. ~= 1/147.7
MUL1_BIAS_BITS = 0xC931  # float16 -10.39..

CODEBOOKS = ("plain", "mcg", "mul1")
MAX_K = 8
_TILE = 16
_HAD = 128
_U32 = 0xFFFFFFFF


class EXL3FormatError(ValueError):
    """Raised when an EXL3 tensor group deviates from the reference format."""


try:  # numpy >= 2.0
    _popcount = np.bitwise_count
except AttributeError:  # pragma: no cover - fallback for numpy 1.x
    _popcount = np.vectorize(  # type: ignore[assignment]
        lambda x: int(x).bit_count(), otypes=[np.int64]
    )


def _f16_from_bits(bits: int) -> float:
    """IEEE float16 value of a 16-bit pattern (scalar helper)."""
    return float(np.array(bits, dtype=np.uint16).view(np.float16))


_h128: np.ndarray | None = None
_perm: np.ndarray | None = None


def hadamard_128() -> np.ndarray:
    """Orthonormal Sylvester Hadamard of order 128, natural order (+-1/sqrt(128)).

    The 7-fold Sylvester construction from ``[[1]]`` used by exllamav3's
    get_hadamard (hadamard_data/hadamard_1.txt + sylvester()) equals
    ``(-1) ** popcount(i & j)``; the 1/sqrt(128) scale matches preapply_had_l/r
    and the fused kernel constant r_scale = 0.08838834764831845.
    """
    global _h128
    if _h128 is None:
        i = np.arange(_HAD)
        signs = (-1.0) ** _popcount(i[:, None] & i[None, :]).astype(np.float64)
        _h128 = (signs / np.sqrt(_HAD)).astype(np.float32)
    return _h128


def tensor_core_perm() -> np.ndarray:
    """Stream position p -> flat 16x16 tile element (row * 16 + col).

    Transcription of tensor_core_perm() in exl3_lib/quantize.py; the store
    side of reconstruct.cu's warp shuffle implements the same mapping:
    p = 8*t + u with row = (t % 4)*2 + (u & 1) + 8*((u >> 1) & 1) and
    col = t // 4 + 8*(u >> 2).
    """
    global _perm
    if _perm is None:
        perm = np.empty(_TILE * _TILE, dtype=np.int64)
        for t in range(32):
            r0 = (t % 4) * 2
            c0 = t // 4
            for u in range(8):
                row = r0 + (u & 1) + 8 * ((u >> 1) & 1)
                col = c0 + 8 * (u >> 2)
                perm[t * 8 + u] = row * _TILE + col
        _perm = perm
    return _perm


def indices_from_packed(packed: np.ndarray, k: int) -> np.ndarray:
    """File-order uint16 words (..., 16k per tile) -> 256 K-bit indices per tile.

    Reversing each 4-byte group undoes the packer's SWAP16 and restores the
    natural MSB-first bitstream byte order (word hi, word lo, next word hi,
    ...). For k == 8 each natural byte is already one index.
    """
    if not 1 <= k <= MAX_K:
        raise EXL3FormatError(f"EXL3 bitrate K must be 1..{MAX_K}, got {k}")
    b = np.ascontiguousarray(packed)
    if b.dtype != np.uint16:
        raise EXL3FormatError(f"trellis words must be uint16, got {b.dtype}")
    if b.shape[-1] != 16 * k:
        raise EXL3FormatError(f"expected {16 * k} words per tile, got {b.shape[-1]}")
    raw = b.view(np.uint8)
    shape = raw.shape
    # one 4-byte group = one uint32 = two file words; reversing it yields the
    # natural byte order of the MSB-first bitstream
    nat = raw.reshape(shape[:-1] + (shape[-1] // 4, 4))[..., ::-1].reshape(shape)
    if k == 8:
        return nat  # uint8: one index per byte
    bits = np.unpackbits(nat, axis=-1)  # (..., 256*k) MSB-first
    groups = bits.reshape(bits.shape[:-1] + (256, k))
    weights = (1 << np.arange(k - 1, -1, -1)).astype(np.uint32)
    return (groups.astype(np.uint32) * weights).sum(axis=-1).astype(np.uint16)


def windows_from_indices(v: np.ndarray, k: int) -> np.ndarray:
    """256 indices per tile -> 256 16-bit codebook windows per tile.

    Window t is the 16-bit bitstream window ending at ring bit (t+1)*k: the
    low 16 bits of ``v[t] | v[t-1]<<k | v[t-2]<<(2k) | ...``, with indices
    wrapping around the 256-element tail-biting ring (np.roll).
    """
    out = v.astype(np.uint32)
    out = out.copy() if out is v else out
    j = 1
    while j * k < 16:
        rolled = np.roll(v, j, axis=-1).astype(np.uint32)
        out |= rolled << (j * k)
        j += 1
    return out & 0xFFFF


def decode_codebook(windows: np.ndarray, codebook: str) -> np.ndarray:
    """16-bit windows -> codebook values (float16, mirroring the CUDA kernels).

    mcg/plain: value = f16(lo) + f16(hi) of the LOP3 of the window product,
    exactly decode_3inst<cb> in codebook.cuh. mul1 approximates the fused
    multiply-add with a single float16 rounding of the float32 product-sum
    (at most 1 ulp from ``__hfma``).
    """
    if codebook not in CODEBOOKS:
        raise EXL3FormatError(f"unknown EXL3 codebook {codebook!r}")
    if codebook == "mul1":
        p = windows.astype(np.uint64) * np.uint64(CODEBOOK_MUL1_MULT)
        p &= np.uint64(_U32)
        acc = np.full(windows.shape, MUL1_ACC, dtype=np.int64)
        for shift in (0, 8, 16, 24):
            byte = ((p >> np.uint64(shift)) & np.uint64(0xFF)).astype(np.int64)
            acc += np.where(byte < 128, byte, byte - 256)
        h = (acc & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
        value = h * _f16_from_bits(MUL1_INV_BITS) + _f16_from_bits(MUL1_BIAS_BITS)
        return value.astype(np.float16)
    mult = CODEBOOK_MCG_MULT if codebook == "mcg" else CODEBOOK_PLAIN_MULT
    p = windows.astype(np.uint64) * np.uint64(mult)
    if codebook == "plain":
        p += np.uint64(CODEBOOK_PLAIN_ADD)
    p &= np.uint64(_U32)
    r = (p & np.uint64(LOP3_AND)) ^ np.uint64(LOP3_XOR)
    lo = (r & np.uint64(0xFFFF)).astype(np.uint16).view(np.float16)
    hi = (r >> np.uint64(16)).astype(np.uint16).view(np.float16)
    return lo + hi  # float16 addition, round-to-nearest (== __hadd)


def reconstruct_hat(trellis: np.ndarray, codebook: str) -> np.ndarray:
    """Packed trellis (kt, nt, 16k) -> W_hat float16 (kt*16, nt*16).

    W_hat is the transformed (regularized) weight. Tiles scatter via
    tensor_core_perm; tile (kb, nb) covers rows kb*16..+15 and columns
    nb*16..+15 (reconstruct.cu).
    """
    t = np.ascontiguousarray(trellis)
    if t.ndim != 3:
        raise EXL3FormatError(f"trellis must be 3-D (kt, nt, 16k), got shape {t.shape}")
    kt, nt, wlast = (int(x) for x in t.shape)
    if wlast % 16 != 0 or not 1 <= wlast // 16 <= MAX_K:
        raise EXL3FormatError(
            f"trellis last dim {wlast} is not 16*K for K in 1..{MAX_K}"
        )
    k = wlast // 16
    packed = t.view(np.uint16)
    perm = tensor_core_perm()
    rows = perm // _TILE
    cols = perm % _TILE
    w_hat = np.empty((kt * _TILE, nt * _TILE), dtype=np.float16)
    for kb in range(kt):
        idx = indices_from_packed(packed[kb], k)
        windows = windows_from_indices(idx, k)
        values = decode_codebook(windows, codebook)
        tile = np.empty((nt, _TILE, _TILE), dtype=np.float16)
        tile[:, rows, cols] = values
        w_hat[kb * _TILE : (kb + 1) * _TILE, :] = tile.transpose(1, 0, 2).reshape(
            _TILE, nt * _TILE
        )
    return w_hat


def apply_scales_and_hadamard(
    w_hat: np.ndarray,
    suh: np.ndarray,
    svh: np.ndarray,
) -> np.ndarray:
    """W_hat (in, out) + per-channel scales -> original-basis weight (float32).

    Mirrors exl3.py get_weight_tensor: blockwise H128 on the in-dimension,
    multiply suh (rows), blockwise H128 on the out-dimension, multiply svh
    (columns). Math in float32 with float16 rounding after each step.
    """
    in_f, out_f = (int(x) for x in w_hat.shape)
    if in_f % _HAD or out_f % _HAD:
        raise EXL3FormatError(
            f"EXL3 requires in/out features divisible by {_HAD}, got {in_f}x{out_f}"
        )
    if tuple(suh.shape) != (in_f,) or tuple(svh.shape) != (out_f,):
        raise EXL3FormatError(
            f"suh/svh shapes {suh.shape}/{svh.shape} do not match W_hat {w_hat.shape}"
        )
    h = hadamard_128()
    x = w_hat.astype(np.float32).reshape(in_f // _HAD, _HAD, out_f)
    a = (h @ x).reshape(in_f, out_f).astype(np.float16)
    b = (a.astype(np.float32) * suh.astype(np.float32)[:, None]).astype(np.float16)
    c0 = b.astype(np.float32).reshape(in_f, out_f // _HAD, _HAD)
    c = (c0 @ h).reshape(in_f, out_f).astype(np.float16)
    w = (c.astype(np.float32) * svh.astype(np.float32)[None, :]).astype(np.float16)
    return np.asarray(w, dtype=np.float32)


def dequantize_exl3_layer(
    trellis: np.ndarray,
    suh: np.ndarray,
    svh: np.ndarray,
    codebook: str,
) -> np.ndarray:
    """Dequantize one EXL3 linear group to the original basis (float32).

    Args:
        trellis: (in/16, out/16, 16*K) int16/uint16 packed trellis.
        suh: (in_features,) float16 input scales.
        svh: (out_features,) float16 output scales.
        codebook: "plain" | "mcg" | "mul1" (from the group's marker tensor).

    Returns:
        (in_features, out_features) float32. Note: the HuggingFace
        nn.Linear convention is (out, in) -- callers transpose.
    """
    w_hat = reconstruct_hat(trellis, codebook)
    return apply_scales_and_hadamard(w_hat, np.asarray(suh), np.asarray(svh))
