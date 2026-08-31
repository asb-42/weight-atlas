"""Measured RTN quantizability: SQNR of INT8/INT4-g128/FP8-e4m3 per tensor.

Adopted from alesha-pro/atlas (MIT, see
docs/2026-08-31_atlas-alesha-pro-analysis.md §3) and re-implemented
deterministically in NumPy: canonical round-to-nearest quantization with
amax scaling (symmetric, no zero-point, no GPTQ/AWQ) — this measures the
*floor* damage of the standard formats, reference-free.

SQNR = 10·log10(‖W‖²/‖W−Ŵ‖²). Guards:
- tensor with zero signal (all zeros) → NaN (nothing to quantize),
- zero quantization error → 300.0 ("effectively lossless" ceiling; finite,
  JSON-safe — the alesha-pro original used a 999.0 sentinel),
- 1-D tensors → NaN (not applicable); tensors ≥ 2-D are flattened
  ``(shape[0], -1)`` like :func:`weight_atlas.stats.spectrum.to_matrix`,
- INT4: flattened row length must be a multiple of the group size, else NaN.

This probe is **opt-in** (``scan --quant-probe``): it adds ~6 chunked passes
over the weights. All accumulation is chunked float64 over row blocks, so no
full-size temporaries are materialized beyond the loaded payload. FP8 uses
:mod:`ml_dtypes` ``float8_e4m3fn`` (round-to-nearest-even, the same rounding
table as torch's hardware cast); values are pre-clamped to the e4m3 range
because the ``fn`` variant overflows to NaN instead of saturating.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from weight_atlas.core.registry import register_stat
from weight_atlas.core.types import TensorHandle

_NA = float("nan")
_LOSSLESS_CEILING = 300.0
_ROWS_PER_BLOCK = 4096

try:  # ml_dtypes is a declared dependency; localize the import anyway
    import ml_dtypes
except ImportError as _exc:  # pragma: no cover - dependency is declared
    raise ImportError(
        "stats.sqnr requires the 'ml_dtypes' package (pip install ml_dtypes)"
    ) from _exc

_E4M3_MAX = 448.0


def _sqnr_db(sig: float, err: float) -> float:
    if sig == 0.0:
        return _NA
    if err == 0.0:
        return _LOSSLESS_CEILING
    return float(10.0 * np.log10(sig / err))


def _to_matrix(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def _blocks(x: np.ndarray) -> Iterator[np.ndarray]:
    rows = x.shape[0]
    for lo in range(0, rows, _ROWS_PER_BLOCK):
        yield x[lo : lo + _ROWS_PER_BLOCK]


def _row_amax(x: np.ndarray) -> np.ndarray:
    """Per-row amax in float64, chunked (no full |x| copy)."""
    amax = np.zeros(x.shape[0], dtype=np.float64)
    start = 0
    for b in _blocks(x):
        blk = np.abs(b).max(axis=1).astype(np.float64)
        amax[start : start + blk.size] = blk
        start += blk.size
    return amax


def _sig_err_pair(
    x: np.ndarray, dequant_rowwise: Callable[[np.ndarray], np.ndarray]
) -> tuple[float, float]:
    """Chunked (Σw², Σ(w−dequant(w))²) with a row-block quantizer."""
    sig = 0.0
    err = 0.0
    for b in _blocks(x):
        w = b.astype(np.float64)
        wq = dequant_rowwise(b).astype(np.float64)
        sig += float((w * w).sum())
        err += float(((w - wq) ** 2).sum())
    return sig, err


def int8_per_channel_sqnr(x: np.ndarray) -> float:
    """INT8 symmetric per-row (last-dim channel) RTN SQNR in dB."""
    if x.ndim < 2 or x.shape[1] == 0:
        return _NA
    m = _to_matrix(x)
    scale = np.maximum(_row_amax(m), 1e-12) / 127.0
    start = 0

    def dequant(b: np.ndarray) -> np.ndarray:
        nonlocal start
        sl = scale[start : start + b.shape[0], None]
        out: np.ndarray = np.round(b / sl).clip(-127, 127) * sl
        start += b.shape[0]
        return out

    sig, err = _sig_err_pair(m, dequant)
    return _sqnr_db(sig, err)


def int4_group128_sqnr(x: np.ndarray, group: int = 128) -> float:
    """INT4 symmetric per-128-group RTN SQNR in dB (row length % group → NaN)."""
    if x.ndim < 2 or x.shape[1] == 0:
        return _NA
    m = _to_matrix(x)
    if m.shape[1] % group:
        return _NA
    groups = m.reshape(m.shape[0], m.shape[1] // group, group)
    amax = np.zeros(groups.shape[:2], dtype=np.float64)
    start = 0
    for b in _blocks(groups):
        blk = np.abs(b).max(axis=2).astype(np.float64)
        amax[start : start + blk.shape[0]] = blk
        start += blk.shape[0]
    scale = np.maximum(amax, 1e-12) / 7.0
    start = 0

    def dequant(b: np.ndarray) -> np.ndarray:
        nonlocal start
        gb = b.reshape(b.shape[0], -1, group)
        sl = scale[start : start + gb.shape[0], :, None]
        q = np.round(gb / sl).clip(-7, 7) * sl
        start += gb.shape[0]
        return q.reshape(b.shape)

    sig, err = _sig_err_pair(m, dequant)
    return _sqnr_db(sig, err)


def fp8_e4m3_sqnr(x: np.ndarray) -> float:
    """FP8 e4m3 (global amax/448 scale) RTN SQNR in dB via ml_dtypes."""
    if x.ndim < 2 or x.shape[1] == 0:
        return _NA
    m = _to_matrix(x)
    amax = max((float(np.abs(b).max()) for b in _blocks(m)), default=0.0)
    scale = max(amax, 1e-12) / _E4M3_MAX

    def dequant(b: np.ndarray) -> np.ndarray:
        v = np.clip(b / scale, -_E4M3_MAX, _E4M3_MAX)
        return v.astype(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale

    sig, err = _sig_err_pair(m, dequant)
    return _sqnr_db(sig, err)


class _SQNRStat:
    """Base for the registered SQNR stats: 2-D+ only, NaN otherwise."""

    _fn: Callable[[np.ndarray], float] = staticmethod(lambda x: _NA)

    def compute(self, t: TensorHandle) -> float:
        return self._fn(t.load()) if t.load().ndim >= 2 else _NA


@register_stat("sqnr_int8_ch")
class SQNRInt8PerChannel(_SQNRStat):
    """Measured INT8 per-channel RTN SQNR in dB (2-D+; NaN otherwise)."""

    stat_id = "sqnr_int8_ch"
    _fn = staticmethod(int8_per_channel_sqnr)


@register_stat("sqnr_int4_g128")
class SQNRInt4Group128(_SQNRStat):
    """Measured INT4 group-128 RTN SQNR in dB (row length % 128 == 0)."""

    stat_id = "sqnr_int4_g128"
    _fn = staticmethod(int4_group128_sqnr)


@register_stat("sqnr_fp8_e4m3")
class SQNRFp8E4M3(_SQNRStat):
    """Measured FP8 e4m3 RTN SQNR in dB (global-scale, ml_dtypes cast)."""

    stat_id = "sqnr_fp8_e4m3"
    _fn = staticmethod(fp8_e4m3_sqnr)
