"""Measured RTN quantizability (P1.4): INT8/INT4-g128/FP8-e4m3 SQNR stats."""

from __future__ import annotations

import numpy as np
import pytest

from weight_atlas.core.types import TensorHandle
from weight_atlas.stats.sqnr import (
    _LOSSLESS_CEILING,
    fp8_e4m3_sqnr,
    int4_group128_sqnr,
    int8_per_channel_sqnr,
)


def _ref_sqnr(x: np.ndarray, fmt: str) -> float:
    """Independent (non-chunked, float64) reference implementation."""
    x64 = x.astype(np.float64)
    if fmt == "int8":
        s = np.maximum(np.abs(x64).max(axis=1), 1e-12) / 127.0
        wq = np.round(x64 / s[:, None]).clip(-127, 127) * s[:, None]
    elif fmt == "int4":
        g = x64.reshape(x64.shape[0], -1, 128)
        s = np.maximum(np.abs(g).max(axis=2), 1e-12) / 7.0
        wq = (np.round(g / s[:, :, None]).clip(-7, 7) * s[:, :, None]).reshape(x64.shape)
    else:
        import ml_dtypes

        s = max(np.abs(x64).max(), 1e-12) / 448.0
        wq = np.clip(x64 / s, -448.0, 448.0).astype(ml_dtypes.float8_e4m3fn).astype(np.float64) * s
    sig = float((x64**2).sum())
    err = float(((x64 - wq) ** 2).sum())
    return 10.0 * np.log10(sig / err)


def test_matches_reference_small_and_multi_block() -> None:
    """Chunked accumulation == vectorized reference, including the >4096-row
    multi-block path (regression: the per-row scale offset must track the
    global row index, not restart per block)."""
    rng = np.random.default_rng(7)
    for x in (
        rng.standard_normal((128, 256)).astype(np.float32),
        rng.standard_normal((5000, 256)).astype(np.float32),  # multi-block
        rng.standard_normal((17, 512)).astype(np.float32),  # ragged rows
    ):
        for fmt, fn in (
            ("int8", int8_per_channel_sqnr),
            ("int4", int4_group128_sqnr),
            ("fp8", fp8_e4m3_sqnr),
        ):
            got, exp = fn(x), _ref_sqnr(x, fmt)
            assert got == pytest.approx(exp, abs=1e-6), f"{fmt}: {got} != {exp}"


def test_na_discipline() -> None:
    rng = np.random.default_rng(3)
    # 1-D tensors: not applicable
    assert np.isnan(int8_per_channel_sqnr(rng.standard_normal(64).astype(np.float32)))
    # INT4 requires the flattened row length to be a multiple of 128
    assert np.isnan(int4_group128_sqnr(rng.standard_normal((16, 100)).astype(np.float32)))
    assert np.isfinite(int4_group128_sqnr(rng.standard_normal((16, 128)).astype(np.float32)))
    # all-zero tensor: no signal → NaN (never an infinite SQNR of 0/0)
    assert np.isnan(int8_per_channel_sqnr(np.zeros((16, 128), dtype=np.float32)))


def test_lossless_ceiling() -> None:
    """A row of constants quantizes exactly → the finite lossless ceiling."""
    x = np.full((8, 128), 0.25, dtype=np.float32)
    assert int8_per_channel_sqnr(x) == _LOSSLESS_CEILING


def test_fp8_no_overflow_nan() -> None:
    """Values at/above the amax scale must not produce NaN (fn-overflow)."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal((32, 64)).astype(np.float32) * 1e6  # large scale
    assert np.isfinite(fp8_e4m3_sqnr(x))


def test_handle_compute_and_registry_shape() -> None:
    """End-to-end through the TensorHandle + registered stat classes."""
    from weight_atlas.stats.sqnr import SQNRInt8PerChannel

    rng = np.random.default_rng(9)
    x = rng.standard_normal((64, 128)).astype(np.float32)
    t = TensorHandle("t", x.shape, "float32", lambda: x)
    v = SQNRInt8PerChannel().compute(t)
    assert v == pytest.approx(_ref_sqnr(x, "int8"), abs=1e-6)
    # 1-D handle → NaN
    v1 = TensorHandle("v", (64,), "float32", lambda: rng.standard_normal(64).astype(np.float32))
    assert np.isnan(SQNRInt8PerChannel().compute(v1))


def test_scan_pipeline_quant_probe_off_and_on(tmp_path) -> None:
    """The probe is opt-in: off → NaN fields, on → populated; both scans are
    deterministic (byte-identical fingerprints per flag)."""
    import json

    from tests.fixtures import make_fake_model

    model_path = tmp_path / "fake_model.safetensors"
    make_fake_model(model_path)
    from weight_atlas.core.types import load_default_spec

    spec = load_default_spec()

    from weight_atlas.scan import scan

    out_off = tmp_path / "off"
    scan(model_path, out_off, spec)
    fp_off = json.loads((out_off / "fingerprint.json").read_text())
    some = next(iter(fp_off["tensors"].values()))
    assert np.isnan(some["sqnr_int8_ch"])

    out_on = tmp_path / "on"
    scan(model_path, out_on, spec, quant_probe=True)
    fp_on = json.loads((out_on / "fingerprint.json").read_text())
    for info in fp_on["tensors"].values():
        shape = info["shape"]
        if len(shape) >= 2 and (shape[-1] if len(shape) > 1 else 0) % 128 == 0:
            assert np.isfinite(info["sqnr_int8_ch"])

    # Determinism: probe scans are byte-identical
    out_on2 = tmp_path / "on2"
    scan(model_path, out_on2, spec, quant_probe=True)
    assert (out_on / "fingerprint.json").read_bytes() == (out_on2 / "fingerprint.json").read_bytes()
