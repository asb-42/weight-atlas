"""Rasterizer, scaling, smoothing, tif_io tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from weight_atlas.core.types import AtlasSpec, TensorStats
from weight_atlas.fields.rasterizer import load_channel_field, rasterize
from weight_atlas.fields.scaling import log1p, quantile_clip, rank_scale
from weight_atlas.fields.smoothing import smooth, upsample
from weight_atlas.fields.tif_io import read_tif, write_tif


@pytest.fixture
def spec():
    return AtlasSpec.from_json(Path("specs/atlas_spec.v2.2.json"))


def test_rasterizer_shape_and_nan(spec):
    stats = [
        TensorStats(name=f"model.layers.{idx}.self_attn.q_proj.weight", shape=(4, 4), spectral_norm=float(idx + 1))
        for idx in range(3)
    ]
    field = rasterize(stats, spec, "spectral_norm")
    assert field.data.shape == (3, len(spec.slots))
    # Only the attn_q column should be populated.
    q_idx = spec.slots.index("attn_q")
    assert np.all(np.isfinite(field.data[:, q_idx]))
    others = [i for i in range(len(spec.slots)) if i != q_idx]
    assert np.all(np.isnan(field.data[:, others[0]]))


def test_rasterizer_multi_slot(spec):
    stats = []
    for idx in range(2):
        for slot_name in ("self_attn.q_proj", "mlp.gate_proj"):
            stats.append(
                TensorStats(
                    name=f"model.layers.{idx}.{slot_name}.weight",
                    shape=(4, 4),
                    frobenius=1.0,
                )
            )
    field = rasterize(stats, spec, "frobenius")
    assert field.data.shape == (2, len(spec.slots))


def test_log1p_nonneg():
    x = np.array([0.0, 1.0, np.nan, 2.0])
    y = log1p(x)
    assert np.isnan(y[2])
    assert pytest.approx(y[0]) == 0.0
    assert pytest.approx(y[1]) == np.log1p(1.0)


def test_quantile_clip_range():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 1000)
    y = quantile_clip(x, lo=0.01, hi=0.99)
    assert y.min() >= -1e-9
    assert y.max() <= 1.0 + 1e-9


def test_rank_scale_range():
    """rank_scale should map values to [0, 1] using percentile ranks."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 1000)
    y = rank_scale(x)
    assert y.min() >= -1e-9
    assert y.max() <= 1.0 + 1e-9


def test_rank_scale_immune_to_outliers():
    """rank_scale should be immune to extreme outliers."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 100)
    x_with_outlier = np.append(x, 1e6)  # extreme outlier
    y = rank_scale(x_with_outlier)
    # All values should be in [0, 1]
    assert y.min() >= -1e-9
    assert y.max() <= 1.0 + 1e-9
    # The outlier should map to 1.0, not dominate the distribution
    assert y[-1] == pytest.approx(1.0)
    # Other values should still span the full range
    assert y[:-1].min() < 0.1
    assert y[:-1].max() > 0.5


def test_rank_scale_preserves_nan():
    """rank_scale should preserve NaN values."""
    x = np.array([1.0, np.nan, 3.0, 2.0])
    y = rank_scale(x)
    assert np.isnan(y[1])
    assert not np.isnan(y[0])
    assert not np.isnan(y[2])
    assert not np.isnan(y[3])


def test_rank_scale_uniform_distribution():
    """rank_scale on uniform data should produce uniform ranks."""
    x = np.arange(100, dtype=np.float64)
    y = rank_scale(x)
    expected = np.arange(100, dtype=np.float64) / 99.0
    np.testing.assert_allclose(y, expected)


def test_rank_scale_reversed():
    """rank_scale on reversed data should produce reversed ranks."""
    x = np.arange(100, 0, -1, dtype=np.float64)
    y = rank_scale(x)
    expected = np.arange(99, -1, -1, dtype=np.float64) / 99.0
    np.testing.assert_allclose(y, expected)


def test_rank_scale_per_column():
    """rank_scale with per_column=True should rank each column independently."""
    # Column 0: values 1-10, Column 1: values 100-1000
    x = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0], [5.0, 500.0],
                  [6.0, 600.0], [7.0, 700.0], [8.0, 800.0], [9.0, 900.0], [10.0, 1000.0]])
    y = rank_scale(x, per_column=True)
    # Each column should span [0, 1] independently
    assert y[:, 0].min() == pytest.approx(0.0)
    assert y[:, 0].max() == pytest.approx(1.0)
    assert y[:, 1].min() == pytest.approx(0.0)
    assert y[:, 1].max() == pytest.approx(1.0)
    # Column 0: value 1 -> rank 0, value 10 -> rank 1
    assert y[0, 0] == pytest.approx(0.0)
    assert y[9, 0] == pytest.approx(1.0)
    # Column 1: value 100 -> rank 0, value 1000 -> rank 1
    assert y[0, 1] == pytest.approx(0.0)
    assert y[9, 1] == pytest.approx(1.0)


def test_rank_scale_per_column_with_nan():
    """rank_scale per_column should handle NaN values."""
    x = np.array([[1.0, np.nan], [2.0, 200.0], [3.0, 300.0]])
    y = rank_scale(x, per_column=True)
    assert not np.isnan(y[0, 0])
    assert np.isnan(y[0, 1])
    assert not np.isnan(y[1, 1])


def test_upsample_bilinear():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = upsample(x, factor=2)
    assert y.shape == (4, 4)


def test_smooth_preserves_finite():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (10, 10))
    y = smooth(x, sigma=1.0)
    assert y.shape == x.shape
    # Gaussian smooth should reduce max absolute magnitude generally.
    assert np.nanmax(np.abs(y)) <= np.max(np.abs(x)) + 1e-6


def test_tif_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "f.tif"
        arr = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        write_tif(p, arr)
        back = read_tif(p)
        assert back.shape == arr.shape
        np.testing.assert_allclose(back, arr, rtol=1e-6)


def test_tif_deterministic():
    """Same array twice → identical bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        p1 = tmp / "a.tif"
        p2 = tmp / "b.tif"
        write_tif(p1, arr)
        write_tif(p2, arr)
        assert p1.read_bytes() == p2.read_bytes()


def _make_scan_dir(tmp: Path, spec) -> Path:
    """Write a scan-like out dir: 4 layers, upsample 8, all channels."""
    out = tmp / "scan"
    out.mkdir(exist_ok=True)
    ups = int(spec.grid["upsample"])
    n_rows, n_cols = 4 * ups, len(spec.slots) * ups
    raw = np.random.default_rng(0).uniform(0.1, 2.0, (4, len(spec.slots)))
    smooth = np.random.default_rng(1).uniform(0.0, 1.0, (n_rows, n_cols))
    for ch in spec.channels:
        write_tif(out / f"field_{ch}_raw.tif", raw)
        write_tif(out / f"field_{ch}_smooth.tif", smooth)
    return out


def test_load_channel_field_smooth_labels(spec, tmp_path):
    """Smooth field should carry slot names + true layer indices + model name."""
    out = _make_scan_dir(tmp_path, spec)
    field = load_channel_field(out, "height", spec, model_name="Bonsai-8B")
    assert field is not None
    assert field.model_name == "Bonsai-8B"
    # Slot names, not upsampled column indices.
    assert field.col_labels == list(spec.slots)
    # True layer count, not upsampled rows.
    assert field.row_labels == ["0", "1", "2", "3"]
    ups = int(spec.grid["upsample"])
    assert field.data.shape == (4 * ups, len(spec.slots) * ups)


def test_load_channel_field_raw_fallback_scales(spec, tmp_path):
    """Raw-only fallback should apply the channel scale and use row count as layers."""
    out = _make_scan_dir(tmp_path, spec)
    for ch in spec.channels:
        (out / f"field_{ch}_smooth.tif").unlink()
    field = load_channel_field(out, "height", spec, model_name="m")
    assert field is not None
    assert field.row_labels == ["0", "1", "2", "3"]
    assert field.data.shape == (4, len(spec.slots))
    # height uses rank_scale → values in [0, 1]
    assert np.nanmin(field.data) >= -1e-9
    assert np.nanmax(field.data) <= 1.0 + 1e-9


def test_load_channel_field_missing_returns_none(spec, tmp_path):
    out = tmp_path / "empty"
    out.mkdir(exist_ok=True)
    assert load_channel_field(out, "height", spec) is None
