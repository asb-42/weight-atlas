"""Tests for degeneration guards: constant fields, low valid_fraction, min_cells."""

from __future__ import annotations

import numpy as np

from weight_atlas.fields.degenerations import (
    check_constant_field,
    diagnose_channel,
    diagnose_fields,
)


def test_constant_field_detected():
    """A field with all identical values should be degenerate."""
    field = np.full((10, 10), 5.0, dtype=np.float64)
    diag = diagnose_channel("test", field)
    assert diag.is_degenerate
    assert "normalized_std" in diag.reason.lower()


def test_all_nan_field_detected():
    """A field with all NaN should be degenerate."""
    field = np.full((10, 10), np.nan, dtype=np.float64)
    diag = diagnose_channel("test", field)
    assert diag.is_degenerate
    assert "no finite" in diag.reason.lower() or diag.valid_fraction == 0.0


def test_normal_field_not_degenerate():
    """A field with varying values should not be degenerate."""
    rng = np.random.default_rng(42)
    field = rng.normal(0, 1, (100, 13))
    diag = diagnose_channel("test", field)
    assert not diag.is_degenerate
    assert diag.valid_fraction == 1.0


def test_low_valid_fraction_detected():
    """A field with < 50% valid values should be degenerate."""
    field = np.full((10, 10), np.nan, dtype=np.float64)
    field[0, 0] = 1.0  # only 1/100 valid
    diag = diagnose_channel("test", field)
    assert diag.is_degenerate
    assert "valid_fraction" in diag.reason.lower()


def test_mixed_nan_and_constant():
    """Field with some NaN and constant values should be degenerate."""
    field = np.full((10, 10), np.nan, dtype=np.float64)
    field[0:5, :] = 3.14  # half valid but constant
    diag = diagnose_channel("test", field)
    assert diag.is_degenerate


def test_check_constant_field_true():
    """check_constant_field returns True for constant field."""
    field = np.full((5, 5), 2.0)
    assert check_constant_field(field) is True


def test_check_constant_field_false():
    """check_constant_field returns False for varying field."""
    field = np.arange(25, dtype=np.float64).reshape(5, 5)
    assert check_constant_field(field) is False


def test_check_constant_field_all_nan():
    """check_constant_field returns True for all-NaN field."""
    field = np.full((5, 5), np.nan)
    assert check_constant_field(field) is True


def test_diagnose_fields_multiple_channels():
    """diagnose_fields reports warnings for degenerate channels."""
    fields = {
        "height": np.full((10, 10), 5.0),  # constant → degenerate
        "tint": np.arange(100, dtype=np.float64).reshape(10, 10),  # ok
        "rough": np.full((10, 10), np.nan),  # all NaN → degenerate
    }
    report = diagnose_fields(fields, file=None)
    assert report.has_degenerations
    assert len(report.warnings) >= 2  # height and rough should warn
    assert "height" in report.channels
    assert "rough" in report.channels
    assert report.channels["height"].is_degenerate
    assert report.channels["rough"].is_degenerate
    assert not report.channels["tint"].is_degenerate


def test_empty_field():
    """Empty field (0 cells) should be degenerate."""
    field = np.array([], dtype=np.float64).reshape(0, 0)
    diag = diagnose_channel("test", field)
    assert diag.is_degenerate


def test_range_compression_extreme_outlier():
    """Field with extreme outlier should trigger range_compression warning."""
    rng = np.random.default_rng(42)
    # Normal values around 1.0, plus one massive outlier
    vals = rng.normal(1.0, 0.1, 99)
    vals = np.append(vals, 1000.0)  # extreme outlier → 100 values
    field = vals.reshape(10, 10)
    diag = diagnose_channel("height", field)
    # Should have low range_compression due to outlier
    assert diag.range_compression is not None
    assert diag.range_compression < 0.5  # much less than 5% threshold


def test_range_compression_uniform_field():
    """Uniform field (no outliers) should have high range_compression."""
    rng = np.random.default_rng(42)
    field = rng.normal(1.0, 0.1, (10, 10))
    diag = diagnose_channel("height", field)
    assert diag.range_compression is not None
    assert diag.range_compression > 0.5  # most of raw range covered


def test_range_compression_warning_emitted():
    """diagnose_fields should emit warning for extreme range compression."""
    rng = np.random.default_rng(42)
    # Many normal values plus one extreme outlier (outlier must not affect 99th percentile)
    vals = rng.normal(1.0, 0.05, 999)
    vals = np.append(vals, 500.0)  # 1000 values total
    field = vals.reshape(40, 25)
    fields = {"height": field}
    report = diagnose_fields(fields, file=None)
    # Should have range_compression warning
    range_warnings = [w for w in report.warnings if "RANGE COMPRESSION" in w]
    assert len(range_warnings) >= 1
    assert "extreme outlier" in range_warnings[0].lower()


def test_min_cells_guard_triggers():
    """Field with < 50 finite cells should be degenerate."""
    field = np.full((10, 10), np.nan, dtype=np.float64)
    field[0:4, :] = 1.0  # 40 valid cells, below threshold
    diag = diagnose_channel("test", field)
    assert diag.is_degenerate
    assert "too few cells" in diag.reason.lower()
    assert diag.n_valid == 40


def test_min_cells_guard_passes():
    """Field with >= 50 finite cells should not trigger min_cells guard."""
    rng = np.random.default_rng(42)
    field = rng.normal(0, 1, (10, 10))  # 100 valid cells
    diag = diagnose_channel("test", field)
    assert not diag.is_degenerate
    assert diag.n_valid == 100


def test_min_cells_warning_in_diagnose_fields():
    """diagnose_fields should emit warning for min_cells."""
    field = np.full((10, 10), np.nan, dtype=np.float64)
    field[0:3, :] = 1.0  # 30 valid cells, below threshold
    fields = {"height": field}
    report = diagnose_fields(fields, file=None)
    assert report.has_degenerations
    min_cell_warnings = [w for w in report.warnings if "too few cells" in w.lower()]
    assert len(min_cell_warnings) >= 1


def test_scan_persists_degeneration_warnings(tmp_path, monkeypatch):
    """Regression: warnings were merged into the fingerprint dict AFTER
    fingerprint.json had been written, so they never reached disk even
    though this module's contract says they flow into fingerprint.json."""
    import json

    import weight_atlas.fields.degenerations as degmod
    from tests.fixtures import make_fake_model
    from weight_atlas.core.types import load_default_spec
    from weight_atlas.fields.degenerations import DegenerationReport

    model_path = tmp_path / "model.safetensors"
    make_fake_model(model_path)
    out = tmp_path / "out"

    fake_report = DegenerationReport(warnings=["synthetic degeneration warning"])
    monkeypatch.setattr(
        degmod, "diagnose_fields", lambda fields, file=None: fake_report
    )

    from weight_atlas.scan import scan

    scan(model_path, out, load_default_spec())

    fp = json.loads((out / "fingerprint.json").read_text())
    assert "synthetic degeneration warning" in fp.get("warnings", [])
