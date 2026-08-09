"""Tests for degeneration guards: constant fields, low valid_fraction."""

from __future__ import annotations

import numpy as np
import pytest

from weight_atlas.fields.degenerations import (
    ChannelDiagnostics,
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
