"""Tests for matplotlib sheet renderer: degenerate banner + per-row normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from weight_atlas.core.types import AtlasSpec, Field2D
from weight_atlas.render.matplotlib_sheet import (
    MatplotlibSheet,
    _check_degenerate,
    _per_row_normalize,
)


@pytest.fixture
def spec() -> AtlasSpec:
    return AtlasSpec.from_json(Path("specs/atlas_spec.v2.2.json"))


@pytest.fixture
def tmp_out(tmp_path: Path) -> Path:
    return tmp_path / "render"


class TestCheckDegenerate:
    def test_constant_field_degenerate(self):
        data = np.full((10, 13), 5.0)
        is_deg, reason = _check_degenerate(data)
        assert is_deg
        assert "normalized_std" in reason

    def test_all_nan_degenerate(self):
        data = np.full((10, 13), np.nan)
        is_deg, reason = _check_degenerate(data)
        assert is_deg
        assert "no finite" in reason

    def test_normal_field_not_degenerate(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (100, 13))
        is_deg, reason = _check_degenerate(data)
        assert not is_deg
        assert reason == ""

    def test_low_valid_fraction_degenerate(self):
        data = np.full((10, 13), np.nan)
        data[0, 0] = 1.0  # only 1/130 valid
        is_deg, reason = _check_degenerate(data)
        assert is_deg
        assert "valid_fraction" in reason

    def test_empty_field_degenerate(self):
        data = np.array([], dtype=np.float64).reshape(0, 0)
        is_deg, reason = _check_degenerate(data)
        assert is_deg
        assert "empty" in reason


class TestPerRowNormalize:
    def test_basic_normalization(self):
        data = np.array([[0.0, 1.0, 2.0], [10.0, 20.0, 30.0]])
        result = _per_row_normalize(data)
        np.testing.assert_allclose(result[0], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(result[1], [0.0, 0.5, 1.0])

    def test_nan_preserved(self):
        data = np.array([[np.nan, 1.0, 2.0], [10.0, 20.0, 30.0]])
        result = _per_row_normalize(data)
        assert np.isnan(result[0, 0])
        # Row 0: finite=[1.0, 2.0], min=1.0, max=2.0 → [0.0, 1.0]
        np.testing.assert_allclose(result[0, 1:], [0.0, 1.0])
        # Row 1: min=10.0, max=30.0 → [0.0, 0.5, 1.0]
        np.testing.assert_allclose(result[1], [0.0, 0.5, 1.0])

    def test_constant_row(self):
        data = np.array([[5.0, 5.0, 5.0], [10.0, 20.0, 30.0]])
        result = _per_row_normalize(data)
        np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(result[1], [0.0, 0.5, 1.0])


class TestMatplotlibSheetRender:
    def test_render_creates_png(self, spec: AtlasSpec, tmp_out: Path):
        renderer = MatplotlibSheet()
        field = Field2D(
            channel="height",
            data=np.random.default_rng(42).normal(0, 1, (10, 13)),
            row_labels=[str(i) for i in range(10)],
            col_labels=[str(i) for i in range(13)],
        )
        paths = renderer.render(field, spec, tmp_out)
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].suffix == ".png"

    def test_render_with_per_row_normalize(self, spec: AtlasSpec, tmp_out: Path):
        # Modify spec to enable per-row normalization
        spec_dict = spec.__dict__.copy()
        spec_dict["sheet"] = {**spec.sheet, "per_row_normalize": True}
        spec2 = AtlasSpec(**spec_dict)

        renderer = MatplotlibSheet()
        field = Field2D(
            channel="height",
            data=np.random.default_rng(42).normal(0, 1, (10, 13)),
            row_labels=[str(i) for i in range(10)],
            col_labels=[str(i) for i in range(13)],
        )
        paths = renderer.render(field, spec2, tmp_out)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_render_constant_field_still_works(self, spec: AtlasSpec, tmp_out: Path):
        """Degenerate field should still render (with banner)."""
        renderer = MatplotlibSheet()
        field = Field2D(
            channel="height",
            data=np.full((10, 13), 5.0),
            row_labels=[str(i) for i in range(10)],
            col_labels=[str(i) for i in range(13)],
        )
        paths = renderer.render(field, spec, tmp_out)
        assert len(paths) == 1
        assert paths[0].exists()
