"""Tests for preview renderer: float32 TIFF → 8-bit PNG with auto-levels + gamma."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from weight_atlas.core.types import AtlasSpec, Field2D
from weight_atlas.render.preview import PreviewRenderer, _auto_levels


@pytest.fixture
def spec() -> AtlasSpec:
    return AtlasSpec.from_json(Path("specs/atlas_spec.v2.1.json"))


@pytest.fixture
def tmp_out(tmp_path: Path) -> Path:
    return tmp_path / "render"


class TestAutoLevels:
    def test_basic_stretching(self):
        data = np.array([[0.0, 1.0], [2.0, 3.0]])
        result = _auto_levels(data, lo=0.0, hi=1.0)
        assert result[0, 0] == 0.0
        assert result[1, 1] == 1.0

    def test_nan_preserved(self):
        data = np.array([[np.nan, 1.0], [2.0, 3.0]])
        result = _auto_levels(data, lo=0.0, hi=1.0)
        assert np.isnan(result[0, 0])
        # Global min=1.0, max=3.0 among finite values
        assert result[0, 1] == 0.0
        assert result[1, 0] == 0.5
        assert result[1, 1] == 1.0

    def test_constant_field(self):
        data = np.full((3, 3), 5.0)
        result = _auto_levels(data, lo=0.0, hi=1.0)
        assert np.all(result == 0.0)

    def test_all_nan(self):
        data = np.full((3, 3), np.nan)
        result = _auto_levels(data, lo=0.0, hi=1.0)
        assert np.all(np.isnan(result))

    def test_quantile_clipping(self):
        data = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
        result = _auto_levels(data, lo=0.01, hi=0.99)
        # 100.0 should be clipped to 1.0
        assert result[-1] == 1.0


class TestPreviewRenderer:
    def test_render_creates_png(self, spec: AtlasSpec, tmp_out: Path):
        renderer = PreviewRenderer()
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
        assert "preview_height" in paths[0].name

    def test_render_with_nan(self, spec: AtlasSpec, tmp_out: Path):
        renderer = PreviewRenderer()
        data = np.random.default_rng(42).normal(0, 1, (10, 13))
        data[0, 0] = np.nan
        field = Field2D(
            channel="tint",
            data=data,
            row_labels=[str(i) for i in range(10)],
            col_labels=[str(i) for i in range(13)],
        )
        paths = renderer.render(field, spec, tmp_out)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_render_constant_field(self, spec: AtlasSpec, tmp_out: Path):
        renderer = PreviewRenderer()
        field = Field2D(
            channel="rough",
            data=np.full((10, 13), 5.0),
            row_labels=[str(i) for i in range(10)],
            col_labels=[str(i) for i in range(13)],
        )
        paths = renderer.render(field, spec, tmp_out)
        assert len(paths) == 1
        assert paths[0].exists()

    def test_render_upsampled_sparse_labels(self, spec: AtlasSpec, tmp_out: Path):
        """Fewer slot/layer labels than columns/rows must not raise."""
        renderer = PreviewRenderer()
        n_rows, n_cols = 32, 120  # 4 layers × 8, 15 slots × 8 (upsampled)
        field = Field2D(
            channel="height",
            data=np.random.default_rng(42).normal(0, 1, (n_rows, n_cols)),
            row_labels=[str(i) for i in range(4)],
            col_labels=list(spec.slots),
            model_name="Bonsai-8B",
        )
        paths = renderer.render(field, spec, tmp_out)
        assert len(paths) == 1
        assert paths[0].exists()
