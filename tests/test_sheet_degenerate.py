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


class TestDropEmptyCols:
    """spec knob ``sheet.drop_empty_cols`` (default off): drop all-NaN slot
    columns display-only so absent slot families don't leave white bands."""

    def _field(self) -> Field2D:
        rng = np.random.default_rng(7)
        data = rng.normal(0, 1, (10, 6))
        data[:, 1] = np.nan  # attn_k absent
        data[:, 4] = np.nan  # mlp_gate absent
        return Field2D(
            channel="height",
            data=data,
            row_labels=[str(i) for i in range(10)],
            col_labels=["attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up"],
        )

    def _spec(self, spec: AtlasSpec, drop: bool) -> AtlasSpec:
        spec_dict = spec.__dict__.copy()
        spec_dict["sheet"] = {**spec.sheet, "drop_empty_cols": drop}
        return AtlasSpec(**spec_dict)

    def test_default_off_keeps_all_columns(self, spec: AtlasSpec, tmp_path: Path):
        """Without the knob, all-NaN columns are NOT dropped (backward compat)."""
        renderer = MatplotlibSheet()
        paths = renderer.render(self._field(), spec, tmp_path / "render")
        assert len(paths) == 1
        assert paths[0].exists()

    def test_on_drops_nan_columns(self, spec: AtlasSpec, tmp_path: Path):
        """With the knob, all-NaN columns are dropped so the sheet is narrower."""
        from PIL import Image

        renderer = MatplotlibSheet()
        field = self._field()
        out_off = tmp_path / "off"
        out_on = tmp_path / "on"
        renderer.render(field, self._spec(spec, drop=False), out_off)
        renderer.render(field, self._spec(spec, drop=True), out_on)
        w_off, _ = Image.open(out_off / "height_raw.png").size
        w_on, _ = Image.open(out_on / "height_raw.png").size
        assert w_on < w_off
        # 6 → 4 kept columns: strictly narrower at the same pixel budget.
        assert w_on / w_off < 0.85

    def test_upsampled_smooth_field_drops_slot_blocks(self, spec: AtlasSpec, tmp_path: Path):
        """Smooth fields are upsampled (n_cols = n_slots * upsample); a slot
        whose entire block is NaN must be dropped as a whole block."""
        from PIL import Image

        # Simulate a 2x-upsampled smooth field: 4 logical slots × 2 = 8 cols.
        data = np.random.default_rng(3).normal(0, 1, (10, 8))
        data[:, 2:4] = np.nan  # slot 1 (attn_k) entirely absent → cols 2,3
        data[:, 6:8] = np.nan  # slot 3 (mlp_gate) entirely absent → cols 6,7
        field = Field2D(
            channel="height",
            data=data,
            row_labels=[str(i) for i in range(10)],
            col_labels=["attn_q", "attn_k", "attn_o", "mlp_gate"],
        )
        renderer = MatplotlibSheet()
        out_off = tmp_path / "off"
        out_on = tmp_path / "on"
        renderer.render(field, self._spec(spec, drop=False), out_off)
        renderer.render(field, self._spec(spec, drop=True), out_on)
        w_off, _ = Image.open(out_off / "height_raw.png").size
        w_on, _ = Image.open(out_on / "height_raw.png").size
        assert w_on < w_off
        assert w_on / w_off < 0.85

    def test_all_nan_columns_still_renders(self, spec: AtlasSpec, tmp_path: Path):
        """If every column is NaN, the field still renders (nothing to drop)."""
        field = Field2D(
            channel="height",
            data=np.full((10, 3), np.nan),
            row_labels=[str(i) for i in range(10)],
            col_labels=["a", "b", "c"],
        )
        renderer = MatplotlibSheet()
        paths = renderer.render(field, self._spec(spec, drop=True), tmp_path / "render")
        assert len(paths) == 1
        assert paths[0].exists()

    def test_deterministic_across_runs(self, spec: AtlasSpec, tmp_path: Path):
        """Two renders with the knob on are byte-identical (determinism contract)."""
        renderer = MatplotlibSheet()
        field = self._field()
        a = tmp_path / "a"
        b = tmp_path / "b"
        renderer.render(field, self._spec(spec, drop=True), a)
        renderer.render(field, self._spec(spec, drop=True), b)
        assert (a / "height_raw.png").read_bytes() == (b / "height_raw.png").read_bytes()
