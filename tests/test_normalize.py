"""Tests for the normalized-depth (Ebene 2) projection."""

from __future__ import annotations

import numpy as np

from weight_atlas.fields.normalize import depth_landmark_labels, project_normalized_depth


class TestProjectNormalizedDepth:
    def test_shape_projected(self):
        data = np.arange(40, dtype=np.float64).reshape(8, 5)
        projected, mask = project_normalized_depth(data, 21)
        assert projected.shape == (21, 5)
        assert mask.shape == (21, 5)
        assert mask.dtype == bool

    def test_dense_field_not_marked_interpolated(self):
        # Dense tall field: every landmark has a measured row within half a
        # landmark spacing, so nothing is shaded.
        rng = np.random.default_rng(0)
        data = rng.normal(size=(64, 7))
        projected, mask = project_normalized_depth(data, 21)
        assert np.all(np.isfinite(projected))
        assert not mask.any()

    def test_hole_is_interpolated_and_marked(self):
        # Column 0 has a gap in the middle; landmarks inside the gap must be
        # interpolated and marked, while collocated landmarks stay measured.
        data = np.full((10, 1), np.nan)
        data[0:4, 0] = [0.0, 1.0, 2.0, 3.0]
        data[8:10, 0] = [8.0, 9.0]
        projected, mask = project_normalized_depth(data, 11)  # 0%,10%,...,100%
        assert projected.shape == (11, 1)
        # Landmarks 0%..30% coincide with measured rows 0..3.
        for i in (0, 1, 2, 3):
            assert not mask[i, 0]
        # Landmarks 40%..80% sit inside the hole -> interpolated.
        for i in (4, 5, 6, 7, 8):
            assert mask[i, 0]
        # 90% and 100% coincide with rows 8 and 9.
        assert not mask[9, 0]
        assert not mask[10, 0]
        # The hole is filled with finite (interpolated) values.
        assert np.all(np.isfinite(projected[4:9, 0]))

    def test_extreme_holes_stay_nan(self):
        # Column measured only in the middle -> outer landmarks stay NaN.
        data = np.full((10, 1), np.nan)
        data[3:7, 0] = [3.0, 4.0, 5.0, 6.0]
        projected, mask = project_normalized_depth(data, 11)
        # 0%..30% and 70%..100% lie outside the measured depth range.
        assert np.isnan(projected[0:4, 0]).all()
        assert np.isnan(projected[7:11, 0]).all()
        # 40%..60% are interpolated across the middle and finite.
        assert np.all(np.isfinite(projected[4:7, 0]))

    def test_all_nan_column_marked(self):
        data = np.full((10, 2), np.nan)
        data[:, 1] = 1.0  # only column 1 is measured
        projected, mask = project_normalized_depth(data, 11)
        assert mask[:, 0].all()      # unmeasured column fully interpolated
        assert not mask[:, 1].all()  # measured column has direct cells

    def test_interpolation_linear_in_gap(self):
        # A single-hole column interpolates linearly between the flanking rows.
        data = np.full((5, 1), np.nan)
        data[0, 0] = 0.0
        data[4, 0] = 4.0
        projected, mask = project_normalized_depth(data, 5)  # 0%,25%,50%,75%,100%
        expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(projected[:, 0], expected)
        # Only the two end landmarks coincide with measured rows.
        assert not mask[0, 0]
        assert not mask[4, 0]
        assert mask[1:4, 0].all()

    def test_single_row_unchanged(self):
        data = np.array([[1.0, 2.0, 3.0]])
        projected, mask = project_normalized_depth(data, 21)
        assert projected.shape == (1, 3)
        assert not mask.any()
        np.testing.assert_array_equal(projected, data)


class TestDepthLandmarkLabels:
    def test_labels(self):
        assert depth_landmark_labels(5) == ["0%", "25%", "50%", "75%", "100%"]

    def test_single(self):
        assert depth_landmark_labels(1) == ["100%"]

    def test_zero(self):
        assert depth_landmark_labels(0) == []
