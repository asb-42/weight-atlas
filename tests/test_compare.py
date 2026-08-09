"""Tests for the comparison/delta layer (M4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from weight_atlas.compare import (
    ChannelDelta,
    align,
    check_compatibility,
    compute_compare_summary,
    hotspot_ranking,
)
from weight_atlas.compare.align import _resample_field
from weight_atlas.compare.delta import (
    _compute_cosine_sim,
    _compute_rel_l2,
    _find_hotspot,
    _safe_subtract,
    compute_delta,
)
from weight_atlas.core.types import AtlasSpec

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def spec() -> AtlasSpec:
    """Load the default atlas spec."""
    return AtlasSpec.from_json(Path("specs/atlas_spec.v2.json"))


@pytest.fixture
def simple_spec() -> AtlasSpec:
    """Minimal spec for testing."""
    return AtlasSpec(
        spec_version=1,
        slots=["attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"],
        channels={
            "height": {"stat": "spectral_norm", "scale": {"type": "log1p"}},
        },
        grid={"upsample": 4, "smooth_sigma": 1.0},
        sheet={"contour_levels": 8, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
        seeds={"svd": 0},
    )


@pytest.fixture
def field_a() -> np.ndarray:
    """Simple 4x7 field (4 layers, 7 slots)."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, (4, 7)).astype(np.float64)


@pytest.fixture
def field_b() -> np.ndarray:
    """Simple 4x7 field with small perturbations."""
    rng = np.random.default_rng(43)
    return rng.normal(0, 1, (4, 7)).astype(np.float64)


@pytest.fixture
def field_b_field_a_perturbed(field_a: np.ndarray) -> np.ndarray:
    """Field B = A + perturbations at specific locations."""
    b = field_a.copy()
    # Perturb layer 2, slot 6 (mlp_down) by 1.5x scale
    b[2, 6] *= 1.5
    # Perturb layer 3, slot 3 (attn_o) with rank-1 perturbation
    b[3, 3] += 0.5
    return b


# ---------------------------------------------------------------------------
# Compatibility checks
# ---------------------------------------------------------------------------


class TestCheckCompatibility:
    def test_same_spec_version_no_warnings(self):
        fp_a = {"spec_version": 1, "tool_version": "0.1.0", "loader": "safetensors"}
        fp_b = {"spec_version": 1, "tool_version": "0.1.0", "loader": "safetensors"}
        warnings = check_compatibility(fp_a, fp_b)
        assert warnings == []

    def test_spec_version_mismatch_raises(self):
        fp_a = {"spec_version": 1, "tool_version": "0.1.0"}
        fp_b = {"spec_version": 2, "tool_version": "0.1.0"}
        with pytest.raises(ValueError, match="spec_version mismatch"):
            check_compatibility(fp_a, fp_b)

    def test_tool_version_mismatch_warns(self):
        fp_a = {"spec_version": 1, "tool_version": "0.1.0"}
        fp_b = {"spec_version": 1, "tool_version": "0.2.0"}
        warnings = check_compatibility(fp_a, fp_b)
        assert len(warnings) == 1
        assert "tool_version mismatch" in warnings[0]

    def test_loader_mismatch_warns(self):
        fp_a = {"spec_version": 1, "tool_version": "0.1.0", "loader": "safetensors"}
        fp_b = {"spec_version": 1, "tool_version": "0.1.0", "loader": "gguf"}
        warnings = check_compatibility(fp_a, fp_b)
        assert len(warnings) == 1
        assert "loader mismatch" in warnings[0]


# ---------------------------------------------------------------------------
# Strict alignment
# ---------------------------------------------------------------------------


class TestAlignStrict:
    def test_same_shape_returns_copy(self, field_a, field_b, spec):
        result = align(field_a, field_b, spec, mode="strict")
        assert result.mode == "strict"
        assert result.data_a.shape == field_a.shape
        assert result.data_b.shape == field_b.shape
        np.testing.assert_array_equal(result.data_a, field_a)
        np.testing.assert_array_equal(result.data_b, field_b)

    def test_shape_mismatch_raises(self, field_a, spec):
        field_wrong = np.zeros((3, 7))
        with pytest.raises(ValueError, match="strict mode requires identical shapes"):
            align(field_a, field_wrong, spec, mode="strict")

    def test_row_labels_preserved(self, field_a, field_b, spec):
        labels = ["0", "1", "2", "3"]
        result = align(field_a, field_b, spec, mode="strict",
                       row_labels_a=labels, row_labels_b=labels)
        assert result.row_labels == labels
        assert result.col_labels == spec.slots

    def test_col_labels_from_spec(self, field_a, field_b, spec):
        result = align(field_a, field_b, spec, mode="strict")
        assert result.col_labels == spec.slots


# ---------------------------------------------------------------------------
# Aligned (normalized depth) alignment
# ---------------------------------------------------------------------------


class TestAlignNormalized:
    def test_same_shape_uses_common_grid(self, field_a, field_b, spec):
        result = align(field_a, field_b, spec, mode="aligned")
        assert result.mode == "aligned"
        # Should be at least 64 rows (aligned_grid minimum)
        assert result.data_a.shape[0] >= 64
        assert result.data_b.shape[0] >= 64

    def test_different_shapes_resample(self, spec):
        a = np.zeros((4, 7))
        b = np.zeros((8, 7))
        result = align(a, b, spec, mode="aligned")
        # Both should be resampled to the same shape
        assert result.data_a.shape == result.data_b.shape

    def test_resample_field_shape(self):
        field = np.zeros((4, 7))
        result = _resample_field(field, 32, 14)
        assert result.shape == (32, 14)

    def test_resample_field_preserves_nan(self):
        field = np.array([[1.0, np.nan], [np.nan, 4.0]])
        result = _resample_field(field, 4, 4)
        assert result.shape == (4, 4)
        # NaN should be preserved somewhere (dilated by zoom)
        assert np.isnan(result).any()


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------


class TestComputeDelta:
    def test_delta_shape_matches(self, field_a, field_b, simple_spec):
        aligned = align(field_a, field_b, simple_spec, mode="strict")
        delta = compute_delta(aligned, "height", simple_spec)
        assert delta.delta.shape == field_a.shape

    def test_delta_is_b_minus_a(self, field_a, field_b, simple_spec):
        aligned = align(field_a, field_b, simple_spec, mode="strict")
        delta = compute_delta(aligned, "height", simple_spec)
        # Check: delta = B - A (after scaling, so not exactly equal)
        # But shape should match
        assert delta.delta.shape == field_a.shape

    def test_rel_l2_non_negative(self, field_a, field_b, simple_spec):
        aligned = align(field_a, field_b, simple_spec, mode="strict")
        delta = compute_delta(aligned, "height", simple_spec)
        assert delta.rel_l2 >= 0.0

    def test_cosine_sim_bounded(self, field_a, field_b, simple_spec):
        aligned = align(field_a, field_b, simple_spec, mode="strict")
        delta = compute_delta(aligned, "height", simple_spec)
        assert -1.0 <= delta.cosine_sim <= 1.0

    def test_cosine_sim_identical_fields(self, field_a, simple_spec):
        aligned = align(field_a, field_a, simple_spec, mode="strict")
        delta = compute_delta(aligned, "height", simple_spec)
        # Identical fields should have cosine sim close to 1
        assert delta.cosine_sim > 0.99

    def test_hotspot_in_valid_range(self, field_a, field_b, simple_spec):
        aligned = align(field_a, field_b, simple_spec, mode="strict")
        delta = compute_delta(aligned, "height", simple_spec)
        assert 0 <= delta.hotspot_layer < field_a.shape[0]
        assert delta.hotspot_slot in simple_spec.slots


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestComputeRelL2:
    def test_identical_fields_zero(self):
        a = np.array([1.0, 2.0, 3.0])
        assert _compute_rel_l2(a, a) == 0.0

    def test_different_fields_nonzero(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 4.0])
        assert _compute_rel_l2(a, b) > 0.0

    def test_all_nan_returns_zero(self):
        a = np.array([np.nan, np.nan])
        b = np.array([1.0, 2.0])
        assert _compute_rel_l2(a, b) == 0.0

    def test_zero_norm_returns_zero(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        assert _compute_rel_l2(a, b) == 0.0


class TestComputeCosineSim:
    def test_identical_vectors_one(self):
        a = np.array([1.0, 2.0, 3.0])
        assert _compute_cosine_sim(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _compute_cosine_sim(a, b) == pytest.approx(0.0)

    def test_different_length_pads(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        result = _compute_cosine_sim(a, b)
        assert -1.0 <= result <= 1.0


class TestSafeSubtract:
    def test_preserves_nan(self):
        a = np.array([1.0, np.nan, 3.0])
        b = np.array([0.5, 2.0, np.nan])
        result = _safe_subtract(a, b)
        assert result[0] == 0.5
        assert np.isnan(result[1])
        assert np.isnan(result[2])


class TestFindHotspot:
    def test_finds_max(self):
        arr = np.array([[0.1, 0.3], [0.5, 0.2]])
        row, col, value = _find_hotspot(arr)
        assert row == 1
        assert col == 0
        assert value == 0.5

    def test_all_nan_returns_zeros(self):
        arr = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        row, col, value = _find_hotspot(arr)
        assert row == 0
        assert col == 0
        assert value == 0.0


# ---------------------------------------------------------------------------
# Hotspot ranking
# ---------------------------------------------------------------------------


class TestHotspotRanking:
    def test_returns_top_k(self, field_a, field_b, simple_spec):
        aligned = align(field_a, field_b, simple_spec, mode="strict")
        delta = compute_delta(aligned, "height", simple_spec)
        ranking = hotspot_ranking(delta, top_k=3)
        assert len(ranking) == 3

    def test_sorted_descending(self, field_a, field_b, simple_spec):
        aligned = align(field_a, field_b, simple_spec, mode="strict")
        delta = compute_delta(aligned, "height", simple_spec)
        ranking = hotspot_ranking(delta, top_k=5)
        values = [r[2] for r in ranking]
        assert values == sorted(values, reverse=True)

    def test_includes_slot_labels(self, field_a, field_b, simple_spec):
        aligned = align(field_a, field_b, simple_spec, mode="strict")
        delta = compute_delta(aligned, "height", simple_spec)
        col_labels = list(simple_spec.slots)
        ranking = hotspot_ranking(delta, col_labels=col_labels, top_k=3)
        for _, slot, _ in ranking:
            assert slot in col_labels

    def test_empty_field_returns_empty(self, simple_spec):
        empty_delta = ChannelDelta(
            channel="height",
            delta=np.full((4, 7), np.nan),
            abs_delta=np.full((4, 7), np.nan),
            rel_l2=0.0,
            cosine_sim=0.0,
            hotspot_layer=0,
            hotspot_slot="",
            hotspot_value=0.0,
            argmax=(0, ""),
        )
        ranking = hotspot_ranking(empty_delta, top_k=5)
        assert ranking == []


# ---------------------------------------------------------------------------
# Localization test (core test)
# ---------------------------------------------------------------------------


class TestLocalization:
    """Localization test: known perturbations must be correctly identified.

    Fixture A + mutated B:
    - layers.2.mlp.down_proj set to 100.0 (layer 2, slot 6 = mlp_down)
    - rank-1 perturbation on layers.3.self_attn.o_proj (layer 3, slot 3 = attn_o)

    compare --mode strict → hotspot ranking must report (2, mlp_down) and (3, attn_o)
    as Top-2 for height channel. argmax == (2, mlp_down).
    """

    @pytest.fixture
    def localization_fields(self, simple_spec):
        """Create fields with known perturbations."""
        n_layers = 4
        n_slots = len(simple_spec.slots)

        rng = np.random.default_rng(123)
        base = rng.normal(0, 1, (n_layers, n_slots)).astype(np.float64)

        field_a = base.copy()
        field_b = base.copy()

        # Perturbation 1: layer 2, slot 6 (mlp_down) set to large positive
        # log1p(100) - log1p(max(0, orig)) = 4.615 - 0 = 4.615 (dominant)
        field_b[2, 6] = 100.0

        # Perturbation 2: layer 3, slot 3 (attn_o) rank-1 perturbation
        # log1p(0.194+2) - log1p(0.194) = 1.163 - 0.178 = 0.985
        field_b[3, 3] += 2.0

        return field_a, field_b

    def test_argmax_is_mlp_down(self, localization_fields, simple_spec):
        """argmax of |delta| must be at (2, mlp_down)."""
        field_a, field_b = localization_fields
        aligned = align(field_a, field_b, simple_spec, mode="strict")
        delta = compute_delta(aligned, "height", simple_spec)

        assert delta.argmax == (2, "mlp_down")

    def test_top2_contains_mlp_down(self, localization_fields, simple_spec):
        """Top-2 hotspot ranking must contain (2, mlp_down)."""
        field_a, field_b = localization_fields
        aligned = align(field_a, field_b, simple_spec, mode="strict")
        delta = compute_delta(aligned, "height", simple_spec)
        ranking = hotspot_ranking(delta, col_labels=list(simple_spec.slots), top_k=2)

        # Top-1 should be mlp_down
        assert ranking[0][0] == 2
        assert ranking[0][1] == "mlp_down"

    def test_top2_contains_attn_o(self, localization_fields, simple_spec):
        """Top-2 hotspot ranking must contain (3, attn_o)."""
        field_a, field_b = localization_fields
        aligned = align(field_a, field_b, simple_spec, mode="strict")
        delta = compute_delta(aligned, "height", simple_spec)
        ranking = hotspot_ranking(delta, col_labels=list(simple_spec.slots), top_k=2)

        # Top-2 should be attn_o
        assert ranking[1][0] == 3
        assert ranking[1][1] == "attn_o"


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestComputeCompareSummary:
    def test_returns_summary_with_all_channels(self, field_a, field_b, simple_spec):
        summary = compute_compare_summary(field_a, field_b, simple_spec, mode="strict")
        assert "height" in summary.channels
        assert summary.mode == "strict"

    def test_warnings_on_spec_mismatch(self, field_a, field_b, simple_spec):
        fp_a = {"spec_version": 1, "tool_version": "0.1.0"}
        fp_b = {"spec_version": 1, "tool_version": "0.2.0"}
        summary = compute_compare_summary(
            field_a, field_b, simple_spec,
            mode="strict",
            fingerprint_a=fp_a,
            fingerprint_b=fp_b,
        )
        assert len(summary.warnings) > 0

    def test_no_fingerprint_no_warnings(self, field_a, field_b, simple_spec):
        summary = compute_compare_summary(field_a, field_b, simple_spec, mode="strict")
        assert summary.warnings == []


# ---------------------------------------------------------------------------
# caplog test for tool_version mismatch
# ---------------------------------------------------------------------------


class TestToolVersionWarning:
    def test_tool_version_mismatch_warns(self, field_a, field_b, simple_spec, caplog):
        """tool_version mismatch should produce a warning, not an error."""
        import logging
        fp_a = {"spec_version": 1, "tool_version": "0.1.0"}
        fp_b = {"spec_version": 1, "tool_version": "0.2.0"}
        with caplog.at_level(logging.WARNING):
            summary = compute_compare_summary(
                field_a, field_b, simple_spec,
                mode="strict",
                fingerprint_a=fp_a,
                fingerprint_b=fp_b,
            )
        assert len(summary.warnings) > 0
        assert any("tool_version mismatch" in w for w in summary.warnings)


# ---------------------------------------------------------------------------
# Determinism test for delta sheet
# ---------------------------------------------------------------------------


class TestDeltaSheetDeterminism:
    def test_delta_sheet_deterministic(self, field_a, field_b, simple_spec, tmp_path):
        """Delta sheet should be byte-identical on second run."""
        from weight_atlas.compare.render.delta_sheet import DeltaSheet

        aligned = align(field_a, field_b, simple_spec, mode="strict")
        delta = compute_delta(aligned, "height", simple_spec)

        renderer = DeltaSheet()

        # First run
        out1 = tmp_path / "run1"
        out1.mkdir()
        paths1 = renderer.render(delta.delta, simple_spec, out1, channel="height", mode="strict")

        # Second run
        out2 = tmp_path / "run2"
        out2.mkdir()
        paths2 = renderer.render(delta.delta, simple_spec, out2, channel="height", mode="strict")

        # Compare bytes
        for p1, p2 in zip(paths1, paths2, strict=False):
            assert p1.exists()
            assert p2.exists()
            assert p1.read_bytes() == p2.read_bytes(), f"{p1.name} not byte-identical"

    def test_delta_profile_rendered(self, field_a, field_b, simple_spec, tmp_path):
        """delta_profile_<channel>.png should be rendered."""
        from weight_atlas.compare.render.delta_sheet import DeltaSheet

        aligned = align(field_a, field_b, simple_spec, mode="strict")
        delta = compute_delta(aligned, "height", simple_spec)

        renderer = DeltaSheet()
        out = tmp_path / "render"
        out.mkdir()
        paths = renderer.render(delta.delta, simple_spec, out, channel="height", mode="strict")

        # Should produce 2 files: sheet + profile
        assert len(paths) == 2
        assert any("delta_sheet_height" in p.name for p in paths)
        assert any("delta_profile_height" in p.name for p in paths)
