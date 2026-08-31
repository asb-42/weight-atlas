"""Distribution-shape statistics (alesha-pro adoption): summary, ratios, sv_decay."""

from __future__ import annotations

import numpy as np
import pytest

from weight_atlas.core.types import TensorHandle
from weight_atlas.stats.distribution import (
    _SAMPLE_CAP,
    amax_ratios,
    distribution_summary,
)
from weight_atlas.stats.norms import SVDecay


def _handle(x: np.ndarray, name: str = "t") -> TensorHandle:
    return TensorHandle(name, x.shape, "float32", lambda: x)


def test_summary_matches_reference_computation() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal((64, 32)).astype(np.float32)
    s = distribution_summary(_handle(x), seed=0)

    flat = x.astype(np.float64).reshape(-1)
    assert s["mean"] == pytest.approx(float(flat.mean()), abs=1e-6)
    assert s["std"] == pytest.approx(float(flat.std()), abs=1e-6)
    assert s["absmax"] == pytest.approx(float(np.abs(flat).max()), abs=1e-5)
    assert s["absmean"] == pytest.approx(float(np.abs(flat).mean()), abs=1e-6)
    a = np.abs(flat)
    for key, q in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("p999", 0.999), ("p9999", 0.9999)):
        assert s[key] == pytest.approx(float(np.quantile(a, q)), rel=1e-3)
    d = np.abs(flat - flat.mean())
    assert s["outlier_3s"] == pytest.approx(float((d > 3 * flat.std()).mean()), rel=0.2)
    assert s["dyn_range"] == pytest.approx(float(np.abs(flat).max() / np.quantile(a, 0.5)), rel=1e-3)


def test_summary_deterministic_and_sampled_identically() -> None:
    """Same (input, seed) → identical summary; large tensors use the seeded draw."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((512, 512)).astype(np.float32)
    h = _handle(x)
    s1 = distribution_summary(h, seed=0)
    s2 = distribution_summary(h, seed=0)
    assert s1 == s2  # cached + deterministic

    # Above the cap the seeded subsample must be used — deterministic across
    # fresh handles (fresh caches) with the same seed.
    big = rng.standard_normal(_SAMPLE_CAP + 1).astype(np.float32)
    a = distribution_summary(_handle(big, "a"), seed=42)
    b = distribution_summary(_handle(big, "b"), seed=42)
    assert a["p99"] == b["p99"]
    assert a["p9999"] == b["p9999"]
    # And a different seed gives a different (but close) percentile estimate.
    c = distribution_summary(_handle(big, "c"), seed=43)
    assert c["p99"] != a["p99"]
    assert c["p99"] == pytest.approx(a["p99"], rel=0.05)


def test_amax_ratios_and_na_discipline() -> None:
    rng = np.random.default_rng(11)
    # One outlier row: row ratio large, column ratio moderate
    x = rng.standard_normal((128, 64)).astype(np.float32)
    x[17, :] *= 50.0
    row_r, col_r = amax_ratios(_handle(x))
    assert row_r > 20.0, f"outlier row must dominate the row ratio, got {row_r}"
    assert col_r < 10.0

    # NaN, never zero, for non-2-D tensors (1-D, 3-D, 4-D)
    for shape in [(128,), (4, 8, 8), (2, 4, 4, 4)]:
        t = _handle(rng.standard_normal(shape).astype(np.float32))
        r, c = amax_ratios(t)
        assert np.isnan(r) and np.isnan(c)
    # sv_decay shares the discipline
    assert np.isnan(SVDecay().compute(_handle(rng.standard_normal(64).astype(np.float32))))


def test_sv_decay_matches_spectrum() -> None:
    rng = np.random.default_rng(5)
    x = rng.standard_normal((48, 24)).astype(np.float32)
    t = _handle(x)
    from weight_atlas.stats.spectrum import truncated_spectrum

    s = truncated_spectrum(t, seed=0)
    assert SVDecay(seed=0).compute(t) == pytest.approx(float(s[-1] / s[0]), rel=1e-5)
    # Low-rank matrix: decay near zero (energy in top modes)
    low = (np.outer(rng.standard_normal(32), rng.standard_normal(32))).astype(np.float32)
    assert SVDecay().compute(_handle(low)) < 1e-2


def test_registered_stat_classes_compute() -> None:
    """The stat classes behind the registry decorators compute and carry the
    canonical stat_ids. The per-field summary stats are anonymous (built by
    _make_summary_stat) with ``_DISTRIBUTION_STATS`` as their source of
    truth; registry-population itself is exercised by the scan pipeline
    (scan.py imports these modules → side-effect decorators register)."""
    from weight_atlas.stats import distribution as dist_mod
    from weight_atlas.stats import norms as norms_mod

    ones = np.ones((4, 4), dtype=np.float32)
    h = _handle(ones)
    assert dist_mod.RowAmaxRatio.stat_id == "row_amax_ratio"
    assert dist_mod.ColAmaxRatio.stat_id == "col_amax_ratio"
    assert dist_mod.RowAmaxRatio().compute(h) == pytest.approx(1.0)
    assert dist_mod.ColAmaxRatio().compute(h) == pytest.approx(1.0)
    assert norms_mod.SVDecay.stat_id == "sv_decay"

    assert [fid for fid, _ in dist_mod._DISTRIBUTION_STATS] == [
        "mean", "std", "absmax", "absmean",
        "p50", "p90", "p99", "p999", "p9999",
        "outlier_3s", "outlier_4s", "outlier_6s",
        "dyn_range",
    ]
    # ones → all values 1: mean/absmax/absmean/percentiles/dyn_range = 1,
    # std = 0, outliers 0 (std == 0 → outlier counts defined-but-zero).
    s = distribution_summary(h, seed=0)
    for fid in ("mean", "absmax", "absmean", "p50", "p90", "p99", "p999", "p9999", "dyn_range"):
        assert s[fid] == pytest.approx(1.0)
    assert s["std"] == 0.0
    for fid in ("outlier_3s", "outlier_4s", "outlier_6s"):
        assert s[fid] == 0.0
