"""Tests for the records-tab outlier visualizations (OCGQuant framing).

Covers the pure query helpers (distribution_strip, outlier_impact,
layer_profile), the deterministic SVG renderers, and the records page
integration (200 + visualization content + byte-determinism).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from weight_atlas.api.jobs import JobQueue
from weight_atlas.api.main import create_app
from weight_atlas.api.query import (
    distribution_strip,
    layer_profile,
    outlier_impact,
)
from weight_atlas.api.routes import _impact_svg, _profile_svg, _strip_svg


def _rec(
    name: str,
    slot: str,
    layer: int,
    row: float | None,
    col: float | None = None,
) -> dict:
    return {
        "tensor_name": name,
        "slot": slot,
        "layer": layer,
        "shape": [4, 4],
        "row_amax_ratio": row,
        "col_amax_ratio": col,
    }


RECORDS = [
    _rec("model.embed_tokens.weight", "embed", -1, 1.814, 1.364),
    _rec("blk.0.attn_q.weight", "attn_q", 0, 4.0, 2.0),
    _rec("blk.1.mlp_down.weight", "mlp_down", 1, 12.5, 2.1),
    _rec("blk.1.attn_q.weight", "attn_q", 1, 8.2, 3.4),
    _rec("blk.2.attn_q.weight", "attn_q", 2, 6.0, 3.0),
]


class TestOutlierImpact:
    def test_ranking_by_dominance(self) -> None:
        rows = outlier_impact(RECORDS, limit=10)
        assert [r["dominance"] for r in rows] == [12.5, 8.2, 6.0, 4.0, 1.814]
        assert rows[0]["tensor_name"] == "blk.1.mlp_down.weight"

    def test_limit(self) -> None:
        assert len(outlier_impact(RECORDS, limit=2)) == 2

    def test_skips_nonfinite_and_absent(self) -> None:
        recs = [
            _rec("a", "attn_q", 0, None, None),
            _rec("b", "attn_q", 1, float("nan"), 2.0),
            _rec("c", "attn_q", 2, 3.0),
        ]
        rows = outlier_impact(recs)
        assert [r["tensor_name"] for r in rows] == ["c", "b"]

    def test_deterministic_tiebreak_by_name(self) -> None:
        recs = [_rec(f"t{i}", "attn_q", i, 5.0) for i in range(4)]
        rows = outlier_impact(recs)
        assert [r["tensor_name"] for r in rows] == ["t0", "t1", "t2", "t3"]

    def test_empty(self) -> None:
        assert outlier_impact([]) == []


class TestLayerProfile:
    def test_per_layer_max_and_ordering(self) -> None:
        profile = layer_profile(RECORDS, "row_amax_ratio")
        assert profile["metric"] == "row_amax_ratio"
        layers = profile["layers"]
        assert [entry["layer"] for entry in layers] == [0, 1, 2]
        assert layers[1]["max"] == 12.5
        assert layers[1]["worst_tensor"] == "blk.1.mlp_down.weight"

    def test_excludes_layerless(self) -> None:
        profile = layer_profile(RECORDS, "row_amax_ratio")
        assert all(entry["layer"] >= 0 for entry in profile["layers"])

    def test_empty(self) -> None:
        profile = layer_profile([_rec("model.embed_tokens.weight", "embed", -1, 2.0)], "row_amax_ratio")
        assert profile["layers"] == []


class TestDistributionStrip:
    def test_percentile_and_quantiles(self) -> None:
        strip = distribution_strip(RECORDS, "row_amax_ratio", 12.5)
        assert strip is not None
        assert strip["n"] == 5
        assert strip["percentile"] == pytest.approx(80.0)  # 4 of 5 below
        assert strip["p50"] == 6.0
        assert strip["value"] == 12.5
        assert strip["log"] is False  # span < 2 decades

    def test_log_when_span_wide(self) -> None:
        recs = [_rec("a", "attn_q", 0, 1.0), _rec("b", "attn_q", 1, 900.0)]
        strip = distribution_strip(recs, "row_amax_ratio", 900.0)
        assert strip is not None and strip["log"] is True

    def test_metric_absent_returns_none(self) -> None:
        assert distribution_strip(RECORDS, "sqnr_int4_g128", 1.0) is None

    def test_deterministic(self) -> None:
        a = distribution_strip(RECORDS, "row_amax_ratio", 12.5)
        b = distribution_strip(RECORDS, "row_amax_ratio", 12.5)
        assert a == b


class TestSvgRenderers:
    def test_impact_svg_shape_and_determinism(self) -> None:
        impact = outlier_impact(RECORDS)
        svg = _impact_svg(impact)
        assert svg.startswith("<svg") and svg.endswith("</svg>")
        assert svg.count("<rect") >= 2 * len(impact)  # track + bar per row
        assert _impact_svg(impact) == svg  # byte-deterministic

    def test_impact_svg_empty(self) -> None:
        assert _impact_svg([]) == ""

    def test_profile_svg_marks_peak(self) -> None:
        svg = _profile_svg(layer_profile(RECORDS, "row_amax_ratio"))
        assert "peak" in svg and "<circle" in svg

    def test_profile_svg_needs_two_layers(self) -> None:
        one = layer_profile(RECORDS[1:2], "row_amax_ratio")  # single layered tensor
        assert len(one["layers"]) == 1
        assert _profile_svg(one) == ""

    def test_strip_svg_content(self) -> None:
        strip = distribution_strip(RECORDS, "row_amax_ratio", 12.5)
        assert strip is not None
        svg = _strip_svg(strip)
        assert "leader" in svg and "p99" in svg

    def test_strip_svg_log_mode(self) -> None:
        recs = [_rec("a", "attn_q", 0, 1.0), _rec("b", "attn_q", 1, 900.0)]
        strip = distribution_strip(recs, "row_amax_ratio", 900.0)
        assert strip is not None
        assert _strip_svg(strip) == _strip_svg(strip)


@pytest.fixture
def scan_dir(tmp_path: Path) -> Path:
    """Scan output with amax metrics + layer structure for the records page."""
    tensors = {}
    for layer in range(4):
        for slot, row, col in (("attn_q", 3.0 + layer, 2.0), ("mlp_down", 8.0 + layer, 1.5)):
            tensors[f"blk.{layer}.{slot}.weight"] = {
                "shape": [16, 16],
                "row_amax_ratio": row,
                "col_amax_ratio": col,
            }
    tensors["token_embd.weight"] = {"shape": [100, 16], "row_amax_ratio": 1.814, "col_amax_ratio": 1.4}
    fp = {
        "spec_version": 2,
        "tool_version": "0.2.0",
        "loader": "safetensors",
        "model": {"n_tensors": len(tensors), "n_layers": 4},
        "quantization": {"F16": len(tensors)},
        "mapping_coverage": {"in_slots": 1.0, "unmapped": 0, "unmapped_tensors": []},
        "tensors": tensors,
    }
    scan = tmp_path / "scan_viz"
    scan.mkdir()
    with open(scan / "fingerprint.json", "w") as f:
        json.dump(fp, f)
    return scan


@pytest.fixture
def client(tmp_path: Path, scan_dir: Path) -> TestClient:
    queue = JobQueue(tmp_path / "viz.db", on_job=lambda j: None)
    queue.import_scan(scan_dir, model_path=str(scan_dir))
    app = create_app(db_path=Path(queue._db_path), output_root=tmp_path / "out")
    return TestClient(app)


class TestRecordsPageIntegration:
    def test_page_renders_visualizations(self, client: TestClient) -> None:
        model_id = client.get("/api/models").json()["scans"][0]["model_id"]
        resp = client.get(f"/models/{model_id}?tab=records")
        assert resp.status_code == 200
        body = resp.text
        assert "Outlier impact" in body
        assert "Depth profile" in body
        assert "impact-svg" in body
        assert "profile-svg" in body
        assert "strip-svg" in body
        # the outlier boards still render alongside the visualizations
        assert "Worst outlier channels" in body

    def test_page_byte_deterministic(self, client: TestClient) -> None:
        model_id = client.get("/api/models").json()["scans"][0]["model_id"]
        first = client.get(f"/models/{model_id}?tab=records").text
        second = client.get(f"/models/{model_id}?tab=records").text
        assert first == second
