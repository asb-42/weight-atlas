"""Scatter + records model tabs (P2): routing, data correctness, determinism."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from weight_atlas.api.jobs import Job, JobQueue, JobStatus
from weight_atlas.api.main import create_app


@pytest.fixture
def scatter_env(tmp_path: Path):
    """App + DONE scan job with a synthetic fingerprint of known metrics."""
    db_path = tmp_path / "jobs.db"
    output_root = tmp_path / "output"
    output_root.mkdir(exist_ok=True)
    app = create_app(db_path=db_path, spec_path=None, output_root=output_root)

    queue = JobQueue(db_path, on_job=lambda j: None)
    now = "2026-01-01T00:00:00"
    job = Job(
        job_id="11111111-2222-3333-4444-555555555555",
        model_path=str(tmp_path / "fake.safetensors"),
        out_dir=str(tmp_path / "scan_out"),
        spec_path="",
        status=JobStatus.DONE,
        created_at=now,
        updated_at=now,
        message="Complete",
        job_type="scan",
    )
    queue._save(job)  # noqa: SLF001
    out_dir = Path(job.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 6 tensors with hand-computable metric values; attn/mlp/embed slots for
    # color grouping; one tensor missing metrics entirely (old-scan case).
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": {
            "shape": [16, 16], "kurtosis": 1.0, "spectral_norm": 10.0,
            "sqnr_int4_g128": 20.0, "row_amax_ratio": 5.0, "sv_decay": 0.01,
        },
        "model.layers.1.self_attn.q_proj.weight": {
            "shape": [16, 16], "kurtosis": 3.0, "spectral_norm": 30.0,
            "sqnr_int4_g128": 25.0, "row_amax_ratio": 9.0, "sv_decay": 0.02,
        },
        "model.layers.0.mlp.down_proj.weight": {
            "shape": [16, 16], "kurtosis": 2.0, "spectral_norm": 20.0,
            "sqnr_int4_g128": 15.0, "row_amax_ratio": 2.0, "sv_decay": 0.05,
        },
        "model.embed_tokens.weight": {
            "shape": [16, 16], "kurtosis": 8.0, "spectral_norm": 5.0,
            "sqnr_int4_g128": 30.0, "row_amax_ratio": 1.5, "sv_decay": 0.4,
        },
        "model.layers.2.mlp.gate_proj.weight": {
            "shape": [16, 16], "kurtosis": 0.5, "spectral_norm": 2.0,
            "sqnr_int4_g128": 12.0, "row_amax_ratio": 3.0, "sv_decay": 0.2,
        },
        "model.layers.3.self_attn.k_proj.weight": {"shape": [16, 16]},
    }
    (out_dir / "fingerprint.json").write_text(json.dumps({"tensors": tensors}))

    client = TestClient(app)
    return client, job.job_id


class TestRecordsTab:
    def test_tab_renders_cards(self, scatter_env) -> None:
        client, job_id = scatter_env
        resp = client.get(f"/models/{job_id}?tab=records")
        assert resp.status_code == 200
        # most kurtotic = embed (8.0), most fragile INT4 = gate_proj (12.0)
        assert "Heaviest tails" in resp.text
        assert "model.embed_tokens.weight" in resp.text
        assert "Most fragile under INT4-g128" in resp.text
        assert "model.layers.2.mlp.gate_proj.weight" in resp.text
        # old-scan tensor (no metrics) must not break anything
        assert "model.layers.3.self_attn.k_proj.weight" not in resp.text

    def test_skips_absent_metrics(self, scatter_env) -> None:
        """sqnr_int8_ch absent from the fixture → its board card must not render."""
        client, job_id = scatter_env
        resp = client.get(f"/models/{job_id}?tab=records")
        assert "Most fragile under INT8" not in resp.text

    def test_deterministic_output(self, scatter_env) -> None:
        client, job_id = scatter_env
        a = client.get(f"/models/{job_id}?tab=records")
        b = client.get(f"/models/{job_id}?tab=records")
        assert a.text == b.text


class TestScatterTab:
    def test_tab_renders_svg(self, scatter_env) -> None:
        client, job_id = scatter_env
        resp = client.get(f"/models/{job_id}?tab=scatter")
        assert resp.status_code == 200
        assert "<svg" in resp.text
        assert "<circle" in resp.text
        # the metric-less tensor is excluded; 5 dots remain
        assert resp.text.count("<circle") == 5
        # axis labels name the metrics
        assert "kurtosis" in resp.text and "sqnr_int4_g128" in resp.text

    def test_xy_params_select_axes(self, scatter_env) -> None:
        client, job_id = scatter_env
        resp = client.get(f"/models/{job_id}?tab=scatter&x=spectral_norm&y=kurtosis")
        assert resp.status_code == 200
        assert "spectral_norm" in resp.text

    def test_invalid_xy_falls_back(self, scatter_env) -> None:
        client, job_id = scatter_env
        resp = client.get(f"/models/{job_id}?tab=scatter&x=not_a_metric&y=also_bad")
        assert resp.status_code == 200
        assert "<svg" in resp.text

    def test_deterministic_svg(self, scatter_env) -> None:
        client, job_id = scatter_env
        a = client.get(f"/models/{job_id}?tab=scatter&x=kurtosis&y=spectral_norm")
        b = client.get(f"/models/{job_id}?tab=scatter&x=kurtosis&y=spectral_norm")
        assert a.text == b.text

    def test_log_axis_for_wide_span(self, scatter_env) -> None:
        """spectral_norm spans 2..30 (15×) → linear; a 1000× span must switch log."""
        client, job_id = scatter_env
        resp = client.get(f"/models/{job_id}?tab=scatter&x=spectral_norm&y=kurtosis")
        assert "(log)" not in resp.text.split("<svg")[1].split("</svg>")[0]

    def test_cap_culls_deterministically(self, scatter_env, tmp_path: Path) -> None:
        """>cap pairs → rendered <= cap (stride of a sorted list), deterministic."""
        client, job_id = scatter_env
        # overwrite fingerprint with cap+1 tensors
        from weight_atlas.api.query import SCATTER_CAP

        tensors = {
            f"model.layers.{i % 4}.t{i:05d}.weight": {
                "shape": [4, 4], "kurtosis": float(i), "spectral_norm": float(i % 7 + 1),
            }
            for i in range(SCATTER_CAP + 1)
        }
        fp_path = tmp_path / "scan_out" / "fingerprint.json"
        fp_path.write_text(json.dumps({"tensors": tensors}))
        resp = client.get(f"/models/{job_id}?tab=scatter")
        assert resp.status_code == 200
        assert f"of {SCATTER_CAP + 1} tensors" in resp.text
        n_circles = resp.text.count("<circle")
        assert n_circles <= SCATTER_CAP
        # deterministic: same request → same subset
        resp2 = client.get(f"/models/{job_id}?tab=scatter")
        assert resp2.text == resp.text


class TestQueryHelpers:
    def test_extreme_records_ordering(self) -> None:
        from weight_atlas.api.query import extreme_records

        records = [
            {"tensor_name": "b", "kurtosis": 2.0},
            {"tensor_name": "a", "kurtosis": 2.0},  # tie → name breaks
            {"tensor_name": "c", "kurtosis": float("nan")},  # filtered
            {"tensor_name": "d", "kurtosis": 9.0},
        ]
        top = extreme_records(records, "kurtosis", "max")
        assert [r["tensor_name"] for r in top] == ["d", "a", "b"]
        low = extreme_records(records, "kurtosis", "min", limit=1)
        assert [r["tensor_name"] for r in low] == ["a"]

    def test_scatter_points_log_axis(self) -> None:
        from weight_atlas.api.query import scatter_points

        records = [
            {"tensor_name": f"t{i:03d}", "slot": "attn", "layer": 0, "numel": 100,
             "a": float(10 ** (i / 50)), "b": 1.0}
            for i in range(0, 300, 3)  # span 10^6 ≫ 100 → log axis
        ]
        data = scatter_points(records, "a", "b")
        assert data["x_axis"]["log"] is True
        assert data["y_axis"]["log"] is False  # constant axis
        assert data["rendered"] == 100

    def test_scatter_points_cap_stride(self) -> None:
        from weight_atlas.api.query import scatter_points

        records = [
            {"tensor_name": f"t{i:05d}", "slot": "attn", "layer": 0, "numel": 1,
             "a": float(i), "b": 1.0}
            for i in range(1000)
        ]
        data = scatter_points(records, "a", "b", cap=100)
        assert data["total"] == 1000
        assert data["rendered"] == 100
        names = [p["tensor_name"] for p in data["points"]]
        assert names == sorted(names)  # stride of a sorted list stays sorted

    def test_scatter_points_empty(self) -> None:
        from weight_atlas.api.query import scatter_points

        data = scatter_points([], "a", "b")
        assert data["points"] == [] and data["total"] == 0
        assert math.isfinite(data["x_axis"]["lo"])
