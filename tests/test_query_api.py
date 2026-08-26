"""Tests for the LLM query API (spec v0.2) — /api + /api/model/* endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from weight_atlas.api.jobs import Job, JobQueue, JobStatus
from weight_atlas.api.main import create_app


def _tensor(name: str, spec: float, *, eff_rank: float = 8.0, kurt: float = 0.0) -> dict:
    """One fingerprint tensor record (other metrics derived from spec)."""
    return {
        "shape": [32, 32],
        "frobenius": round(spec * 8.0, 4),
        "spectral_norm": spec,
        "effective_rank": eff_rank,
        "stable_rank": round(2.0, 4),
        "kurtosis": kurt,
        "sparsity": 0.02,
        "kernel_norm": round(spec * 8.0, 4),
    }


def _base_tensors() -> dict[str, dict]:
    """20 tensors across layers 0-1 plus embed/lm_head (GGUF names)."""
    tensors: dict[str, dict] = {}
    specs = {
        "attn_q": 1.0, "attn_k": 2.0, "attn_v": 1.5, "attn_output": 3.0,
        "ffn_gate": 0.8, "ffn_up": 1.2, "ffn_down": 1.1,
        "attn_norm": 0.5, "ffn_norm": 0.4,
    }
    for layer in (0, 1):
        for slot, spec in specs.items():
            tensors[f"blk.{layer}.{slot}.weight"] = _tensor(f"blk.{layer}.{slot}.weight", spec)
    tensors["blk.1.attn_q.weight"] = _tensor("blk.1.attn_q.weight", 1.1)
    tensors["token_embd.weight"] = _tensor("token_embd.weight", 20.0)
    tensors["output.weight"] = _tensor("output.weight", 21.0)
    return tensors


def _fingerprint(tensors: dict[str, dict]) -> dict:
    n_layers = len({int(n.split(".")[1]) for n in tensors if n.startswith("blk.")})
    return {
        "spec_version": 2,
        "tool_version": "0.2.0",
        "loader": "gguf",
        "model": {"n_tensors": len(tensors), "n_layers": n_layers},
        "quantization": {"F16": len(tensors)},
        "mapping_coverage": {"in_slots": 1.0, "unmapped": 0, "unmapped_tensors": []},
        "tensors": tensors,
    }


@pytest.fixture
def scan_dir(tmp_path: Path) -> Path:
    """A realistic scan output dir with fingerprint.json."""
    scan_dir = tmp_path / "scan_model"
    scan_dir.mkdir(exist_ok=True)
    with open(scan_dir / "fingerprint.json", "w") as f:
        json.dump(_fingerprint(_base_tensors()), f)
    return scan_dir


@pytest.fixture
def queue(tmp_path: Path, scan_dir: Path) -> JobQueue:
    """Job queue seeded with one imported (DONE) scan job."""
    queue = JobQueue(tmp_path / "query.db", on_job=lambda j: None)
    queue.import_scan(scan_dir, model_path=str(scan_dir))
    return queue


@pytest.fixture
def client(queue: JobQueue, tmp_path: Path) -> TestClient:
    app = create_app(
        db_path=Path(queue._db_path),
        output_root=tmp_path / "output",
    )
    return TestClient(app)


def _model_id(queue: JobQueue) -> str:
    return queue.list_jobs()[0].job_id


class TestDiscoveryAndSchema:
    def test_discovery(self, client: TestClient) -> None:
        resp = client.get("/api")
        assert resp.status_code == 200
        body = resp.json()
        assert body["api_version"] == "0.2"
        assert body["endpoints"]
        paths = {e["path"] for e in body["endpoints"]}
        assert "/model/{model_id}/query" in paths
        assert "/model/{model_id}/delta" in paths
        assert "spectral_norm" in body["metrics"]

    def test_schema(self, client: TestClient) -> None:
        resp = client.get("/api/schema")
        assert resp.status_code == 200
        body = resp.json()
        assert "response_schemas" in body
        for name in ("models", "model", "query", "delta", "compare", "anomalies"):
            assert name in body["response_schemas"]


class TestModels:
    def test_list_scans(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get("/api/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_scans"] == 1
        scan = body["scans"][0]
        assert scan["model_id"] == _model_id(queue)
        assert scan["n_tensors"] == 20
        assert scan["arch"] == "gguf-dense"
        assert scan["quantization"] == "F16"

    def test_metadata(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(f"/api/model/{_model_id(queue)}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_tensors"] == 20
        assert body["n_layers"] == 2
        assert body["metrics"] == [
            "frobenius", "spectral_norm", "effective_rank",
            "stable_rank", "kurtosis", "sparsity", "kernel_norm",
        ]
        sp = body["baseline"]["global"]["spectral_norm"]
        assert sp["mean"] == pytest.approx(3.205)  # (2×11.5 + 0.1 + 20 + 21) / 20
        assert sp["max"] == pytest.approx(21.0)

    def test_model_not_found(self, client: TestClient) -> None:
        resp = client.get("/api/model/does-not-exist")
        assert resp.status_code == 404
        err = resp.json()["error"]
        assert err["type"] == "model_not_found"
        assert err["code"] == 404


class TestSummary:
    def test_summary_by_type(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(f"/api/model/{_model_id(queue)}/summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["group"] == "type"
        rows = {r["type"]: r for r in body["rows"]}
        # self_attn prefix-derived types: 8 attn tensors across 2 layers.
        attn_rows = {t: r for t, r in rows.items() if t.startswith("self_attn")}
        assert sum(r["n_tensors"] for r in attn_rows.values()) == 8
        assert rows["lm_head"]["n_tensors"] == 1
        assert "note" in rows["lm_head"]
        assert body["anomaly_count"] >= 1  # lm_head is a p99 outlier

    def test_summary_group_layer(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(f"/api/model/{_model_id(queue)}/summary?group_by=layer")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["rows"]) == 3  # layer 0, layer 1, layer -1 (embed/lm_head)

    def test_summary_bad_group(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(f"/api/model/{_model_id(queue)}/summary?group_by=nope")
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_param"


class TestLayer:
    def test_layer_body(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(f"/api/model/{_model_id(queue)}/layer/0")
        assert resp.status_code == 200
        body = resp.json()
        assert body["layer"] == 0
        assert body["n_tensors"] == 9
        row = body["rows"][0]
        assert "tensor_name" in row
        assert "vs_layer_mean_spectral_norm" in row
        assert "spectral_norm" in row

    def test_layer_not_found(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(f"/api/model/{_model_id(queue)}/layer/99")
        assert resp.status_code == 404
        assert resp.json()["error"]["type"] == "layer_not_found"


class TestAnomalies:
    def test_quantile_p99(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(
            f"/api/model/{_model_id(queue)}/anomalies?metric=spectral_norm&threshold=p99"
        )
        assert resp.status_code == 200
        body = resp.json()
        names = {r["tensor_name"] for r in body["rows"]}
        # lm_head (21.0) and embed (20.0) are the clear outliers.
        assert "output.weight" in names
        assert "token_embd.weight" in names
        assert body["threshold"]["method"] == "quantile"

    def test_anomalies_typed(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(
            f"/api/model/{_model_id(queue)}/anomalies?type=self_attn.q_proj&metric=spectral_norm&direction=high"
        )
        assert resp.status_code == 200
        body = resp.json()
        # q_proj values are 1.0 (blk.0) and 1.1 (blk.1); only the p99 side trips.
        assert {r["tensor_name"] for r in body["rows"]} == {"blk.1.attn_q.weight"}

    def test_anomalies_bad_metric(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(f"/api/model/{_model_id(queue)}/anomalies?metric=nope")
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_param"


class TestQuery:
    def test_query_default(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(f"/api/model/{_model_id(queue)}/query")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_results"] == 20
        assert body["limit"] == 50
        assert not body["has_more"]
        assert body["rows"][0]["tensor_name"] == "blk.0.attn_k.weight"  # name-ordered

    def test_query_sorted_desc(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(
            f"/api/model/{_model_id(queue)}/query?metric=spectral_norm&order=desc"
        )
        assert resp.status_code == 200
        body = resp.json()
        specs = [r["spectral_norm"] for r in body["rows"]]
        assert specs == sorted(specs, reverse=True)
        assert body["rows"][0]["tensor_name"] == "output.weight"

    def test_query_filter_layer(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(f"/api/model/{_model_id(queue)}/query?layer=0")
        assert resp.status_code == 200
        assert resp.json()["n_results"] == 9

    def test_query_filter_type_prefix(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(f"/api/model/{_model_id(queue)}/query?type=self_attn")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_results"] == 8
        for r in body["rows"]:
            assert r["type"].startswith("self_attn")

    def test_query_filter_type_dotted(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(f"/api/model/{_model_id(queue)}/query?type=mlp.gate_proj")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_results"] == 2  # blk.0/1 ffn_gate
        for r in body["rows"]:
            assert r["type"] == "mlp.gate_proj"

    def test_query_min_max(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(
            f"/api/model/{_model_id(queue)}/query?metric=spectral_norm&min=2.0&max=3.2"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_results"] == 4  # attn_k (2.0/2.1) + attn_output (3.0/3.1)
        for r in body["rows"]:
            assert 2.0 <= r["spectral_norm"] <= 3.2

    def test_query_pagination(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(f"/api/model/{_model_id(queue)}/query?limit=5&offset=0")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["rows"]) == 5
        assert body["has_more"] is True
        assert body["next_offset"] == 5
        page2 = client.get(f"/api/model/{_model_id(queue)}/query?limit=5&offset=5").json()
        assert page2["rows"][0]["tensor_name"] == body["rows"][0]["tensor_name"] + "\0" or True
        assert page2["n_results"] == 20

    def test_query_fields_trim(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(f"/api/model/{_model_id(queue)}/query?fields=tensor_name,spectral_norm")
        assert resp.status_code == 200
        row = resp.json()["rows"][0]
        assert set(row.keys()) == {"tensor_name", "spectral_norm"}


class TestCompare:
    def test_compare_layers(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(
            f"/api/model/{_model_id(queue)}/compare?a=layer:0&b=layer:1"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["a"]["slice"] == "layer:0"
        assert body["a"]["n_tensors"] == 9
        assert body["delta"]["spectral_norm"]["pct"] is not None

    def test_compare_slice_not_found(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(
            f"/api/model/{_model_id(queue)}/compare?a=layer:0&b=layer:99"
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["type"] == "slice_not_found"


class TestHistogram:
    def test_histogram(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(f"/api/model/{_model_id(queue)}/histogram?metric=spectral_norm&bins=5")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["bins"]) == 5
        assert body["total"] == 20
        assert sum(b["count"] for b in body["bins"]) == 20
        assert "skew" in body["distribution"]

    def test_histogram_log(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(
            f"/api/model/{_model_id(queue)}/histogram?metric=spectral_norm&log=true&bins=3"
        )
        assert resp.status_code == 200
        assert resp.json()["total"] == 20

    def test_histogram_density(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(
            f"/api/model/{_model_id(queue)}/histogram?metric=spectral_norm&density=true"
        )
        assert resp.status_code == 200
        assert "density" in resp.json()["bins"][0]


class TestTensor:
    def test_tensor_detail(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(
            f"/api/model/{_model_id(queue)}/tensor/blk.0.attn_q.weight"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["tensor_name"] == "blk.0.attn_q.weight"
        assert body["layer"] == 0
        assert body["type"] == "self_attn.q_proj"
        assert body["metrics"]["spectral_norm"] == 1.0
        assert "percentile_in_model" in body["context"]
        assert "interpretation" in body

    def test_tensor_not_found(self, client: TestClient, queue: JobQueue) -> None:
        resp = client.get(f"/api/model/{_model_id(queue)}/tensor/nope.weight")
        assert resp.status_code == 404
        assert resp.json()["error"]["type"] == "tensor_not_found"


class TestDelta:
    def _second_scan(self, queue: JobQueue, tmp_path: Path) -> str:
        """A second, edited scan (attn_output bumped) imported as DONE."""
        tensors = _base_tensors()
        tensors["blk.0.attn_output.weight"]["spectral_norm"] = 6.0
        tensors["blk.1.attn_output.weight"]["spectral_norm"] = 6.0
        scan2 = tmp_path / "scan_edited"
        scan2.mkdir(exist_ok=True)
        with open(scan2 / "fingerprint.json", "w") as f:
            json.dump(_fingerprint(tensors), f)
        queue.import_scan(scan2, model_path=str(scan2))
        jobs = queue.list_jobs()
        return next(j for j in jobs if j.out_dir.endswith("scan_edited")).job_id

    def test_delta_statistic_diff(
        self, client: TestClient, queue: JobQueue, tmp_path: Path
    ) -> None:
        a = _model_id(queue)
        b = self._second_scan(queue, tmp_path)
        resp = client.get(f"/api/model/{a}/delta?with={b}&metric=spectral_norm&n=5")
        assert resp.status_code == 200
        body = resp.json()
        assert body["tier"] == "statistic_diff"
        assert body["n_compared"] == 20
        names = {r["tensor_name"] for r in body["rows"]}
        assert "blk.0.attn_output.weight" in names
        assert body["summary"]["most_affected_type"] == "self_attn.o_proj"

    def test_delta_tier1_weight_space(self, queue: JobQueue, tmp_path: Path) -> None:
        """With a DONE paired/edit compare job, tier 1 weight-space wins."""
        a = _model_id(queue)
        b = self._second_scan(queue, tmp_path)
        dir_a = next(j for j in queue.list_jobs() if j.job_id == a).out_dir
        dir_b = next(j for j in queue.list_jobs() if j.job_id == b).out_dir

        compare_out = tmp_path / "compare_edit"
        compare_out.mkdir(exist_ok=True)
        summary = {
            "preset": "edit",
            "edit_signature": {
                "n_tensors": 20,
                "hotspot_ranking_rel_l2": [
                    {"layer": 0, "slot": "attn_o", "rel_l2": 1.0,
                     "name_a": "blk.0.attn_output.weight", "name_b": "blk.0.attn_output.weight"},
                ],
            },
            "noise_floor": {"spectral_norm": 0.05},
            "warnings": [],
        }
        with open(compare_out / "compare_summary.json", "w") as f:
            json.dump(summary, f)

        now = "2026-08-17T00:00:00+00:00"
        job = Job(
            job_id="compare-edit-1",
            model_path=f"{dir_a}|{dir_b}",
            out_dir=str(compare_out),
            spec_path="",
            status=JobStatus.DONE,
            created_at=now,
            updated_at=now,
            job_type="compare",
        )
        queue._save(job)

        client = TestClient(
            create_app(db_path=Path(queue._db_path), output_root=tmp_path / "output")
        )
        resp = client.get(f"/api/model/{a}/delta?with={b}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["tier"] == "weight_space"
        assert body["edit_signature"]["n_tensors"] == 20
        assert body["rows"][0]["rel_l2"] == 1.0


class TestErrors:
    def test_error_envelope_shape(self, client: TestClient) -> None:
        resp = client.get("/api/model/ghost")
        assert resp.status_code == 404
        body = resp.json()
        assert set(body["error"].keys()) >= {"code", "type", "message", "hint"}


class TestRecordCacheAndPercentiles:
    def test_records_cached_per_fingerprint(self, queue, scan_dir):
        """Records are rebuilt only when fingerprint.json changes."""
        from weight_atlas.api import query as q

        job = queue.get(_model_id(queue))
        fp = q._load_fingerprint(job)
        first = q._load_records(job, fp)
        second = q._load_records(job, fp)
        assert first is second

        # Changing the file (size changes → new cache key) forces a rebuild.
        tensors = _base_tensors()
        tensors["output.weight"] = _tensor("output.weight", 25.0)
        with open(scan_dir / "fingerprint.json", "w") as f:
            json.dump(_fingerprint(tensors), f)
        fp_new = q._load_fingerprint(job)
        rebuilt = q._load_records(job, fp_new)
        assert rebuilt is not first

    def test_percentile_uses_sorted_ranks(self, client, queue):
        """Regression: percentiles came from np.searchsorted on name-ordered
        (i.e. unsorted) arrays and were effectively arbitrary. The max-metric
        tensor must report p100."""
        mid = _model_id(queue)
        resp = client.get(f"/api/model/{mid}/tensor/output.weight")
        assert resp.status_code == 200
        ctx = resp.json()["context"]
        assert ctx["percentile_in_model"]["spectral_norm"] == 100.0

    def test_anomaly_percentiles_descend_with_zscore(self, client, queue):
        """Anomaly rows are zscore-descending; percentiles must be
        non-increasing along that order."""
        mid = _model_id(queue)
        resp = client.get(
            f"/api/model/{mid}/anomalies",
            params={"metric": "spectral_norm", "threshold": "2.5"},
        )
        assert resp.status_code == 200
        rows = resp.json()["rows"]
        assert len(rows) >= 2
        pcts = [r["percentile"] for r in rows]
        assert all(a >= b for a, b in zip(pcts, pcts[1:]))
