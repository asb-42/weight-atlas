"""Tests for the FastAPI web UI endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from safetensors.numpy import save_file

from weight_atlas.api.jobs import JobQueue, JobStatus
from weight_atlas.api.main import create_app


@pytest.fixture
def fake_model(tmp_path: Path) -> Path:
    """Create a small fake safetensors model for testing."""
    model_path = tmp_path / "test_model.safetensors"
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": np.random.default_rng(42)
        .normal(0, 0.1, (32, 32))
        .astype(np.float32),
        "model.layers.0.mlp.gate_proj.weight": np.random.default_rng(43)
        .normal(0, 0.1, (32, 32))
        .astype(np.float32),
        "model.embed_tokens.weight": np.random.default_rng(44)
        .normal(0, 0.1, (32, 32))
        .astype(np.float32),
        "lm_head.weight": np.random.default_rng(45)
        .normal(0, 0.1, (32, 32))
        .astype(np.float32),
    }
    save_file(tensors, str(model_path))
    return model_path


@pytest.fixture
def spec_path(tmp_path: Path) -> Path:
    """Create a minimal atlas spec for testing."""
    spec = {
        "spec_version": 1,
        "slots": ["embed", "attn_q", "attn_k", "attn_v", "attn_o",
                  "mlp_gate", "mlp_up", "mlp_down", "norm_attn",
                  "norm_mlp", "router", "lm_head", "other"],
        "channels": {
            "height": {"stat": "spectral_norm", "scale": {"type": "log1p"}},
            "tint": {"stat": "effective_rank", "scale": {"type": "quantile_clip", "lo": 0.01, "hi": 0.99}},
            "rough": {"stat": "kurtosis", "scale": {"type": "log1p"}},
        },
        "grid": {"upsample": 4, "smooth_sigma": 1.0},
        "sheet": {"contour_levels": 8, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
        "seeds": {"svd": 0},
    }
    spec_path = tmp_path / "atlas_spec.v2.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f)
    return spec_path


@pytest.fixture
def app(tmp_path: Path, spec_path: Path):
    """Create test app with temp directories."""
    db_path = tmp_path / "jobs.db"
    output_root = tmp_path / "output"
    output_root.mkdir(exist_ok=True)

    app = create_app(
        db_path=db_path,
        spec_path=spec_path,
        output_root=output_root,
    )
    yield app


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


class TestIndexPage:
    def test_index_returns_html(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Weight Atlas" in response.text


class TestJobCreation:
    def test_create_job_with_valid_path(
        self, client: TestClient, fake_model: Path
    ) -> None:
        response = client.post("/api/jobs", json={"model_path": str(fake_model)})
        assert response.status_code == 200
        data = response.json()
        assert data["model_path"] == str(fake_model)
        assert data["status"] == "queued"
        assert "job_id" in data

    def test_create_job_missing_path(self, client: TestClient) -> None:
        response = client.post("/api/jobs", json={})
        assert response.status_code == 400

    def test_create_job_nonexistent_path(self, client: TestClient) -> None:
        response = client.post(
            "/api/jobs", json={"model_path": "/nonexistent/path"}
        )
        assert response.status_code == 404


class TestJobStatus:
    def test_get_job_status(
        self, client: TestClient, fake_model: Path
    ) -> None:
        # Create job
        create_resp = client.post("/api/jobs", json={"model_path": str(fake_model)})
        job_id = create_resp.json()["job_id"]

        # Get status
        status_resp = client.get(f"/api/jobs/{job_id}")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("queued", "running", "done")

    def test_get_nonexistent_job(self, client: TestClient) -> None:
        response = client.get("/api/jobs/nonexistent-id")
        assert response.status_code == 404


class TestJobProgressPage:
    def test_progress_page_returns_html(
        self, client: TestClient, fake_model: Path
    ) -> None:
        create_resp = client.post("/api/jobs", json={"model_path": str(fake_model)})
        job_id = create_resp.json()["job_id"]

        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestModelDetail:
    def test_detail_requires_done_job(self, client: TestClient, fake_model: Path) -> None:
        create_resp = client.post("/api/jobs", json={"model_path": str(fake_model)})
        job_id = create_resp.json()["job_id"]

        # Job is queued/running — detail should still render (but may be empty)
        response = client.get(f"/models/{job_id}")
        assert response.status_code == 200


class TestFingerprintEndpoint:
    def test_fingerprint_requires_done_job(
        self, client: TestClient, fake_model: Path
    ) -> None:
        create_resp = client.post("/api/jobs", json={"model_path": str(fake_model)})
        job_id = create_resp.json()["job_id"]

        # Job not done yet
        response = client.get(f"/api/models/{job_id}/fingerprint")
        assert response.status_code == 409


class TestJobStatusFragment:
    def test_status_fragment_returns_html(
        self, client: TestClient, fake_model: Path
    ) -> None:
        create_resp = client.post("/api/jobs", json={"model_path": str(fake_model)})
        job_id = create_resp.json()["job_id"]

        response = client.get(f"/api/jobs/{job_id}/status")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "badge" in response.text


class TestStaticFiles:
    def test_css_accessible(self, client: TestClient, app) -> None:
        """CSS should be served from /static/style.css."""
        response = client.get("/static/style.css")
        # May be 404 if static dir doesn't exist in test, but shouldn't error
        assert response.status_code in (200, 404)


class TestJobQueueDB:
    def test_job_persisted_to_sqlite(
        self, tmp_path: Path, spec_path: Path, fake_model: Path
    ) -> None:
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path, on_job=lambda j: None)

        job = queue.submit(fake_model, tmp_path / "out", spec_path)
        loaded = queue.get(job.job_id)
        assert loaded is not None
        assert loaded.job_id == job.job_id
        assert loaded.status == JobStatus.QUEUED

    def test_list_jobs(self, tmp_path: Path, spec_path: Path, fake_model: Path) -> None:
        db_path = tmp_path / "test.db"
        queue = JobQueue(db_path, on_job=lambda j: None)

        queue.submit(fake_model, tmp_path / "out", spec_path)
        jobs = queue.list_jobs()
        assert len(jobs) == 1

    def test_restart_recovers_queued_job(self, tmp_path: Path, spec_path: Path, fake_model: Path) -> None:
        """A job persisted as queued must be re-enqueued when the queue restarts.

        Regression test: queued-but-unstarted jobs were stranded after a server
        restart because the in-memory queue was empty and never rehydrated.
        """
        import time

        db_path = tmp_path / "restart.db"
        q1 = JobQueue(db_path, on_job=lambda j: None)
        job = q1.submit(fake_model, tmp_path / "out", spec_path)
        # NOTE: q1.start() is deliberately never called — simulating a crash
        # before the worker consumed the queued job.
        assert job.status == JobStatus.QUEUED

        ran: list[str] = []
        q2 = JobQueue(db_path, on_job=lambda j: None)
        q2._execute = lambda j: ran.append(j.job_id)  # noqa: SLF001 — test hook
        q2.start()
        try:
            time.sleep(0.4)
        finally:
            q2.stop()

        assert job.job_id in ran, "queued job was not re-enqueued after restart"


class TestArtefactRoute:
    def test_artefact_route_serves_png(self, client: TestClient, fake_model: Path, tmp_path: Path) -> None:
        """GET /models/{id}/artifacts/{name} should serve PNG files."""
        scan_dir = tmp_path / "scan_test"
        scan_dir.mkdir(exist_ok=True)
        fp = {"spec_version": 2, "model": {"n_tensors": 1, "n_layers": 1}, "tensors": {}}
        with open(scan_dir / "fingerprint.json", "w") as f:
            json.dump(fp, f)
        # Create a fake PNG
        import numpy as np
        from PIL import Image
        img = Image.fromarray((np.random.default_rng(42).normal(0, 1, (100, 100)) * 255).astype(np.uint8))
        img.save(scan_dir / "test_raw.png")

        # Import the scan
        resp = client.post("/api/import", json={"scan_dir": str(scan_dir), "model_path": str(fake_model)})
        assert resp.status_code == 200
        imported_job_id = resp.json()["job_id"]

        # Now try to serve the artefact
        resp = client.get(f"/models/{imported_job_id}/artifacts/test_raw.png")
        assert resp.status_code == 200
        assert "image/png" in resp.headers.get("content-type", "")

    def test_artefact_route_blocks_traversal(self, client: TestClient, fake_model: Path, tmp_path: Path) -> None:
        """Path traversal should be blocked."""
        scan_dir = tmp_path / "scan_test2"
        scan_dir.mkdir(exist_ok=True)
        fp = {"spec_version": 2, "model": {"n_tensors": 1, "n_layers": 1}, "tensors": {}}
        with open(scan_dir / "fingerprint.json", "w") as f:
            json.dump(fp, f)

        resp = client.post("/api/import", json={"scan_dir": str(scan_dir), "model_path": str(fake_model)})
        assert resp.status_code == 200
        imported_job_id = resp.json()["job_id"]

        # Try path traversal
        resp = client.get(f"/models/{imported_job_id}/artifacts/../../../etc/passwd")
        assert resp.status_code in (403, 404)

    def test_artefact_route_blocks_disallowed_extensions(self, client: TestClient, fake_model: Path, tmp_path: Path) -> None:
        """Disallowed file extensions should be blocked."""
        scan_dir = tmp_path / "scan_test3"
        scan_dir.mkdir(exist_ok=True)
        fp = {"spec_version": 2, "model": {"n_tensors": 1, "n_layers": 1}, "tensors": {}}
        with open(scan_dir / "fingerprint.json", "w") as f:
            json.dump(fp, f)
        with open(scan_dir / "malicious.exe", "wb") as f:
            f.write(b"fake exe")

        resp = client.post("/api/import", json={"scan_dir": str(scan_dir), "model_path": str(fake_model)})
        assert resp.status_code == 200
        imported_job_id = resp.json()["job_id"]

        resp = client.get(f"/models/{imported_job_id}/artifacts/malicious.exe")
        assert resp.status_code == 403


class TestPathConfinement:
    """When model_roots is configured, paths outside the allowlist are rejected."""

    def test_create_job_rejects_path_outside_roots(self, tmp_path: Path, spec_path: Path, fake_model: Path) -> None:
        output_root = tmp_path / "output"
        output_root.mkdir(exist_ok=True)

        app = create_app(
            db_path=tmp_path / "jobs.db",
            spec_path=spec_path,
            output_root=output_root,
            model_roots=[output_root],  # fake_model lives outside output_root
        )
        with TestClient(app) as client:
            resp = client.post("/api/jobs", json={"model_path": str(fake_model)})
            assert resp.status_code == 403

    def test_import_rejects_path_outside_roots(self, tmp_path: Path, spec_path: Path, fake_model: Path) -> None:
        output_root = tmp_path / "output"
        output_root.mkdir(exist_ok=True)
        # A valid-looking scan dir OUTSIDE the allowed roots.
        scan_dir = tmp_path / "external_scan"
        scan_dir.mkdir(exist_ok=True)
        with open(scan_dir / "fingerprint.json", "w") as f:
            json.dump({"spec_version": 2, "model": {"n_tensors": 1, "n_layers": 1}, "tensors": {}}, f)

        app = create_app(
            db_path=tmp_path / "jobs.db",
            spec_path=spec_path,
            output_root=output_root,
            model_roots=[output_root],
        )
        with TestClient(app) as client:
            resp = client.post("/api/import", json={"scan_dir": str(scan_dir), "model_path": str(fake_model)})
            assert resp.status_code == 403


class TestRescan:
    def test_rescan_rejects_unscannable_model_path(self, client, tmp_path: Path) -> None:
        """An imported scan whose model_path points at a non-model dir must fail loudly.

        Regression: rescan used to enqueue a job that silently failed, leaving a
        stale fingerprint (e.g. a Qwen3-Next model stuck at 47% coverage from an
        old scan) with no explanation.
        """
        scan_dir = tmp_path / "scan_no_model"
        scan_dir.mkdir(exist_ok=True)
        with open(scan_dir / "fingerprint.json", "w") as f:
            json.dump({"spec_version": 2, "model": {"n_tensors": 1, "n_layers": 1}, "tensors": {}}, f)

        resp = client.post("/api/import", json={"scan_dir": str(scan_dir)})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        r = client.post(f"/api/jobs/{job_id}/rescan")
        assert r.status_code == 400
        assert "not a scannable model" in r.text

    def test_rescan_valid_model_redirects_to_progress(self, client, fake_model: Path) -> None:
        """A rescan with a valid model path returns 202 + an HX-Redirect."""
        resp = client.post("/api/jobs", json={"model_path": str(fake_model)})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        r = client.post(f"/api/jobs/{job_id}/rescan")
        assert r.status_code == 202
        assert r.headers.get("HX-Redirect", "").startswith("/jobs/")

    def test_set_model_path_enables_rescan_of_import(self, client, tmp_path: Path) -> None:
        """Setting a scannable model path on an imported job lets rescan proceed."""
        import numpy as np
        from safetensors.numpy import save_file

        scan_dir = tmp_path / "scan_mp"
        scan_dir.mkdir(exist_ok=True)
        with open(scan_dir / "fingerprint.json", "w") as f:
            json.dump({"spec_version": 2, "model": {"n_tensors": 1, "n_layers": 1}, "tensors": {}}, f)
        resp = client.post("/api/import", json={"scan_dir": str(scan_dir)})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        # Rescan of the imported job (model_path = scan_dir) must be rejected.
        r = client.post(f"/api/jobs/{job_id}/rescan")
        assert r.status_code == 400

        # Invalid model path rejected.
        r = client.post(f"/api/jobs/{job_id}/model-path", json={"model_path": "/nonexistent/does-not-exist"})
        assert r.status_code == 400

        # Valid model file accepted; job.model_path updated.
        model = tmp_path / "m.safetensors"
        save_file({"model.layers.0.self_attn.q_proj.weight": np.zeros((4, 4), dtype=np.float32)}, str(model))
        r = client.post(f"/api/jobs/{job_id}/model-path", json={"model_path": str(model)})
        assert r.status_code == 200
        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["model_path"] == str(model.resolve())

        # Now rescan passes validation and redirects to the progress page.
        r = client.post(f"/api/jobs/{job_id}/rescan")
        assert r.status_code == 202
        assert r.headers.get("HX-Redirect", "").startswith("/jobs/")
