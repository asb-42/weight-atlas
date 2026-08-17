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

    def test_compare_rejects_non_scan_directory(
        self, client: TestClient, tmp_path: Path, fake_model: Path
    ) -> None:
        """A comparison must reject a compare/render output dir (no
        manifest.json) up front instead of failing inside the worker."""
        scan_dir = tmp_path / "scan_valid"
        scan_dir.mkdir(exist_ok=True)
        (scan_dir / "manifest.json").write_text("{}")

        # A "model" dir that is actually a compare output — no manifest.
        compare_dir = tmp_path / "compare_a_vs_b_abcd1234"
        compare_dir.mkdir(exist_ok=True)
        (compare_dir / "compare_summary.json").write_text("{}")

        # dir_b is the real scan dir, dir_a is the compare output → rejected.
        resp = client.post(
            "/api/compare",
            json={"dir_a": str(compare_dir), "dir_b": str(scan_dir)},
        )
        assert resp.status_code == 400
        assert "not a scan output directory" in resp.json()["detail"]

    def test_compare_accepts_scan_directories(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Two valid scan dirs (with manifest.json) pass validation."""
        dir_a = tmp_path / "scan_a"
        dir_b = tmp_path / "scan_b"
        for d in (dir_a, dir_b):
            d.mkdir(exist_ok=True)
            (d / "manifest.json").write_text("{}")
        resp = client.post(
            "/api/compare",
            json={"dir_a": str(dir_a), "dir_b": str(dir_b)},
        )
        assert resp.status_code == 200

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
    def _import_scan(self, client: TestClient, tmp_path: Path, fake_model: Path, n_tensors: int = 5) -> str:
        """Create + import a done scan so fingerprint stats are available."""
        scan_dir = tmp_path / f"scan_{n_tensors}"
        scan_dir.mkdir(exist_ok=True)
        tensors = {}
        for i in range(n_tensors):
            tensors[f"blk.{i}.attn_q.weight"] = {
                "shape": [32, 32], "frobenius": 1.0, "spectral_norm": 1.0,
                "effective_rank": 2.0, "kurtosis": 0.0, "sparsity": 0.0,
            }
        fp = {
            "spec_version": 2, "model": {"n_tensors": n_tensors, "n_layers": 1},
            "mapping_coverage": {"in_slots": 1.0, "unmapped": 0, "unmapped_tensors": []},
            "tensors": tensors,
        }
        with open(scan_dir / "fingerprint.json", "w") as f:
            json.dump(fp, f)
        resp = client.post(
            "/api/import",
            json={"scan_dir": str(scan_dir), "model_path": str(fake_model)},
        )
        assert resp.status_code == 200
        return resp.json()["job_id"]

    def test_detail_requires_done_job(self, client: TestClient, fake_model: Path) -> None:
        create_resp = client.post("/api/jobs", json={"model_path": str(fake_model)})
        job_id = create_resp.json()["job_id"]

        # Job is queued/running — detail should still render (but may be empty)
        response = client.get(f"/models/{job_id}")
        assert response.status_code == 200

    def test_detail_tabs_render(self, client: TestClient, fake_model: Path, tmp_path: Path) -> None:
        """Each model sub-page (tab) must render 200 with its own content."""
        job_id = self._import_scan(client, tmp_path, fake_model)

        base = client.get(f"/models/{job_id}")
        assert base.status_code == 200
        # Tab bar is present; stats table is NOT inline on the overview page.
        assert 'class="model-tabs"' in base.text
        assert 'data-tab="stats"' in base.text
        assert "stats-table" not in base.text

        for tab, needle in (
            ("overview", "Operations"),
            ("sheets", "Sheet"),
            ("terrain", "Terrain"),
            ("stats", "Statistics"),
            ("spec", "Spec"),
        ):
            resp = client.get(f"/models/{job_id}?tab={tab}")
            assert resp.status_code == 200, f"tab {tab} failed"
            assert "model-tab--active" in resp.text
            assert needle in resp.text, f"tab {tab} missing {needle}"

    def test_stats_tab_paginates(self, client: TestClient, fake_model: Path, tmp_path: Path) -> None:
        """Statistics sub-page must paginate server-side."""
        job_id = self._import_scan(client, tmp_path, fake_model, n_tensors=5)
        resp = client.get(f"/models/{job_id}?tab=stats")
        assert resp.status_code == 200
        assert "stats-table" in resp.text

    def test_stats_tab_clamps_page(self, client: TestClient, fake_model: Path, tmp_path: Path) -> None:
        """Out-of-range stats page must clamp, not error."""
        job_id = self._import_scan(client, tmp_path, fake_model, n_tensors=450)
        resp = client.get(f"/models/{job_id}?tab=stats&page=999999")
        assert resp.status_code == 200
        assert "Page 3 / 3" in resp.text


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

    def test_restart_preserves_render_job_type(self, tmp_path: Path, spec_path: Path, fake_model: Path) -> None:
        """A render job must come back as a render job after a restart.

        Regression: job type used to be encoded in ``message`` and recovery
        overwrote it with ``re-queued after restart``, so a render job was
        re-executed as a full scan on restart.
        """
        import time

        db_path = tmp_path / "restart_render.db"
        q1 = JobQueue(db_path, on_job=lambda j: None)
        base = q1.submit(fake_model, tmp_path / "out", spec_path)
        job = q1.submit_render(base.job_id, "preview")
        assert job.job_type == "render"
        assert job.renderer == "preview"

        ran: list[str] = []
        q2 = JobQueue(db_path, on_job=lambda j: None)
        q2._execute = lambda j: ran.append(j.job_id)  # noqa: SLF001 — test hook
        q2.start()
        try:
            time.sleep(0.4)
        finally:
            q2.stop()

        reloaded = q2.get(job.job_id)
        assert reloaded is not None
        assert reloaded.job_type == "render", (
            "render job type was lost across restart"
        )
        assert reloaded.renderer == "preview"
        assert job.job_id in ran, "render job was not re-enqueued after restart"

    def test_render_job_sheet_knobs_persist_and_apply(self, tmp_path: Path, spec_path: Path, fake_model: Path) -> None:
        """Per-render sheet knobs survive a restart and are overlaid on the spec."""
        import time

        db_path = tmp_path / "restart_knobs.db"
        q1 = JobQueue(db_path, on_job=lambda j: None)
        base = q1.submit(fake_model, tmp_path / "out", spec_path)
        job = q1.submit_render(base.job_id, "sheet", sheet_knobs={"normalized_depth": True})
        assert job.sheet_knobs == {"normalized_depth": True}

        q2 = JobQueue(db_path, on_job=lambda j: None)
        q2._execute = lambda j: None  # noqa: SLF001 — don't actually render
        q2.start()
        try:
            time.sleep(0.4)
        finally:
            q2.stop()

        reloaded = q2.get(job.job_id)
        assert reloaded is not None
        assert reloaded.sheet_knobs == {"normalized_depth": True}, (
            "sheet_knobs were lost across restart"
        )

        from weight_atlas.core.types import AtlasSpec
        base_spec = AtlasSpec.from_json(spec_path)
        assert base_spec.sheet.get("normalized_depth") is None
        overlaid = q2._apply_sheet_knobs(base_spec, {"normalized_depth": True})  # noqa: SLF001
        assert overlaid.sheet["normalized_depth"] is True
        assert base_spec.sheet.get("normalized_depth") is None, (
            "overlay must not mutate the recorded spec"
        )

    def test_render_job_sheet_knobs_roundtrip_through_db(self, tmp_path: Path, spec_path: Path, fake_model: Path) -> None:
        """Sheet knobs round-trip through the SQLite persistence layer."""
        db_path = tmp_path / "knobs_roundtrip.db"
        q1 = JobQueue(db_path, on_job=lambda j: None)
        base = q1.submit(fake_model, tmp_path / "out", spec_path)
        job = q1.submit_render(base.job_id, "sheet", sheet_knobs={"drop_empty_cols": True, "normalized_depth": True})
        q1._save(job)  # noqa: SLF001
        loaded = q1._load(job.job_id)  # noqa: SLF001
        assert loaded is not None
        assert loaded.sheet_knobs == {"drop_empty_cols": True, "normalized_depth": True}

        # A render without knobs defaults to an empty dict.
        bare = q1.submit_render(base.job_id, "preview")
        q1._save(bare)  # noqa: SLF001
        assert q1._load(bare.job_id).sheet_knobs == {}  # noqa: SLF001

    def test_restart_preserves_compare_job_type(self, tmp_path: Path, spec_path: Path, fake_model: Path) -> None:
        """A compare job must keep its mode/interp after a restart."""
        import time

        db_path = tmp_path / "restart_compare.db"
        q1 = JobQueue(db_path, on_job=lambda j: None)
        job = q1.submit_compare(
            tmp_path / "a", tmp_path / "b", tmp_path / "cmp_out", spec_path,
            mode="aligned", interp="nearest",
        )
        assert job.job_type == "compare"

        ran: list[str] = []
        q2 = JobQueue(db_path, on_job=lambda j: None)
        q2._execute = lambda j: ran.append(j.job_id)  # noqa: SLF001 — test hook
        q2.start()
        try:
            time.sleep(0.4)
        finally:
            q2.stop()

        reloaded = q2.get(job.job_id)
        assert reloaded is not None
        assert reloaded.job_type == "compare"
        assert reloaded.compare_mode == "aligned"
        assert reloaded.compare_interp == "nearest"
        assert job.job_id in ran, "compare job was not re-enqueued after restart"

    def test_stale_running_job_recovered_by_sweep(self, tmp_path: Path, spec_path: Path, fake_model: Path) -> None:
        """A ``running`` row stale since before startup must be re-queued.

        Regression: a job marked ``running`` after start()'s recovery ran (a
        worker that then died) stayed ``running`` forever — the UI showed two
        jobs "running" at once even though the single-threaded worker could
        only execute one. The periodic sweep must reset it to ``queued``.
        """
        from datetime import UTC, datetime, timedelta

        db_path = tmp_path / "stale_running.db"
        q1 = JobQueue(db_path, on_job=lambda j: None)
        job = q1.submit(fake_model, tmp_path / "out", spec_path)
        # Simulate a process that picked the job up, marked it running, and
        # then died *after* start()'s startup recovery had already run.
        old = (datetime.now(UTC) - timedelta(seconds=3600)).isoformat(timespec="seconds")
        q1._save(job)  # noqa: SLF001
        with q1._connection() as conn:  # noqa: SLF001
            conn.execute(
                "UPDATE jobs SET status=?, updated_at=?, message=? WHERE job_id=?",
                (JobStatus.RUNNING.value, old, "Rendering height...", job.job_id),
            )

        q1._recover_stale_running()  # noqa: SLF001 — the sweep body
        recovered = q1.get(job.job_id)
        assert recovered is not None
        assert recovered.status == JobStatus.QUEUED
        assert recovered.message == "re-queued after stale running"

    def test_stale_sweep_skips_recent_and_current_running(self, tmp_path: Path, spec_path: Path, fake_model: Path) -> None:
        """The sweep must not reset freshly-updated or in-flight running rows."""
        from datetime import UTC, datetime, timedelta

        db_path = tmp_path / "stale_running2.db"
        q = JobQueue(db_path, on_job=lambda j: None)
        fresh = q.submit(fake_model, tmp_path / "out", spec_path)
        stale = q.submit(fake_model, tmp_path / "out", spec_path)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        old = (datetime.now(UTC) - timedelta(seconds=3600)).isoformat(timespec="seconds")
        with q._connection() as conn:  # noqa: SLF001
            conn.execute(
                "UPDATE jobs SET status=?, updated_at=? WHERE job_id=?",
                (JobStatus.RUNNING.value, now, fresh.job_id),
            )
            conn.execute(
                "UPDATE jobs SET status=?, updated_at=? WHERE job_id=?",
                (JobStatus.RUNNING.value, old, stale.job_id),
            )
        # The worker is "executing" the fresh job right now.
        q._current_job_id = fresh.job_id  # noqa: SLF001
        q._recover_stale_running()  # noqa: SLF001 — the sweep body

        assert q.get(fresh.job_id).status == JobStatus.RUNNING, (
            "freshly-updated running row must not be reset"
        )
        assert q.get(stale.job_id).status == JobStatus.QUEUED, (
            "stale running row must be reset even while another job executes"
        )

    def test_legacy_db_backfills_job_type_from_message(self, tmp_path: Path, spec_path: Path) -> None:
        """Pre-job_type DBs must be migrated, backfilling type from message.

        Legacy rows encoded the type in ``message`` (``render:<id>``,
        ``compare[:mode[:interp]]``); after migration they must carry the
        dedicated job_type column so recovery does not turn them into scans.
        """
        import sqlite3

        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                model_path TEXT NOT NULL,
                out_dir TEXT NOT NULL,
                spec_path TEXT NOT NULL,
                status TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0.0,
                message TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error TEXT NOT NULL DEFAULT '',
                artefacts TEXT NOT NULL DEFAULT '[]'
            )
        """)
        now = "2026-01-01T00:00:00"
        conn.executemany(
            "INSERT INTO jobs (job_id, model_path, out_dir, spec_path, status,"
            " progress, message, created_at, updated_at, error, artefacts)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("r1", "/m", "/o", "", "queued", 0.0, "render:preview", now, now, "", "[]"),
                ("c1", "/a|/b", "/o", "", "queued", 0.0, "compare:aligned:nearest", now, now, "", "[]"),
                ("c2", "/a|/b", "/o", "", "queued", 0.0, "compare", now, now, "", "[]"),
                ("s1", "/m", "/o", "", "queued", 0.0, "Queued", now, now, "", "[]"),
                ("c3", "/a|/b", "/out/compare_a_vs_b_abcd1234", "", "done", 1.0, "Complete", now, now, "", "[]"),
            ],
        )
        conn.commit()
        conn.close()

        queue = JobQueue(db_path, on_job=lambda j: None)
        assert queue.get("r1").job_type == "render"
        assert queue.get("r1").renderer == "preview"
        assert queue.get("c1").job_type == "compare"
        assert queue.get("c1").compare_mode == "aligned"
        assert queue.get("c1").compare_interp == "nearest"
        assert queue.get("c2").job_type == "compare"
        assert queue.get("s1").job_type == "scan"
        assert queue.get("c3").job_type == "compare"

    def test_db_access_does_not_leak_fds(
        self, tmp_path: Path, spec_path: Path, fake_model: Path
    ) -> None:
        """Repeated DB reads/writes must close their connections.

        Regression: ``with sqlite3.Connection`` commits but never closes, so a
        long scan (worker progress writes + UI polling every 2 s) leaked two
        file descriptors per call until the process hit its fd limit and failed
        with ``unable to open database file``.
        """
        import os

        db_path = tmp_path / "leak.db"
        queue = JobQueue(db_path, on_job=lambda j: None)
        job = queue.submit(fake_model, tmp_path / "out", spec_path)

        def fd_count() -> int:
            return len(os.listdir(f"/proc/{os.getpid()}/fd"))

        before = fd_count()
        for _ in range(300):
            queue._save(job)  # noqa: SLF001
            queue._load(job.job_id)
            queue.list_jobs()
        growth = fd_count() - before
        assert growth < 30, f"file descriptors leaked: {growth}"


class TestComparePageCandidates:
    def test_compare_page_lists_only_scan_jobs(
        self, client: TestClient, fake_model: Path, tmp_path: Path
    ) -> None:
        """Render jobs pointing at compare output dirs must not appear as
        compare candidates (they lack manifest.json / field_*.tif)."""
        import uuid

        from weight_atlas.api.jobs import Job, JobQueue

        db_path = tmp_path / "jobs.db"
        queue = JobQueue(db_path, on_job=lambda j: None)

        now = "2026-01-01T00:00:00"
        scan_job = Job(
            job_id=str(uuid.uuid4()),
            model_path=str(fake_model),
            out_dir=str(tmp_path / "scan_out"),
            spec_path="",
            status=JobStatus.DONE,
            created_at=now,
            updated_at=now,
            message="Complete",
            job_type="scan",
        )
        compare_out = tmp_path / "compare_a_vs_b_abcd1234"
        compare_out.mkdir(exist_ok=True)
        (compare_out / "compare_summary.json").write_text("{}")
        render_job = Job(
            job_id=str(uuid.uuid4()),
            model_path=str(fake_model),
            out_dir=str(compare_out),
            spec_path="",
            status=JobStatus.DONE,
            created_at=now,
            updated_at=now,
            message="Complete",
            job_type="render",
            renderer="sheet",
        )
        queue._save(scan_job)  # noqa: SLF001
        queue._save(render_job)  # noqa: SLF001

        app = create_app(db_path=db_path, spec_path=None, output_root=tmp_path / "output")
        client = TestClient(app)
        resp = client.get("/compare")
        assert resp.status_code == 200
        html = resp.text
        assert str(scan_job.out_dir) in html
        assert str(compare_out) not in html


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


class TestFileBrowser:
    """The /api/browse endpoint lists directories and model files (HTMX fragment)."""

    def test_browse_lists_model_files(self, client: TestClient, tmp_path: Path) -> None:
        root = tmp_path / "browse_root"
        root.mkdir()
        (root / "my_model.safetensors").write_bytes(b"")
        (root / "subdir").mkdir()
        (root / "notes.txt").write_bytes(b"not a model")
        (root / ".hidden").write_bytes(b"hidden")

        resp = client.get("/api/browse", params={"path": str(root), "mode": "model"})
        assert resp.status_code == 200
        assert "my_model.safetensors" in resp.text
        assert "subdir" in resp.text
        assert "notes.txt" not in resp.text
        assert ".hidden" not in resp.text

    def test_browse_dir_mode_selectable(self, client: TestClient, tmp_path: Path) -> None:
        root = tmp_path / "browse_root"
        root.mkdir()
        (root / "artefacts").mkdir()
        (root / "artefacts" / "fingerprint.json").write_text("{}")

        resp = client.get("/api/browse", params={"path": str(root), "mode": "dir"})
        assert resp.status_code == 200
        assert "artefacts" in resp.text
        assert "select dir" in resp.text

    def test_browse_marks_model_dirs(self, client: TestClient, tmp_path: Path) -> None:
        root = tmp_path / "browse_root"
        root.mkdir()
        model_dir = root / "hf_model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(b"")
        (root / "plain_dir").mkdir()

        resp = client.get("/api/browse", params={"path": str(root), "mode": "model"})
        assert resp.status_code == 200
        # Model dir gets the "model" badge + select button
        assert "hf_model" in resp.text
        assert "badge badge--info" in resp.text
        # Plain dir has no select button in model mode
        assert "plain_dir" in resp.text

    def test_browse_rejects_path_outside_roots(
        self, tmp_path: Path, spec_path: Path, fake_model: Path
    ) -> None:
        output_root = tmp_path / "output"
        output_root.mkdir(exist_ok=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "m.safetensors").write_bytes(b"")

        app = create_app(
            db_path=tmp_path / "jobs.db",
            spec_path=spec_path,
            output_root=output_root,
            model_roots=[output_root],
        )
        with TestClient(app) as client:
            resp = client.get("/api/browse", params={"path": str(outside), "mode": "model"})
            assert resp.status_code == 200
            assert "outside the allowed model roots" in resp.text

    def test_browse_nonexistent_path(self, client: TestClient) -> None:
        resp = client.get("/api/browse", params={"path": "/nonexistent/path", "mode": "model"})
        assert resp.status_code == 200
        assert "not a directory" in resp.text


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


class TestRenderEndpoint:
    def test_render_sheet_with_knobs_enqueues_override(
        self, client, tmp_path: Path, fake_model: Path
    ) -> None:
        """POSTing sheet checkboxes to the render route enqueues a knobs job."""
        resp = client.post("/api/jobs", json={"model_path": str(fake_model)})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        r = client.post(
            f"/api/jobs/{job_id}/render/sheet",
            data={"normalized_depth": "on", "drop_empty_cols": "on"},
        )
        assert r.status_code == 202
        assert r.headers.get("HX-Redirect", "").startswith("/jobs/")

        render_job_id = r.headers["HX-Redirect"].split("/")[-1]
        render_job = client.get(f"/api/jobs/{render_job_id}").json()
        assert render_job["job_type"] == "render"
        assert render_job["sheet_knobs"] == {
            "normalized_depth": True,
            "drop_empty_cols": True,
        }

    def test_render_blender_ignores_sheet_knobs(
        self, client, tmp_path: Path, fake_model: Path
    ) -> None:
        """The Blender (terrain) renderer must not carry sheet knobs."""
        resp = client.post("/api/jobs", json={"model_path": str(fake_model)})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        r = client.post(
            f"/api/jobs/{job_id}/render/blender",
            data={"normalized_depth": "on"},
        )
        assert r.status_code == 202
        render_job_id = r.headers["HX-Redirect"].split("/")[-1]
        render_job = client.get(f"/api/jobs/{render_job_id}").json()
        assert render_job["job_type"] == "render"
        assert render_job["sheet_knobs"] == {}

    def test_render_fractal_with_mode_enqueues_override(
        self, client, tmp_path: Path, fake_model: Path
    ) -> None:
        """POSTing a fractal_mode to the fractal render route enqueues the override."""
        resp = client.post("/api/jobs", json={"model_path": str(fake_model)})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        r = client.post(
            f"/api/jobs/{job_id}/render/fractal",
            data={"fractal_mode": "sdf"},
        )
        assert r.status_code == 202
        render_job_id = r.headers["HX-Redirect"].split("/")[-1]
        render_job = client.get(f"/api/jobs/{render_job_id}").json()
        assert render_job["sheet_knobs"] == {"fractal_mode": "sdf"}

    def test_render_fractal_rejects_unknown_mode(
        self, client, tmp_path: Path, fake_model: Path
    ) -> None:
        """An invalid fractal_mode must be dropped, not enqueued."""
        resp = client.post("/api/jobs", json={"model_path": str(fake_model)})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        r = client.post(
            f"/api/jobs/{job_id}/render/fractal",
            data={"fractal_mode": "bogus"},
        )
        assert r.status_code == 202
        render_job_id = r.headers["HX-Redirect"].split("/")[-1]
        render_job = client.get(f"/api/jobs/{render_job_id}").json()
        assert render_job["sheet_knobs"] == {}

    def test_render_unknown_renderer_404(self, client, tmp_path: Path, fake_model: Path) -> None:
        """An unknown renderer id must 404 before enqueueing."""
        resp = client.post("/api/jobs", json={"model_path": str(fake_model)})
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        r = client.post(f"/api/jobs/{job_id}/render/nonexistent")
        assert r.status_code == 404


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
