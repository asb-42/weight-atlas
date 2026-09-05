"""Job-store backend parity: SQLite (always) + MariaDB (live server only).

The battery runs every store through the same behavioral contract:
schema init idempotence, save/load round-trip of every Job field,
upsert overwrite, newest-first listing with limit, restart-recovery
rows, stale-sweeper rows, reset flow, and missing-job None.

MariaDB tests need a live server:
``WEIGHT_ATLAS_TEST_MYSQL_URL=mysql://user:pass@host:port/db`` —
skipped otherwise (including CI without the service). The MariaDB
*dialect* (REPLACE INTO, %s placeholders, SHOW COLUMNS) is additionally
pinned by ``TestMariaDBDialect`` against a recording fake connection,
so CI covers the SQL strings with no server at all.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from weight_atlas.api.jobs import Job, JobStatus
from weight_atlas.api.store import (
    COLUMN_NAMES,
    MariaDBJobStore,
    SQLiteJobStore,
    transfer_jobs,
)


def _job(job_id: str = "job-1", **over: Any) -> Job:
    kw: dict[str, Any] = {
        "job_id": job_id,
        "model_path": "/models/m.gguf",
        "out_dir": "/tmp/out",
        "spec_path": "",
        "status": JobStatus.QUEUED,
        "progress": 0.0,
        "message": "",
        "created_at": "2026-09-05T10:00:00+00:00",
        "updated_at": "2026-09-05T10:00:00+00:00",
        "error": "",
        "artefacts": [],
        "job_type": "scan",
        "renderer": "",
        "compare_mode": "strict",
        "compare_interp": "linear",
        "sheet_knobs": {},
        "quant_probe": False,
    }
    kw.update(over)
    return Job(**kw)


def _row(job: Job) -> dict[str, Any]:
    import json

    return {
        "job_id": job.job_id,
        "model_path": job.model_path,
        "out_dir": job.out_dir,
        "spec_path": job.spec_path,
        "status": job.status.value,
        "progress": job.progress,
        "message": job.message,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "error": job.error,
        "artefacts": json.dumps(job.artefacts),
        "job_type": job.job_type,
        "renderer": job.renderer,
        "compare_mode": job.compare_mode,
        "compare_interp": job.compare_interp,
        "sheet_knobs": json.dumps(job.sheet_knobs),
        "quant_probe": 1 if job.quant_probe else 0,
    }


def run_store_battery(store: Any) -> None:
    """The behavioral contract every backend must satisfy."""
    store.init_schema()
    store.init_schema()  # idempotent re-init

    # missing job
    assert store.load_row("nope") is None

    # full-field round-trip
    job = _job(
        progress=0.42,
        message="Working",
        error="",
        artefacts=["a.tif", "b.json"],
        status=JobStatus.RUNNING,
        job_type="compare",
        compare_mode="aligned",
        compare_interp="nearest",
        sheet_knobs={"normalized_depth": True},
        quant_probe=True,
    )
    store.save_row(_row(job))
    got = store.load_row("job-1")
    assert got is not None
    for key in COLUMN_NAMES:
        assert got[key] == _row(job)[key], key

    # upsert overwrites
    job2 = _job(progress=1.0, status=JobStatus.DONE, message="Complete")
    store.save_row(_row(job2))
    got2 = store.load_row("job-1")
    assert got2 is not None
    assert got2["progress"] == 1.0
    assert got2["status"] == "done"

    # listing: newest first, limit honored
    store.save_row(_row(_job("job-2", created_at="2026-09-05T11:00:00+00:00")))
    store.save_row(_row(_job("job-3", created_at="2026-09-05T09:00:00+00:00")))
    listed = store.list_rows(10)
    assert [r["job_id"] for r in listed] == ["job-2", "job-1", "job-3"]
    assert [r["job_id"] for r in store.list_rows(2)] == ["job-2", "job-1"]

    # restart recovery surface: only queued/running rows are visible;
    # the caller (JobQueue.start) filters and resets.
    rec = dict(store.recoverable_rows())
    assert "job-1" not in rec  # done → not recoverable
    store.save_row(_row(_job("job-4", status=JobStatus.QUEUED)))
    assert dict(store.recoverable_rows())["job-4"] == "queued"

    # stale sweeper surface + reset
    assert store.running_rows() == [] or all(
        isinstance(r, tuple) for r in store.running_rows()
    )
    store.save_row(_row(_job("job-5", status=JobStatus.RUNNING,
                             updated_at="2026-09-05T08:00:00+00:00")))
    running = dict(store.running_rows())
    assert running["job-5"] == "2026-09-05T08:00:00+00:00"
    store.reset_row("job-5", "queued", "re-queued after stale running",
                    "2026-09-05T12:00:00+00:00")
    back = store.load_row("job-5")
    assert back is not None
    assert back["status"] == "queued"
    assert back["message"] == "re-queued after stale running"


class TestSQLiteBackend:
    def test_battery(self, tmp_path: Path) -> None:
        run_store_battery(SQLiteJobStore(tmp_path / "jobs.db"))

    def test_legacy_migration(self, tmp_path: Path) -> None:
        """Pre-job_type SQLite DBs upgrade with type backfill intact."""
        import sqlite3

        db = tmp_path / "legacy.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE jobs (job_id TEXT PRIMARY KEY, model_path TEXT NOT NULL,"
            " out_dir TEXT NOT NULL, spec_path TEXT NOT NULL, status TEXT NOT NULL,"
            " progress REAL NOT NULL DEFAULT 0.0, message TEXT NOT NULL DEFAULT '',"
            " created_at TEXT NOT NULL, updated_at TEXT NOT NULL,"
            " error TEXT NOT NULL DEFAULT '', artefacts TEXT NOT NULL DEFAULT '[]')"
        )
        conn.execute(
            "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("r1", "/m", "/o", "", "done", 1.0, "render:sheet",
             "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00", "", "[]"),
        )
        conn.commit()
        conn.close()

        store = SQLiteJobStore(db)
        store.init_schema()
        row = store.load_row("r1")
        assert row is not None
        assert row["job_type"] == "render"
        assert row["renderer"] == "sheet"


def _mysql_url() -> str | None:
    return os.environ.get("WEIGHT_ATLAS_TEST_MYSQL_URL")


class TestMariaDBBackend:
    @pytest.fixture
    def store(self, tmp_path: Path) -> Any:
        url = _mysql_url()
        if url is None:
            pytest.skip("no live MariaDB (WEIGHT_ATLAS_TEST_MYSQL_URL unset)")
        store = MariaDBJobStore.from_url(url)
        store.init_schema()
        # isolated table per test run via unique job_id prefix (table is
        # shared with any other user of the test database)
        return store

    def test_battery(self, store: Any) -> None:
        import uuid

        prefix = uuid.uuid4().hex[:8] + "-"
        inner = store

        class _Prefixed:
            def init_schema(self) -> None:
                inner.init_schema()

            def save_row(self, row: dict[str, Any]) -> None:
                row = dict(row)
                row["job_id"] = prefix + row["job_id"]
                inner.save_row(row)

            def load_row(self, job_id: str) -> dict[str, Any] | None:
                row = inner.load_row(prefix + job_id)
                if row is not None:
                    row = dict(row)
                    row["job_id"] = job_id
                return row

            def list_rows(self, limit: int) -> list[dict[str, Any]]:
                rows = [r for r in inner.list_rows(100000)
                        if r["job_id"].startswith(prefix)]
                for r in rows:
                    r["job_id"] = r["job_id"][len(prefix):]
                return rows[:limit]

            def recoverable_rows(self) -> list[tuple[str, str]]:
                return [(j[len(prefix):], s) for j, s in inner.recoverable_rows()
                        if j.startswith(prefix)]

            def running_rows(self) -> list[tuple[str, str]]:
                return [(j[len(prefix):], s) for j, s in inner.running_rows()
                        if j.startswith(prefix)]

            def reset_row(self, job_id: str, *a: Any) -> None:
                inner.reset_row(prefix + job_id, *a)

        run_store_battery(_Prefixed())

    def test_transfer_sqlite_to_mariadb(self, tmp_path: Path, store: Any) -> None:
        src = SQLiteJobStore(tmp_path / "src.db")
        src.init_schema()
        src.save_row(_row(_job("t1", message="hello")))
        src.save_row(_row(_job("t2", status=JobStatus.DONE)))
        moved = transfer_jobs(src, store)
        assert moved == 2
        assert store.load_row("t1") is not None
        assert store.load_row("t1")["message"] == "hello"


class FakeCursor:
    def __init__(self, recorder: list[tuple[str, tuple]]) -> None:
        self._recorder = recorder
        self._rows: list[tuple] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *a: Any) -> None:
        return None

    def execute(self, sql: str, params: tuple = ()) -> None:
        self._recorder.append((sql, params))

    def fetchall(self) -> list[tuple]:
        return self._rows


class FakeConnection:
    """Recording stand-in: pins the MariaDB dialect with no server."""

    def __init__(self, recorder: list[tuple[str, tuple]]) -> None:
        self.statements = recorder
        self.committed = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.statements)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass


class TestMariaDBDialect:
    """SQL-string contract without a live server."""

    def _store(self, rec: list[tuple[str, tuple]]) -> MariaDBJobStore:
        store = MariaDBJobStore.__new__(MariaDBJobStore)
        store._params = {}
        store._fake = FakeConnection(rec)  # type: ignore[attr-defined]
        orig = MariaDBJobStore._connect
        store._connect = lambda: store._fake  # type: ignore[method-assign]
        assert orig is not None
        return store

    def test_upsert_uses_replace_and_percent_placeholders(self) -> None:
        rec: list[tuple[str, tuple]] = []
        store = self._store(rec)
        store.save_row(_row(_job()))
        sql, params = rec[0]
        assert sql.startswith("REPLACE INTO jobs")
        assert "?" not in sql
        assert sql.count("%s") == len(COLUMN_NAMES)
        assert len(params) == len(COLUMN_NAMES)

    def test_reads_use_percent_placeholders(self) -> None:
        rec = []
        store = self._store(rec)
        store.load_row("x")
        store.list_rows(7)
        store.recoverable_rows()
        store.running_rows()
        store.reset_row("x", "queued", "m", "t")
        joined = "\n".join(s for s, _ in rec)
        assert "?" not in joined
        assert "SHOW COLUMNS FROM jobs" not in joined  # introspection is init-only
        assert any("LIMIT %s" in s for s, _ in rec)

    def test_init_uses_show_columns(self) -> None:
        rec = []
        store = self._store(rec)
        store.init_schema()
        joined = "\n".join(s for s, _ in rec)
        assert "CREATE TABLE IF NOT EXISTS jobs" in joined
        assert "SHOW COLUMNS FROM jobs" in joined
        assert "PRAGMA" not in joined

    def test_transactions_commit(self) -> None:
        rec = []
        store = self._store(rec)
        store.save_row(_row(_job()))
        assert store._fake.committed is True

    def test_parse_db_url(self) -> None:
        from weight_atlas.api.store import parse_db_url as _parse

        p = _parse("mysql://u:p%40ss@db.example.com:3307/atlas")
        assert p == {"host": "db.example.com", "port": 3307, "user": "u",
                     "password": "p@ss", "database": "atlas"}
        p2 = _parse("mariadb://u@h/db")
        assert p2["port"] == 3306 and p2["password"] == ""
        with pytest.raises(ValueError):
            _parse("sqlite:///x.db")
        with pytest.raises(ValueError):
            _parse("mysql://host-only")
