"""End-to-end smoke test: `weight-atlas serve` starts a reachable HTTP server."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_until_up(url: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=2.0)
            if resp.status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise TimeoutError(f"server did not come up at {url}")


@pytest.fixture
def server(tmp_path: Path):
    """Start `weight-atlas serve` on a free loopback port, stop it afterwards.

    The subprocess must be fully sandboxed: its startup recovery resets
    running/queued rows in whatever database it opens, so pointing it at the
    real ``data/jobs.db`` would clobber live jobs (this actually happened —
    see docs/reports/2026-08-31_atlas-analysis-review-A0.md). The env
    overrides redirect the database and output root into tmp_path.
    """
    port = _free_port()
    env = dict(os.environ)
    env["WEIGHT_ATLAS_DB_PATH"] = str(tmp_path / "data" / "jobs.db")
    env["WEIGHT_ATLAS_OUTPUT_ROOT"] = str(tmp_path / "output")
    proc = subprocess.Popen(
        [sys.executable, "-m", "weight_atlas.cli", "serve", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        _wait_until_up(base)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)


def test_serve_responds_on_loopback(server):
    resp = httpx.get(f"{server}/", timeout=5.0)
    assert resp.status_code == 200
    assert "weight-atlas" in resp.text.lower() or "model" in resp.text.lower()

    compare = httpx.get(f"{server}/compare", timeout=5.0)
    assert compare.status_code == 200


def test_serve_is_sandboxed_from_real_state(server, tmp_path: Path):
    """The smoke server must run against its tmp sandbox, never the repo's
    real data/jobs.db + output/ (startup recovery would reset live jobs)."""
    import json
    import sqlite3

    resp = httpx.get(f"{server}/api/models", timeout=5.0)
    assert resp.status_code == 200
    # The sandbox DB was created and the model list is empty (fresh state)
    sandbox_db = tmp_path / "data" / "jobs.db"
    assert sandbox_db.exists(), "server did not use WEIGHT_ATLAS_DB_PATH sandbox"
    con = sqlite3.connect(f"file:{sandbox_db}?mode=ro", uri=True)
    n_jobs = con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    con.close()
    assert n_jobs == 0, "sandbox server sees foreign jobs"
    assert json.loads(resp.text)["n_scans"] == 0
