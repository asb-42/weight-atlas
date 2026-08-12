"""End-to-end smoke test: `weight-atlas serve` starts a reachable HTTP server."""

from __future__ import annotations

import socket
import subprocess
import sys
import time

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
def server():
    """Start `weight-atlas serve` on a free loopback port, stop it afterwards."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "weight_atlas.cli", "serve", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
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
