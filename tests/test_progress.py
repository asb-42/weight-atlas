"""Tests for illustrative progress reporting in scan() and the job worker."""

from __future__ import annotations

import json
import time
from pathlib import Path

from tests.fixtures import make_fake_model
from weight_atlas.api.jobs import JobQueue, JobStatus
from weight_atlas.core.types import AtlasSpec
from weight_atlas.scan import scan as run_scan


def _spec() -> AtlasSpec:
    return AtlasSpec(
        spec_version=1,
        slots=["embed", "attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"],
        channels={
            "height": {"stat": "spectral_norm", "scale": {"type": "log1p"}},
            "tint": {"stat": "kurtosis", "scale": {"type": "log1p"}},
        },
        grid={"upsample": 2, "smooth_sigma": 1.0},
        sheet={"contour_levels": 4, "light_azdeg": 315, "light_altdeg": 45, "dpi": 72},
        seeds={"svd": 0},
        embedding={},
    )


def _write_spec(path: Path) -> Path:
    spec = _spec()
    raw = {
        "spec_version": spec.spec_version,
        "slots": spec.slots,
        "channels": spec.channels,
        "grid": spec.grid,
        "sheet": spec.sheet,
        "seeds": spec.seeds,
        "embedding": {},
    }
    path.write_text(json.dumps(raw))
    return path


def test_scan_reports_increasing_progress(tmp_path: Path) -> None:
    """scan() must emit monotonic, phase-labelled progress events."""
    model = tmp_path / "model.safetensors"
    make_fake_model(model, n_layers=2, hidden=16)

    events: list[tuple[float, str]] = []
    out = tmp_path / "out"
    run_scan(model, out, _spec(), progress=lambda pct, msg: events.append((pct, msg)))

    assert events, "scan must report progress"
    pcts = [p for p, _ in events]
    assert all(b >= a for a, b in zip(pcts, pcts[1:], strict=False)), "progress must be non-decreasing"
    assert pcts[0] <= 0.02
    assert pcts[-1] >= 0.95
    assert all(msg for _, msg in events), "every progress event needs a message"

    joined = " ".join(msg for _, msg in events).lower()
    assert "statistics" in joined
    assert "rasterizing" in joined
    assert "manifest" in joined


def test_worker_reports_phase_messages(tmp_path: Path) -> None:
    """A scan job run through the worker must surface granular phase messages."""
    model = tmp_path / "model.safetensors"
    make_fake_model(model, n_layers=1, hidden=8)
    spec_path = _write_spec(tmp_path / "atlas_spec.json")

    seen: list[str] = []
    queue = JobQueue(tmp_path / "jobs.db", on_job=lambda j: seen.append(j.message))
    job = queue.submit(model, tmp_path / "out", spec_path)
    queue.start()
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            cur = queue.get(job.job_id)
            if cur is not None and cur.status == JobStatus.DONE:
                break
            time.sleep(0.05)
        cur = queue.get(job.job_id)
        assert cur is not None
        assert cur.status == JobStatus.DONE
        assert cur.progress == 1.0
    finally:
        queue.stop()

    lowered = " ".join(m.lower() for m in seen)
    assert "statistics" in lowered, "worker should report a statistics phase"
    assert "rasterizing" in lowered, "worker should report a rasterizing phase"
    assert "manifest" in lowered, "worker should report a manifest phase"
