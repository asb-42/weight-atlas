"""End-to-end: scan + render via CLI."""

from __future__ import annotations

import tempfile
from pathlib import Path

from tests.fixtures import make_fake_model
from weight_atlas.cli import main


def test_cli_scan_render():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model = tmp / "model.safetensors"
        make_fake_model(model)
        out = tmp / "artefacts"

        assert main(["scan", str(model), "--out", str(out)]) == 0
        assert (out / "fingerprint.json").exists()
        assert (out / "manifest.json").exists()
        tifs = list(out.glob("field_*.tif"))
        assert len(tifs) >= 3  # at least one channel

        assert main(["render", str(out)]) == 0
