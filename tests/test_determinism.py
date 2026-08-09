"""Determinism tests: scan twice → byte-identical manifest."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tests.fixtures import make_fake_model
from weight_atlas.core.types import AtlasSpec
from weight_atlas.scan import scan as run_scan


def test_scan_twice_same_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model = tmp / "model.safetensors"
        make_fake_model(model)
        spec = AtlasSpec.from_json(Path("specs/atlas_spec.v2.json"))

        out1 = tmp / "out1"
        out2 = tmp / "out2"
        run_scan(model, out1, spec)
        run_scan(model, out2, spec)

        m1 = json.loads((out1 / "manifest.json").read_text())
        m2 = json.loads((out2 / "manifest.json").read_text())
        assert m1 == m2

        # Same fingerprint content.
        f1 = json.loads((out1 / "fingerprint.json").read_text())
        f2 = json.loads((out2 / "fingerprint.json").read_text())
        assert f1 == f2


def test_tif_byte_identical():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model = tmp / "model.safetensors"
        make_fake_model(model)
        spec = AtlasSpec.from_json(Path("specs/atlas_spec.v2.json"))

        out1 = tmp / "out1"
        out2 = tmp / "out2"
        run_scan(model, out1, spec)
        run_scan(model, out2, spec)

        for p in out1.glob("field_*.tif"):
            twin = out2 / p.name
            assert twin.exists()
            assert p.read_bytes() == twin.read_bytes()
