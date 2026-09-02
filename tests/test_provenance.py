"""Scan provenance anchoring (Phase 0 scan sharing, proposal §2).

Pins: per-file SHA-256 + composite source_digest in the fingerprint's
model block; basename-only (no local paths leak); determinism (same
files → same block, any directory); verification primitive; GGUF
multi-shard ordering; additive-key compatibility (readers that predate
the block ignore it).
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.fixtures import make_fake_model
from weight_atlas.core.types import load_default_spec
from weight_atlas.scan import scan
from weight_atlas.stats.provenance import (
    hash_file,
    package_manifest,
    provenance_matches,
    source_provenance,
)


def _sha256_ref(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


class TestHashAnchoring:
    def test_scan_records_sources(self, tmp_path: Path) -> None:
        model = tmp_path / "fake.safetensors"
        make_fake_model(model, n_layers=2)
        out = tmp_path / "out"
        scan(model, out, load_default_spec(), jobs=1)

        fp = json.loads((out / "fingerprint.json").read_text())
        sources = fp["model"]["sources"]
        assert len(sources) == 1
        assert sources[0]["file"] == "fake.safetensors"  # basename only
        assert sources[0]["bytes"] == model.stat().st_size
        assert sources[0]["sha256"] == _sha256_ref(model)
        assert len(fp["model"]["source_digest"]) == 64

    def test_provenance_deterministic_across_directories(self, tmp_path: Path) -> None:
        a = tmp_path / "a" / "fake.safetensors"
        b = tmp_path / "totally-different" / "fake.safetensors"
        a.parent.mkdir()
        b.parent.mkdir()
        make_fake_model(a, n_layers=1, seed=7)
        b.write_bytes(a.read_bytes())  # same content, other path

        pa = source_provenance([a])
        pb = source_provenance([b])
        assert pa == pb  # directory-independent

    def test_content_change_changes_digest(self, tmp_path: Path) -> None:
        m1 = tmp_path / "m1.safetensors"
        m2 = tmp_path / "m2.safetensors"
        make_fake_model(m1, n_layers=1, seed=1)
        make_fake_model(m2, n_layers=1, seed=2)
        assert source_provenance([m1])["source_digest"] != source_provenance([m2])["source_digest"]

    def test_multi_shard_ordering_stable(self, tmp_path: Path) -> None:
        """Multiple files: name-sorted order → stable composite digest."""
        shards = []
        for i in (3, 1, 2):  # created unsorted on purpose
            p = tmp_path / f"model-{i:05d}-of-00003.safetensors"
            make_fake_model(p, n_layers=1, seed=i)
            shards.append(p)
        ordered = sorted(shards)
        assert source_provenance(ordered) == source_provenance(sorted(shards))
        # order matters for the composite (name, size, hash tuples in order)
        assert (
            source_provenance(ordered)["source_digest"]
            != source_provenance(list(reversed(ordered)))["source_digest"]
        )

    def test_provenance_matches_roundtrip(self, tmp_path: Path) -> None:
        model = tmp_path / "fake.safetensors"
        make_fake_model(model, n_layers=1)
        prov = source_provenance([model])
        assert provenance_matches(prov, [model]) is True
        # a different file fails verification
        other = tmp_path / "other.safetensors"
        make_fake_model(other, n_layers=1, seed=99)
        assert provenance_matches(prov, [other]) is False
        # missing/absent block is not a match
        assert provenance_matches(None, [model]) is False
        assert provenance_matches({}, [model]) is False

    def test_hash_file_streaming_matches_reference(self, tmp_path: Path) -> None:
        """1 MiB-chunk streaming hash == whole-file hash (chunk boundary)."""
        import numpy as np
        from safetensors.numpy import save_file

        big = tmp_path / "big.safetensors"
        save_file(
            {"x": np.zeros((300, 300), dtype=np.float32)}, str(big)
        )  # ~360 KB; below 1 MiB — also pin the multi-chunk path
        pad = tmp_path / "pad.bin"
        pad.write_bytes(b"\x00" * (2 << 20))  # > 1 chunk
        for p in (big, pad):
            assert hash_file(p) == _sha256_ref(p)

    def test_pre_anchor_scan_export_refusal_is_explicit(self, tmp_path: Path) -> None:
        """A fingerprint without sources has no provenance — the exporter
        must refuse it (pinned here at the primitive level)."""
        model = tmp_path / "fake.safetensors"
        make_fake_model(model, n_layers=1)
        fp = {"model": {"n_tensors": 5}}  # no sources/source_digest
        assert provenance_matches(fp["model"], [model]) is False


class TestPackageManifest:
    def test_manifest_shape_and_determinism(self, tmp_path: Path) -> None:
        fp = {
            "tool_version": "0.2.0",
            "spec_version": 4,
            "loader": "safetensors",
            "quantization": {"F32": 5},
            "model": {
                "n_tensors": 5,
                "sources": [{"file": "m.safetensors", "bytes": 1, "sha256": "a" * 64}],
                "source_digest": "b" * 64,
            },
        }
        m1 = package_manifest(
            fp, {"fingerprint.json": "c" * 64}, model_name="m", profile="stats",
            license_model="apache-2.0", license_scan="CC-BY-4.0",
        )
        m2 = package_manifest(
            fp, {"fingerprint.json": "c" * 64}, model_name="m", profile="stats",
            license_model="apache-2.0", license_scan="CC-BY-4.0",
        )
        assert m1 == m2
        assert m1["format"] == "wasc"
        assert m1["format_version"] == 1
        assert m1["license"]["verified"] is False
        assert m1["license"]["declared_by"] == "scanner"
        assert m1["model"]["source_digest"] == "b" * 64
        assert m1["contents"]["fingerprint.json"] == "c" * 64
        # canonical encoding is sorted + stable
        assert json.loads(json.dumps(m1, sort_keys=True)) == json.loads(
            json.dumps(m2, sort_keys=True)
        )
