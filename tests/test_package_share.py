"""Shareable scan packages (.wasc) — export/import (Phase 0, proposal §3–4).

Pins: byte-identical double export, export→import round-trip preserving
fingerprint.json byte-identically, provenance refusal (pre-anchor scans),
corrupt-hash refusal, zip-slip refusal, foreign spec_version refusal,
profile contents.
"""

from __future__ import annotations

import json
import stat
import zipfile
from pathlib import Path

import pytest

from tests.fixtures import make_fake_model
from weight_atlas.core.types import load_default_spec
from weight_atlas.scan import scan
from weight_atlas.sharing.package import (
    PackageError,
    export_package,
    import_package,
)


@pytest.fixture(scope="module")
def anchored_scan(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real (tiny) scan with provenance, exported once for the module."""
    tmp = tmp_path_factory.mktemp("wasc")
    model = tmp / "fake.safetensors"
    make_fake_model(model, n_layers=2)
    out = tmp / "scan"
    scan(model, out, load_default_spec(), jobs=1)
    return out


class TestExport:
    def test_export_deterministic(self, anchored_scan: Path, tmp_path: Path) -> None:
        a = export_package(anchored_scan, tmp_path / "a.wasc", model_name="m")
        b = export_package(anchored_scan, tmp_path / "b.wasc", model_name="m")
        assert a.read_bytes() == b.read_bytes()

    def test_stats_profile_contents(self, anchored_scan: Path, tmp_path: Path) -> None:
        pkg = export_package(anchored_scan, tmp_path / "s.wasc", profile="stats")
        with zipfile.ZipFile(pkg) as zf:
            names = set(zf.namelist())
        assert names == {"package.json", "fingerprint.json"}

    def test_full_profile_includes_fields(self, anchored_scan: Path, tmp_path: Path) -> None:
        pkg = export_package(anchored_scan, tmp_path / "f.wasc", profile="full")
        with zipfile.ZipFile(pkg) as zf:
            names = set(zf.namelist())
        assert "manifest.json" in names
        assert any(n.startswith("field_") for n in names)

    def test_manifest_declares_licenses_and_hashes(
        self, anchored_scan: Path, tmp_path: Path
    ) -> None:
        pkg = export_package(
            anchored_scan, tmp_path / "l.wasc",
            model_name="m", license_model="apache-2.0", license_scan="CC-BY-4.0",
        )
        with zipfile.ZipFile(pkg) as zf:
            m = json.loads(zf.read("package.json"))
        assert m["license"]["model_license"] == "apache-2.0"
        assert m["license"]["scan_license"] == "CC-BY-4.0"
        assert m["license"]["verified"] is False
        assert set(m["contents"]) >= {"fingerprint.json"}
        assert len(m["model"]["source_digest"]) == 64

    def test_refuses_pre_anchor_scan(self, tmp_path: Path) -> None:
        """A fingerprint without model.sources is not shareable."""
        d = tmp_path / "old_scan"
        d.mkdir()
        (d / "fingerprint.json").write_text(json.dumps({"model": {"n_tensors": 3}}))
        with pytest.raises(PackageError, match="provenance"):
            export_package(d, tmp_path / "x.wasc")


class TestImport:
    def test_roundtrip_byte_identical(self, anchored_scan: Path, tmp_path: Path) -> None:
        pkg = export_package(anchored_scan, tmp_path / "r.wasc", profile="full")
        dest = tmp_path / "restored"
        import_package(pkg, dest)
        a = (anchored_scan / "fingerprint.json").read_bytes()
        b = (dest / "fingerprint.json").read_bytes()
        assert a == b
        # every field tif survived with identical bytes
        src_tifs = sorted(p.name for p in anchored_scan.glob("field_*.tif"))
        dst_tifs = sorted(p.name for p in dest.glob("field_*.tif"))
        assert src_tifs == dst_tifs
        for name in src_tifs:
            assert (anchored_scan / name).read_bytes() == (dest / name).read_bytes()

    def test_refuses_corrupt_hash(self, anchored_scan: Path, tmp_path: Path) -> None:
        pkg = export_package(anchored_scan, tmp_path / "c.wasc")
        # tamper with the fingerprint inside the package
        with zipfile.ZipFile(pkg) as zf:
            entries = {n: zf.read(n) for n in zf.namelist()}
        fp = json.loads(entries["fingerprint.json"])
        fp["tensors"]["x"] = {}  # the tamper
        entries["fingerprint.json"] = json.dumps(fp, sort_keys=True).encode()
        with zipfile.ZipFile(pkg, "w", zipfile.ZIP_DEFLATED) as zf:
            for n, data in entries.items():
                zf.writestr(n, data)
        with pytest.raises(PackageError, match="hash mismatch"):
            import_package(pkg, tmp_path / "dest")

    @staticmethod
    def _evil_package(path: Path, extra: None = None) -> None:
        """Minimal valid .wasc manifest + hash-anchored fingerprint, so the
        extraction guards themselves are exercised."""
        fp = {"model": {"n_tensors": 1, "sources": [{"file": "m", "bytes": 1, "sha256": "a" * 64}]}}
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(
                "package.json",
                json.dumps({"format": "wasc", "format_version": 1, "contents": {}}),
            )
            zf.writestr("fingerprint.json", json.dumps(fp, sort_keys=True))

    def test_refuses_zip_slip(self, tmp_path: Path) -> None:
        evil = tmp_path / "evil.wasc"
        self._evil_package(evil)
        with zipfile.ZipFile(evil, "a") as zf:
            zf.writestr("../escaped.txt", "pwn")
        with pytest.raises(PackageError, match="unsafe|escapes"):
            import_package(evil, tmp_path / "dest")

    def test_refuses_symlink_entry(self, tmp_path: Path) -> None:
        """A symlink entry (unix create_system=3, S_IFLNK mode) is refused."""
        evil = tmp_path / "link.wasc"
        self._evil_package(evil)
        with zipfile.ZipFile(evil, "a") as zf:
            zi = zipfile.ZipInfo("real.txt")
            zi.create_system = 3
            zi.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(zi, "/etc/passwd")
        with pytest.raises(PackageError, match="symlink"):
            import_package(evil, tmp_path / "dest")

    def test_refuses_not_a_package(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain.zip"
        with zipfile.ZipFile(plain, "w") as zf:
            zf.writestr("hello.txt", "hi")
        with pytest.raises(PackageError, match="package.json"):
            import_package(plain, tmp_path / "dest")

    def test_refuses_missing_package(self, tmp_path: Path) -> None:
        with pytest.raises(PackageError, match="not found"):
            import_package(tmp_path / "nope.wasc", tmp_path / "dest")
