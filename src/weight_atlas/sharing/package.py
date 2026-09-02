"""Shareable scan packages (.wasc) — export/import (Phase 0, proposal §3–4).

A package is a deterministic zip: fixed timestamps, sorted entry names,
deflate-6. Contents: ``package.json`` (manifest with content hashes and
license declarations) + the scan artefacts (fingerprint always; fields and
renders per profile). Import verifies every inner hash, rejects zip-slip
and foreign versions, then extracts for registration via the existing
``JobQueue.import_scan``.

Determinism contract: two exports of the same scan are byte-identical
(pinned by test), the export→import round-trip preserves
``fingerprint.json`` byte-identically.
"""

from __future__ import annotations

import json
import stat
import zipfile
from pathlib import Path

from weight_atlas.stats.provenance import (
    hash_file,
    package_manifest,
)

# Zip determinism: fixed DOS timestamp (zip has no TZ; 1980-01-01 is the
# format's epoch), so exports never encode wall-clock time.
_FIXED_DATE = (1980, 1, 1, 0, 0, 0)

# What each profile packages. Order irrelevant (names are sorted on write).
_PROFILES: dict[str, tuple[str, ...]] = {
    "stats": ("fingerprint.json",),
    "full": ("fingerprint.json", "manifest.json"),
}


class PackageError(Exception):
    """Refusal during export/import — message is user-facing."""


def _profile_files(scan_dir: Path, profile: str) -> list[Path]:
    """Deterministic artefact list for a profile: named files + field tifs."""
    if profile not in _PROFILES:
        raise PackageError(f"unknown profile {profile!r} (want {'/'.join(_PROFILES)})")
    files = [f for f in _PROFILES[profile] if (scan_dir / f).exists()]
    if "fingerprint.json" not in [f for f in files]:
        raise PackageError(f"{scan_dir} has no fingerprint.json — not a scan dir")
    if profile == "full":
        files.extend(sorted(p.name for p in scan_dir.glob("field_*.tif")))
        files.extend(sorted(f"render/{p.name}" for p in (scan_dir / "render").glob("*.png")))
    return [scan_dir / f for f in files]


def export_package(
    scan_dir: Path,
    out_path: Path,
    *,
    profile: str = "stats",
    model_name: str = "",
    license_model: str = "",
    license_scan: str = "",
) -> Path:
    """Write a .wasc package for a scan directory.

    Refuses scans whose fingerprint lacks provenance (pre-anchor scans):
    a shareable package must be verifiable, and without source hashes it
    is an unverifiable claim (proposal §2.3).
    """
    fp_path = scan_dir / "fingerprint.json"
    if not fp_path.exists():
        raise PackageError(f"{scan_dir} has no fingerprint.json — not a scan dir")
    fingerprint = json.loads(fp_path.read_text())
    model = fingerprint.get("model", {})
    if not isinstance(model, dict) or not model.get("sources"):
        raise PackageError(
            "fingerprint has no model.sources provenance — re-scan with a "
            "current tool version before exporting (shareable packages "
            "must be hash-anchored)"
        )

    artefacts = _profile_files(scan_dir, profile)
    contents: dict[str, str] = {}
    for f in artefacts:
        rel = f.relative_to(scan_dir).as_posix()
        contents[rel] = hash_file(f)

    manifest = package_manifest(
        fingerprint,
        contents,
        model_name=model_name or scan_dir.name,
        profile=profile,
        license_model=license_model,
        license_scan=license_scan,
    )
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zi = zipfile.ZipInfo("package.json", date_time=_FIXED_DATE)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, manifest_bytes)
        for f in sorted(artefacts, key=lambda p: p.relative_to(scan_dir).as_posix()):
            rel = f.relative_to(scan_dir).as_posix()
            zi = zipfile.ZipInfo(rel, date_time=_FIXED_DATE)
            zi.compress_type = zipfile.ZIP_DEFLATED
            with open(f, "rb") as fh:
                zf.writestr(zi, fh.read())
    return out_path


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract with zip-slip guards: no absolute paths, no .., no symlinks."""
    dest = dest.resolve()
    for info in zf.infolist():
        target = (dest / info.filename).resolve()
        if not str(target).startswith(str(dest) + "/"):
            raise PackageError(f"package entry escapes target dir: {info.filename!r}")
        if info.filename.startswith("/") or ".." in Path(info.filename).parts:
            raise PackageError(f"unsafe package entry: {info.filename!r}")
        if info.create_system == 3 and (info.external_attr >> 16) & 0o170000 == stat.S_IFLNK:
            raise PackageError(f"symlink entry refused: {info.filename!r}")
        zf.extract(info, dest)


def import_package(pkg_path: Path, out_dir: Path) -> Path:
    """Verify + extract a .wasc package; returns the scan dir.

    Verifies: format/spec versions, every content hash, then extracts with
    zip-slip guards. The caller registers the result (e.g. via
    ``JobQueue.import_scan``).
    """
    if not pkg_path.exists():
        raise PackageError(f"package not found: {pkg_path}")
    try:
        zf = zipfile.ZipFile(pkg_path)
    except zipfile.BadZipFile as exc:
        raise PackageError(f"not a valid .wasc package: {exc}") from exc

    with zf:
        names = set(zf.namelist())
        if "package.json" not in names:
            raise PackageError("package.json missing — not a .wasc package")
        manifest = json.loads(zf.read("package.json"))
        if manifest.get("format") != "wasc":
            raise PackageError(f"format {manifest.get('format')!r} != 'wasc'")
        if manifest.get("format_version") != 1:
            raise PackageError(
                f"unsupported format_version {manifest.get('format_version')} (want 1)"
            )

        # verify every content hash BEFORE extracting anything
        for rel, want in manifest.get("contents", {}).items():
            if rel not in names:
                raise PackageError(f"manifest lists {rel!r} but package lacks it")
            import hashlib

            got = hashlib.sha256(zf.read(rel)).hexdigest()
            if got != want:
                raise PackageError(f"content hash mismatch for {rel!r} — package corrupt")

        # spec_version governs numeric compatibility (compare policy)
        fp = json.loads(zf.read("fingerprint.json")) if "fingerprint.json" in names else {}
        pkg_spec = fp.get("spec_version")
        if pkg_spec is not None and str(pkg_spec) not in ("1", "2", "2.1", "2.2", "2.3", "2.4", "4"):
            raise PackageError(f"foreign spec_version {pkg_spec!r} — refuse import")

        # fingerprint must carry provenance to be registrable
        model = fp.get("model", {})
        if not isinstance(model, dict) or not model.get("sources"):
            raise PackageError("packaged fingerprint lacks provenance — refuse import")

        out_dir.mkdir(parents=True, exist_ok=True)
        _safe_extract(zf, out_dir)

        # re-verify on disk after extraction (paranoia: extraction wrote it)
        got_fp = json.loads((out_dir / "fingerprint.json").read_text())
        if got_fp != fp:
            raise PackageError("fingerprint changed during extraction — package corrupt")
    return out_dir
