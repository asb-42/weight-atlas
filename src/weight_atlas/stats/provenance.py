"""Scan provenance: anchor fingerprints to their source model files.

Phase 0 scan sharing (docs/2026-09-02_scan-sharing-phase0.md §2). Every
scan records per-file SHA-256 of the model files it consumed, plus a
composite ``source_digest`` that identifies the file set independently of
scan settings. Determinism: file lists are name-sorted (loader discovery
invariant), digests are pure functions of file contents.

Trust model (stated honestly): the hashes make a fingerprint checkable by
anyone HOLDING the model (re-scan → same hashes → same fingerprint). They
do not make it checkable from the fingerprint alone — that needs
recomputation (determinism guarantees success) or a trusted registry
(Phase 1, out of scope).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def hash_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, streaming in 1 MiB chunks (never whole-file)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def source_provenance(files: list[Path]) -> dict[str, object]:
    """Per-file hashes + composite digest for a name-sorted file list.

    Returns the fingerprint-ready block:
    ``{"sources": [{"file", "bytes", "sha256"}], "source_digest": "…"}``.
    File names are BASENAMES only — no local paths leak into shareable
    data. The composite digest hashes the (name, bytes, sha256) tuples in
    list order, so it is stable for the same file set regardless of the
    directory it was scanned from.
    """
    sources: list[dict[str, object]] = []
    for f in files:
        sources.append(
            {
                "file": f.name,
                "bytes": f.stat().st_size,
                "sha256": hash_file(f),
            }
        )
    payload = "\n".join(
        f"{s['file']}|{s['bytes']}|{s['sha256']}" for s in sources
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return {"sources": sources, "source_digest": digest}


def provenance_matches(
    provenance: dict[str, object] | None, files: list[Path]
) -> bool:
    """True when ``files`` hash exactly to the recorded provenance.

    The re-scan verification primitive: same file set → same digests →
    the fingerprint's numbers are recomputable. Used by the export path
    (refuse pre-anchor scans carry no block) and available to any future
    registry check.
    """
    if not isinstance(provenance, dict) or not provenance.get("sources"):
        return False
    try:
        return source_provenance(files) == provenance
    except OSError:
        return False


# ── .wasc package manifest helpers (proposal §3.2) ─────────────────────────


def package_manifest(
    fingerprint: dict[str, object],
    contents_hashes: dict[str, str],
    *,
    model_name: str = "",
    profile: str = "stats",
    license_model: str = "",
    license_scan: str = "",
) -> dict[str, object]:
    """Build the ``package.json`` block for a .wasc export.

    ``contents_hashes`` maps package-relative paths → sha256 (computed by
    the exporter over the exact bytes it writes). Deterministic for the
    same scan inputs.
    """
    model = fingerprint.get("model", {})
    if not isinstance(model, dict):
        model = {}
    return {
        "format": "wasc",
        "format_version": 1,
        "profile": profile,
        "created_by": {
            "tool": "weight-atlas",
            "tool_version": str(fingerprint.get("tool_version", "")),
            "spec_version": str(fingerprint.get("spec_version", "")),
        },
        "model": {
            "name": model_name,
            "sources": model.get("sources", []),
            "source_digest": model.get("source_digest", ""),
        },
        "scan": {
            "loader": str(fingerprint.get("loader", "")),
            "n_tensors": model.get("n_tensors", 0),
            "quantization": str(fingerprint.get("quantization", "")),
        },
        "license": {
            "model_license": license_model,
            "scan_license": license_scan,
            "declared_by": "scanner",
            "verified": False,
        },
        "contents": contents_hashes,
        "provenance_note": (
            "hashes anchor the scan to model files; verification requires "
            "the model (determinism) or a trusted registry"
        ),
    }


def manifest_json(manifest: dict[str, object]) -> str:
    """Canonical JSON encoding for package manifests (sorted, stable)."""
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"
