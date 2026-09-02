# AGENTS.md — sharing

## Purpose

Shareable scan packages (`.wasc`) — Phase 0 scan sharing
(`docs/2026-09-02_scan-sharing-phase0.md`): deterministic export of a
scan directory into a versioned zip with provenance + license
declarations, and verified import (hashes, versions, zip-slip guards).

## Ownership

- `package.py` (export/import + the `.wasc` format). Provenance hashing
  primitives live in `stats/provenance.py` (owned by parent doc).

## Local Contracts

- **Determinism**: two exports of the same scan are byte-identical —
  fixed zip timestamps (1980-01-01), sorted entry order, deflate-6,
  canonical (sorted-key) JSON manifests. Pinned by test.
- **Provenance is mandatory for sharing**: export refuses fingerprints
  without `model.sources` (pre-anchor scans); import refuses packages
  whose fingerprint lacks them. Hash-anchoring is what makes a package
  verifiable by re-scan.
- **Import hardening**: every content hash verified BEFORE extraction;
  zip-slip / absolute path / symlink entries refused; fingerprint
  re-verified after extraction; foreign `spec_version` hard-rejected
  (same policy as compare).
- **Local-only**: extraction happens into a user-chosen directory; there
  is NO public upload surface here. The server exposes LAN-local package
  endpoints (`POST /api/packages/prepare`, `POST /api/packages` in
  `routes.py` — verified extract + register); public upload/download is
  Phase 1, explicitly NOT IMPLEMENTED.
- **Additive format versioning**: `format_version: 1`; new keys are
  tolerated by old readers, never bump the version for additions.

## Work Guidance

- Extend profiles in `_PROFILES`; keep `stats` = fingerprint-only.
- New verification guards go BEFORE extraction in `import_package`.

## Verification

- `cd /media/data/coding/weight-atlas && .venv/bin/python -m pytest tests/test_package_share.py tests/test_provenance.py`
- Full suite: `.venv/bin/python -m pytest tests/`

## Child DOX Index

(none — this is a leaf module)
