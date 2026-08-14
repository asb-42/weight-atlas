# AGENTS.md — scripts

## Purpose

Diagnostic and release-support scripts that are not part of the package:
artefact diagnosis, distribution checks, and the Blender smoke test.

## Ownership

- `diagnose_*.py` (nan, distribution, k3, tensor-name coverage),
  `debug_visualization.py`, `release_check.sh`, `smoke_blender.sh`.

## Local Contracts

- **Run on real machines**: `diagnose_*.py` and `smoke_blender.sh` operate on
  scan artefacts. The dev machine's artefacts are stale — these scripts are
  intended to run on a machine with current artefacts (Blender smoke on a
  separate machine). Their results are not asserted in CI.
- **Venv**: scripts use `.venv/bin/python` when available (`release_check.sh`
  falls back to `python3`); run diagnostics with `.venv/bin/python`.
- **Read-only**: diagnostics must not mutate scan artefacts (no writes into
  `data/`, `artefacts*/`, `output/`).
- **Smoke test contract**: `smoke_blender.sh` verifies byte-identical PNGs on
  two runs (SHA-256) — determinism, not visual quality.

## Work Guidance

- New diagnostic: name `diagnose_<topic>.py`, CLI takes a path, uses the venv
  python, and prints a human-readable report. Prefer `tests/` for anything
  assertable; scripts are for exploration/reporting.

## Verification

- Shell scripts: `bash -n scripts/*.sh`. Python scripts: import + help smoke,
  or run against a real scan on a suitable machine. No CI hook.