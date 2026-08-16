# AGENTS.md — tests

## Purpose

Test suite (400+ tests, 599 current) covering loaders, fields, render,
compare, api, LLM query API, determinism, paired (qimpact + edit signatures),
and degradation paths. The suite is the quality gate for every change.

## Ownership

- `tests/` root: `conftest.py` (fixtures/registry isolation), `fixtures.py`
  + `fixtures/` (seeded fake-model factories, name-mapping fixtures), and all
  `test_*.py` modules.

## Local Contracts

- **Venv required**: run only via
  `cd /media/data/coding/weight-atlas && .venv/bin/python -m pytest tests/`
  (addopts `-q`). A bare `python` is not on PATH.
- **Determinism tests are contract**: `test_determinism.py` and per-module
  determinism tests (delta sheet, OBJ, TIFF) must stay green — they pin
  byte-identical output.
- **Blender tests are dry-run**: never invoke a real Blender binary or assert
  on its output here; mock `subprocess.run`. Real renders happen on a separate
  machine via `scripts/smoke_blender.sh`.
- **Fixtures are seeded**: use `np.random.default_rng(seed)` / fixed seeds in
  `fixtures.py`; never unseeded RNG. New fixtures belong in `fixtures.py` or
  `fixtures/`.
- **Registry isolation**: tests that exercise the plugin registry must use the
  `_isolated_registry`-style fixture to avoid cross-test registration bleed.

## Work Guidance

- New feature → new test in the matching `test_*.py`; extend determinism
  coverage when the feature emits an artefact.
- Keep the suite fast enough to run locally before commit.

## Verification

- `cd /media/data/coding/weight-atlas && .venv/bin/python -m pytest tests/`
  — full suite must pass before any commit.