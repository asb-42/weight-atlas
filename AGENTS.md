# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if there are no specific standards or instructions yet, leave it empty
- Verification must reflect an existing check; if no verification framework exists yet, leave it empty and update it when one exists

Default section order:
- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## User Preferences

- **Python**: project requires the venv — use `.venv/bin/python` (a bare `python`
  is not on PATH). Never run tests/lint outside the venv.
- **Tests**: `python -m pytest tests/` (default addopts `-q`). Full suite is
  400+ tests and must stay green; determinism tests are part of the contract.
- **Git hygiene**: do not stage build artefacts — `data/jobs.db`, `*.pyc`,
  `*.npy` temp files, and other gitignored output. Stage only source, tests,
  specs, scripts, and docs. Filter with
  `git status --short | grep -v __pycache__ | grep -v "data/jobs.db"`.
- **Specs**: edit `specs/*.json` surgically (one-line additions via text edits),
  never reformat whole files — they use compact single-line blocks. Keep all
  spec versions in sync for shared extension keys (e.g. `compare.aligned_interp`).
- **Determinism is a feature**: renders, OBJ exports, and compare artefacts must
  stay byte-identical for identical inputs. Never introduce sampling noise,
  timestamps, or hash-order-dependent output.
- **Blender smoke test** runs on a separate machine with newer artefacts
  (this dev machine's artefacts are stale). Never assert on Blender output here;
  unit tests are dry-run (mocked `subprocess.run`).

## Child DOX Index

- `src/weight_atlas/AGENTS.md` — the main Python package: scan/render/compare
  pipeline, plugin registry conventions, and its own Child DOX Index.
- `specs/AGENTS.md` — atlas spec JSON files (versions v1–v2.4, activity protocol),
  extension and versioning rules.
- `tests/AGENTS.md` — test suite: fixtures, determinism tests, venv test command.
- `scripts/AGENTS.md` — diagnostic and smoke scripts (diagnose_*.py, smoke_blender.sh).
- `docs/AGENTS.md` — project documentation (ARCHITECTURE, user_manual, proposals, reports).

Root-owned files: `README.md`, `LICENSE`, `CHANGELOG.md`, `pyproject.toml`,
`uv.lock`, `banner.jpg`, `video-thumbnail.jpg`, `.github/`, and root-level
build/config files. The `poc/` directory is a frozen proof-of-concept, owned by root.
