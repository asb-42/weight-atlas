# AGENTS.md — api (web server)

## Purpose

FastAPI web UI backend: serves templates, exposes the job queue for
scan/render/compare, and provides file-browsing and result endpoints. The UI
(ui/) is the primary interface for the tool.

## Ownership

- `main.py` (app factory), `routes.py` (all HTTP endpoints), `jobs.py`
  (persistent job queue + worker thread).
- The `ui/` templates + static files are consumed here but owned by the
  parent (weight_atlas) doc; coordinate UI changes through `routes.py` /
  `main.py`.

## Local Contracts

- **Job queue**: `JobQueue` in `jobs.py` persists jobs in `data/jobs.db`.
  Job type is encoded in `job.message` — `scan`, `compare:{mode}:{interp}`,
  `render:{renderer_id}`. Keep the delimiter-based message format backward
  compatible when adding params.
- **Deterministic job IDs**: `uuid4` is fine (not an output artefact).
- **File browsing is confined**: `GET /api/browse` must never escape the
  allowed roots (models/ + scan output dirs) — `_require_allowed` guard.
- **Compare payload**: `POST /api/compare` accepts `mode` (strict/aligned)
  and `interp` (linear/nearest); validate unknown values with 400 before
  enqueueing.
- **Errors surface on the job**: worker catches exceptions → `job.status =
  FAILED` + `job.error`; the API never raises unhandled.

## Work Guidance

- Prefer htmx partials + `hx-trigger` over full-page JS for UI updates.
- When adding an endpoint that reads scan artefacts, respect the allowed-roots
  confinement in routes.py.

## Verification

- `tests/test_api.py`, `tests/test_serve.py` (HTTP-level), `tests/test_jobs.py`
  style checks live in `tests/`; run via
  `cd /media/data/coding/weight-atlas && .venv/bin/python -m pytest tests/test_api.py tests/test_serve.py`.