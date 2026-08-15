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
  Job type lives in its own column (`job.job_type` = `scan`/`render`/
  `compare`), with per-type params in `renderer`, `compare_mode`,
  `compare_interp`. `job.message` is progress text only and is overwritten
  during execution + recovery — never parse the type back from it. `_migrate`
  backfills `job_type` from legacy `message` markers (`render:<id>`,
  `compare[:mode[:interp]]`) for pre-job_type DBs.
- **Deterministic job IDs**: `uuid4` is fine (not an output artefact).
- **File browsing is confined**: `GET /api/browse` must never escape the
  allowed roots (models/ + scan output dirs) — `_require_allowed` guard.
- **Compare payload**: `POST /api/compare` accepts `mode` (strict/aligned)
  and `interp` (linear/nearest); validate unknown values with 400 before
  enqueueing.
- **Errors surface on the job**: worker catches exceptions → `job.status =
  FAILED` + `job.error`; the API never raises unhandled.
- **Model detail is tabbed sub-pages**: `GET /models/{id}` is a light overview;
  sheets/terrain/stats/spec load as htmx fragments via `?tab=`. The statistics
  table is server-paginated (200 rows/page, clamped) — never render the full
  tensor table inline (74k-tensor fingerprints made the old page ~25 MB).
  Fragment templates are `ui/templates/_model_{tab}.html`; add new tabs to
  `model_tabs` in `routes.py` + a template.

## Work Guidance

- Prefer htmx partials + `hx-trigger` over full-page JS for UI updates.
- When adding an endpoint that reads scan artefacts, respect the allowed-roots
  confinement in routes.py.

## Verification

- `tests/test_api.py`, `tests/test_serve.py` (HTTP-level), `tests/test_jobs.py`
  style checks live in `tests/`; run via
  `cd /media/data/coding/weight-atlas && .venv/bin/python -m pytest tests/test_api.py tests/test_serve.py`.