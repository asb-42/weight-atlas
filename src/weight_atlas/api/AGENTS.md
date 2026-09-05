# AGENTS.md — api (web server)

## Purpose

FastAPI web UI backend: serves templates, exposes the job queue for
scan/render/compare, and provides file-browsing and result endpoints. The UI
(ui/) is the primary interface for the tool.

## Ownership

- `main.py` (app factory + QueryError handler), `routes.py` (web UI HTTP
  endpoints), `query_routes.py` (LLM query API endpoints, spec v0.2),
  `query.py` (read-side query engine), `jobs.py` (persistent job queue +
  worker thread).
- The `ui/` templates + static files are consumed here but owned by the
  parent (weight_atlas) doc; coordinate UI changes through `routes.py` /
  `main.py`.

## Local Contracts

- **Job queue**: `JobQueue` in `jobs.py` persists jobs via a `JobStore`
  (`store.py`) — `SQLiteJobStore` for local dev/tests (default
  `data/jobs.db`), `MariaDBJobStore` for server deployments (Phase 1
  M1). Both backends share one ordered column list (`COLUMNS`), so
  `SELECT *` positional mapping can never drift; row dicts cross the
  boundary, never SQL. `WEIGHT_ATLAS_DB_URL=mysql://...` selects
  MariaDB (takes precedence over `WEIGHT_ATLAS_DB_PATH`). History
  migration is `weight-atlas db-copy --from <sqlite> --to <mysql-url>`
  (verbatim row transfer). MariaDB DDL avoids TEXT defaults
  (unsupported): sized VARCHAR/MEDIUMTEXT. `pymysql` is a `mysql`
  extra, imported lazily — the base install never requires it.
  Backend parity is pinned by `tests/test_store_backends.py` (shared
  battery vs SQLite always, vs live MariaDB when
  `WEIGHT_ATLAS_TEST_MYSQL_URL` is set, plus a serverless dialect
  unit test).
  Job type lives in its own column (`job.job_type` = `scan`/`render`/
  `compare`), with per-type params in `renderer`, `compare_mode`,
  `compare_interp`. `job.message` is progress text only and is overwritten
  during execution + recovery — never parse the type back from it. `_migrate`
  backfills `job_type` from legacy `message` markers (`render:<id>`,
  `compare[:mode[:interp]]`) for pre-job_type DBs. Restart recovery
  (`start()`) resets persisted `queued`/`running` jobs to `queued`
  (`re-queued after restart`); a background sweeper thread additionally
  re-queues `running` rows whose `updated_at` is older than
  `_STALE_RUNNING_SECONDS` (5 min) and not the worker's current job
  (`re-queued after stale running`) — a job marked running by a process that
  died after startup would otherwise show as stuck/running forever.
- **Per-render sheet overrides**: `job.sheet_knobs` (JSON dict, `sheet_knobs`
  column) carries optional per-render overrides for a single render. The
  worker overlays them onto the spec (`_apply_sheet_knobs`,
  dataclasses.replace) before rendering — the scan's recorded spec is never
  mutated. Sheet display knobs (`normalized_depth`, `drop_empty_cols`) map
  onto the spec's `sheet` block; the UI sends them as checkbox form fields to
  `/api/jobs/{id}/render/sheet`. The fractal-mode knob (`fractal_mode` =
  `fbm`/`sdf`) overlays onto the spec's `fractal.mode`; the UI sends it as a
  `<select name="fractal_mode">` alongside the `/render/fractal` button.
  Only the raster sheet renderers accept sheet knobs, only the fractal
  renderer accepts `fractal_mode`; unknown knob values are dropped. Default
  empty.
- **Deterministic job IDs**: `uuid4` is fine (not an output artefact).
- **File browsing is confined**: `GET /api/browse` must never escape the
  allowed roots (models/ + scan output dirs) — `_require_allowed` guard.
- **Compare payload**: `POST /api/compare` accepts `mode` (strict/aligned)
  and `interp` (linear/nearest); validate unknown values with 400 before
  enqueueing. Both `dir_a`/`dir_b` must be actual scan output dirs —
  `manifest.json` presence is validated up front (400 otherwise), because a
  comparison consumes scan artefacts (`manifest.json` + `field_*.tif`) and a
  compare/render output dir (delta sheets, etc.) would otherwise fail inside
  the worker. The compare page candidate list only surfaces DONE **scan**
  jobs (`job_type == "scan"`) — render jobs may point their `out_dir` at a
  compare output dir and must never be offered as a model to compare.
- **Compare report shows model names**: `GET /compare/{job_id}` derives the
  display names from `job.model_path` (`"dir_a|dir_b"` → `dir_a.name` /
  `dir_b.name`) and passes them to the template as `model_a_name` /
  `model_b_name` — the fingerprint carries no human-readable display name, so
  the scan dir names are the labels shown on the report instead of bare
  "A"/"B".
- **Errors surface on the job**: worker catches exceptions → `job.status =
  FAILED` + `job.error` (exception type + truncated traceback, also logged
  with `exc_info`); the API never raises unhandled. Per-render channel
  failures are logged and surfaced via the completion message
  (`Complete (N render failure(s): ...)`) — the artefacts list stays a pure
  list of file names.
- **Output dirs are unique per job**: scan jobs write to
  `output_root/<model_name>_<uuid8>` (same convention as `compare_*`
  outputs) so same-named models in different roots never overwrite each
  other. `POST /api/import` runs the heavy render via
  `run_in_threadpool` — never on the event loop.
- **LLM query API is a separate router**: `query_routes.py` owns the read-only
  `/api`, `/api/schema`, `/api/models`, `/api/model/{model_id}/*` endpoints
  (spec `docs/2026-08-16_weight-atlas-api-spec-v0.2.md`); `routes.py` owns the
  web UI. `model_id` == DONE scan job's `job_id`; a model is a DONE job whose
  `out_dir/fingerprint.json` exists. All analytics live in `query.py` (pure
  functions over the fingerprint) — never inline statistics in the router.
  Derived tensor records are cached (`_load_records`) under the fingerprint's
  `(path, mtime_ns, size)` key; percentile ranks use ascending-sorted arrays
  with side='right' (weak) semantics. Invalid filter input (e.g.
  `layer=abc`) raises `QueryError(400)` via the error envelope, never a bare
  500. Errors use the spec's `{error: {code, type, message, hint}}` envelope
  via `QueryError` (handled in `main.py`). Response bodies must stay
  deterministic: fixed ordering, no timestamps, floats rounded to 4 decimals.
- **Model detail is tabbed sub-pages**: `GET /models/{id}` is a light overview;
  sheets/terrain/stats/scatter/records/spec load as htmx fragments via `?tab=`.
  The statistics table is server-paginated (200 rows/page, clamped) — never render the full
  tensor table inline (74k-tensor fingerprints made the old page ~25 MB).
  **htmx fragment protocol**: requests with `HX-Request: true` get the BARE
  tab fragment (`HTMLResponse(ctx["tab_content"])`); only direct URLs get
  the full page. A full page swapped into `#model-tab-content` duplicated
  header/nav/footer on every tab click — pinned by
  `test_htmx_tab_request_gets_fragment_only`.
  Fragment templates are `ui/templates/_model_{tab}.html`; add new tabs to
  `model_tabs` in `routes.py` + a template. The scatter tab renders a
  deterministic server-side SVG (`x`/`y` metric params, p1–p99 clamped axes,
  log10 when a metric spans ≥2 orders of magnitude, stride-culled to
  `SCATTER_CAP`); the records tab renders extremes boards
  (`query.RECORD_BOARDS` → `extreme_records`), linking into the stats table
  page the tensor sits on. Both are pure `query.py` data + presentation in
  `routes.py` — no client JS.
- **Query-API twins of the UI tabs**: `GET /api/model/{id}/scatter`
  (x/y metric params, cap-clamped points, axis configs) and
  `GET /api/model/{id}/records` (boards, optional `metric` filter) expose
  the same pure `query.py` data to agents — `scatter_body` /
  `records_body` in `query.py`, registered in the discovery body. Errors
  via the spec's QueryError envelope.
- **Package endpoints (Phase 0 scan sharing)**: `POST /api/packages/prepare`
  (job_id + profile stats|full → deterministic `.wasc` under
  `output_root/packages/`) and `POST /api/packages` (local package PATH →
  verified extract + register via `import_package`/`import_scan`).
  LAN-local by contract: public upload/download is Phase 1 and NOT
  IMPLEMENTED — the prepare response's `note` says so and a public
  deployment must not expose these. Import auto-render titles use
  `Path(model_path).name` — never the full local path (shared PNGs are
  public artefacts).

## Work Guidance

- Prefer htmx partials + `hx-trigger` over full-page JS for UI updates.
- When adding an endpoint that reads scan artefacts, respect the allowed-roots
  confinement in routes.py.

## Verification

- `tests/test_api.py`, `tests/test_serve.py` (HTTP-level), `tests/test_jobs.py`
  style checks live in `tests/`; run via
  `cd /media/data/coding/weight-atlas && .venv/bin/python -m pytest tests/test_api.py tests/test_serve.py`.