"""FastAPI application factory for weight-atlas web UI."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from weight_atlas.api import jobs as jobmod
from weight_atlas.api.query import QueryError
from weight_atlas.api.query_routes import create_query_router
from weight_atlas.api.routes import create_router
from weight_atlas.core.types import get_default_spec_path, load_default_spec
from weight_atlas.render import (  # noqa: F401 — registers renderers
    blender,
    fractal,
    matplotlib_sheet,
    preview,
)


def create_app(
    db_path: Path | None = None,
    db_url: str | None = None,
    spec_path: Path | None = None,
    output_root: Path | None = None,
    model_roots: list[Path] | None = None,
) -> FastAPI:
    """Create the FastAPI application.

    Args:
        db_path: Path to SQLite job database. Defaults to ./data/jobs.db,
        overridable via ``WEIGHT_ATLAS_DB_PATH``.
        db_url: MariaDB URL (``mysql://user:pass@host:port/dbname``),
        overridable via ``WEIGHT_ATLAS_DB_URL``. Takes precedence over
        ``db_path`` when set — this is the server-deployment backend
        (Phase 1 M1); local dev keeps SQLite.
        spec_path: Path to atlas spec JSON. Defaults to the canonical
            ``get_default_spec_path()`` (atlas_spec.v2.4.json).
        output_root: Root directory for scan outputs. Defaults to ./output,
        overridable via ``WEIGHT_ATLAS_OUTPUT_ROOT``.
        model_roots: Optional allowlist of directories from which scan/import/
            compare paths are accepted. When None (default) any existing path is
            accepted — only safe for localhost/LAN-trusted deployments.

    Environment overrides exist so that out-of-tree `serve` launches (the
    `test_serve.py` smoke test runs the real CLI binary) can redirect the
    database and outputs to a sandbox: the smoke test must never touch the
    developer's real job database — its startup recovery resets rows in it
    ("re-queued after restart") and its sweeper can clobber live jobs.
    """
    base = Path(__file__).resolve().parent.parent.parent.parent
    _db_path = db_path or Path(os.environ.get("WEIGHT_ATLAS_DB_PATH") or (base / "data" / "jobs.db"))
    if spec_path is None:
        # Assert the shipped default spec is canonical before serving, so the
        # web UI and CLI can never silently produce incompatible fingerprints.
        load_default_spec()
    _spec_path = spec_path or get_default_spec_path()
    _output_root = output_root or Path(os.environ.get("WEIGHT_ATLAS_OUTPUT_ROOT") or (base / "output"))
    _model_roots = [r.resolve() for r in model_roots] if model_roots else None

    _db_path.parent.mkdir(parents=True, exist_ok=True)
    _output_root.mkdir(parents=True, exist_ok=True)

    _db_url = db_url or os.environ.get("WEIGHT_ATLAS_DB_URL")
    if _db_url:
        from weight_atlas.api.store import MariaDBJobStore

        store: Any = MariaDBJobStore.from_url(_db_url)
        job_queue = jobmod.JobQueue(None, on_job=lambda j: None, store=store)
    else:
        job_queue = jobmod.JobQueue(_db_path, on_job=lambda j: None)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Start the worker on startup, stop it cleanly on shutdown."""
        job_queue.start()
        yield
        job_queue.stop()

    app = FastAPI(title="Weight Atlas", version="0.2.0", lifespan=lifespan)

    # Output artefacts (PNG, TIFF, JSON) must be mounted BEFORE /static, or the
    # broader /static mount shadows /static/outputs and every image/JSON served
    # from a scan output directory 404s (e.g. the compare report's delta sheets).
    if _output_root.exists():
        app.mount("/static/outputs", StaticFiles(directory=str(_output_root)), name="outputs")

    # Static files (CSS)
    static_dir = Path(__file__).resolve().parent.parent / "ui" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Templates
    templates_dir = Path(__file__).resolve().parent.parent / "ui" / "templates"
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory=str(templates_dir))

    router = create_router(job_queue, templates, _spec_path, _output_root, _model_roots)
    app.include_router(router)

    query_router = create_query_router(job_queue)
    app.include_router(query_router)

    @app.exception_handler(QueryError)
    async def query_error_handler(_request: Request, exc: QueryError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_body())

    return app


# Module-level app so both `uvicorn weight_atlas.api.main:app` and the
# `weight-atlas serve` factory mode work. The background job worker is NOT
# started here — it starts inside the lifespan context manager — so importing
# this module has no thread side effects (safe under `--reload`).
app = create_app()
