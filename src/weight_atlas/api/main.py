"""FastAPI application factory for weight-atlas web UI."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from weight_atlas.api import jobs as jobmod
from weight_atlas.api.routes import create_router
from weight_atlas.core.types import get_default_spec_path, load_default_spec
from weight_atlas.render import (  # noqa: F401 — registers renderers
    blender,
    matplotlib_sheet,
    preview,
)


def create_app(
    db_path: Path | None = None,
    spec_path: Path | None = None,
    output_root: Path | None = None,
    model_roots: list[Path] | None = None,
) -> FastAPI:
    """Create the FastAPI application.

    Args:
        db_path: Path to SQLite job database. Defaults to ./data/jobs.db
        spec_path: Path to atlas spec JSON. Defaults to the canonical
            ``get_default_spec_path()`` (atlas_spec.v2.4.json).
        output_root: Root directory for scan outputs. Defaults to ./output
        model_roots: Optional allowlist of directories from which scan/import/
            compare paths are accepted. When None (default) any existing path is
            accepted — only safe for localhost/LAN-trusted deployments.
    """
    base = Path(__file__).resolve().parent.parent.parent.parent
    _db_path = db_path or base / "data" / "jobs.db"
    if spec_path is None:
        # Assert the shipped default spec is canonical before serving, so the
        # web UI and CLI can never silently produce incompatible fingerprints.
        load_default_spec()
    _spec_path = spec_path or get_default_spec_path()
    _output_root = output_root or base / "output"
    _model_roots = [r.resolve() for r in model_roots] if model_roots else None

    _db_path.parent.mkdir(parents=True, exist_ok=True)
    _output_root.mkdir(parents=True, exist_ok=True)

    job_queue = jobmod.JobQueue(_db_path, on_job=lambda j: None)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Start the worker on startup, stop it cleanly on shutdown."""
        job_queue.start()
        yield
        job_queue.stop()

    app = FastAPI(title="Weight Atlas", version="0.2.0", lifespan=lifespan)

    # Static files (CSS)
    static_dir = Path(__file__).resolve().parent.parent / "ui" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Output artefacts (PNG, TIFF, JSON)
    output_static_dir = base / "output"
    if output_static_dir.exists():
        app.mount("/static/outputs", StaticFiles(directory=str(output_static_dir)), name="outputs")

    # Templates
    templates_dir = Path(__file__).resolve().parent.parent / "ui" / "templates"
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory=str(templates_dir))

    router = create_router(job_queue, templates, _spec_path, _output_root, _model_roots)
    app.include_router(router)

    return app


# Module-level app so both `uvicorn weight_atlas.api.main:app` and the
# `weight-atlas serve` factory mode work. The background job worker is NOT
# started here — it starts inside the lifespan context manager — so importing
# this module has no thread side effects (safe under `--reload`).
app = create_app()
