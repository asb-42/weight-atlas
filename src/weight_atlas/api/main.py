"""FastAPI application factory for weight-atlas web UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from weight_atlas.api import jobs as jobmod
from weight_atlas.api.routes import create_router


def create_app(
    db_path: Path | None = None,
    spec_path: Path | None = None,
    output_root: Path | None = None,
) -> FastAPI:
    """Create the FastAPI application.

    Args:
        db_path: Path to SQLite job database. Defaults to ./data/jobs.db
        spec_path: Path to atlas spec JSON. Defaults to ./specs/atlas_spec.v1.json
        output_root: Root directory for scan outputs. Defaults to ./output
    """
    base = Path(__file__).resolve().parent.parent.parent.parent
    _db_path = db_path or base / "data" / "jobs.db"
    _spec_path = spec_path or base / "specs" / "atlas_spec.v1.json"
    _output_root = output_root or base / "output"

    _db_path.parent.mkdir(parents=True, exist_ok=True)
    _output_root.mkdir(parents=True, exist_ok=True)

    job_queue = jobmod.JobQueue(_db_path, on_job=lambda j: None)
    job_queue.start()

    app = FastAPI(title="Weight Atlas", version="0.1.0")

    # Static files (CSS)
    static_dir = Path(__file__).resolve().parent.parent / "ui" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Templates
    templates_dir = Path(__file__).resolve().parent.parent / "ui" / "templates"
    from fastapi.templating import Jinja2Templates
    templates = Jinja2Templates(directory=str(templates_dir))

    router = create_router(job_queue, templates, _spec_path, _output_root)
    app.include_router(router)

    @app.on_event("shutdown")
    def _shutdown() -> None:
        job_queue.stop()

    return app


app = create_app()
