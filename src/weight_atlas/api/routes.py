"""HTTP routes for the weight-atlas web UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from weight_atlas.api.jobs import JobQueue, JobStatus
from weight_atlas.core.types import AtlasSpec


def create_router(
    job_queue: JobQueue,
    templates: Jinja2Templates,
    spec_path: Path,
    output_root: Path,
) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        """Model list page — scans output_root for completed jobs."""
        models = []
        for job in job_queue.list_jobs(limit=100):
            if job.status == JobStatus.DONE:
                models.append(
                    {
                        "job_id": job.job_id,
                        "model_path": job.model_path,
                        "out_dir": job.out_dir,
                        "artefacts": job.artefacts,
                        "updated_at": job.updated_at,
                    }
                )
        return templates.TemplateResponse(
            request,
            "models.html",
            {"models": models},
        )

    @router.post("/api/jobs")
    async def create_job(payload: dict[str, str]) -> JSONResponse:
        """Submit a new scan job."""
        model_path_str = payload.get("model_path", "")
        if not model_path_str:
            raise HTTPException(status_code=400, detail="model_path required")
        model_path = Path(model_path_str).resolve()
        if not model_path.exists():
            raise HTTPException(
                status_code=404, detail=f"model path not found: {model_path}"
            )

        out_dir = output_root / model_path.name
        out_dir.mkdir(parents=True, exist_ok=True)

        job = job_queue.submit(model_path, out_dir, spec_path)
        return JSONResponse(job.to_dict())

    @router.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> JSONResponse:
        """Get job status (for HTMX polling)."""
        job = job_queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return JSONResponse(job.to_dict())

    @router.get("/jobs/{job_id}", response_class=HTMLResponse)
    async def job_progress(request: Request, job_id: str) -> HTMLResponse:
        """Job progress page."""
        job = job_queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return templates.TemplateResponse(
            request,
            "job_progress.html",
            {"job": job.to_dict()},
        )

    @router.get("/models/{job_id}", response_class=HTMLResponse)
    async def model_detail(request: Request, job_id: str) -> HTMLResponse:
        """Model detail view: sheet, terrain, stats table, spec."""
        job = job_queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")

        out_dir = Path(job.out_dir)
        spec = AtlasSpec.from_json(Path(job.spec_path))

        # Load fingerprint for stats table
        fp_path = out_dir / "fingerprint.json"
        fingerprint: dict[str, Any] = {}
        if fp_path.exists():
            with open(fp_path) as f:
                fingerprint = json.load(f)

        # Discover artefacts by type
        sheet_pngs = sorted(out_dir.glob("*_raw.png"))
        terrain_pngs = sorted(out_dir.glob("terrain_*.png"))
        obj_meshes = sorted(out_dir.glob("*.obj"))
        tif_files = sorted(out_dir.glob("field_*.tif"))

        return templates.TemplateResponse(
            request,
            "detail.html",
            {
                "job": job.to_dict(),
                "fingerprint": fingerprint,
                "spec": {
                    "spec_version": spec.spec_version,
                    "slots": spec.slots,
                    "channels": spec.channels,
                    "grid": spec.grid,
                    "sheet": spec.sheet,
                },
                "sheet_pngs": [str(p.name) for p in sheet_pngs],
                "terrain_pngs": [str(p.name) for p in terrain_pngs],
                "obj_meshes": [str(p.name) for p in obj_meshes],
                "tif_files": [str(p.name) for p in tif_files],
                "out_dir": str(out_dir.relative_to(output_root)),
            },
        )

    @router.get("/api/models/{job_id}/fingerprint")
    async def model_fingerprint(job_id: str) -> JSONResponse:
        """Return fingerprint.json for a completed job."""
        job = job_queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status != JobStatus.DONE:
            raise HTTPException(status_code=409, detail="job not complete")
        fp_path = Path(job.out_dir) / "fingerprint.json"
        if not fp_path.exists():
            raise HTTPException(status_code=404, detail="fingerprint.json not found")
        with open(fp_path) as f:
            return JSONResponse(json.load(f))

    @router.get("/api/jobs/{job_id}/status")
    async def job_status_fragment(request: Request, job_id: str) -> HTMLResponse:
        """HTMX partial: job status badge + progress bar."""
        job = job_queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return templates.TemplateResponse(
            request,
            "_job_status.html",
            {"job": job.to_dict()},
        )

    # Compare endpoints

    @router.get("/compare", response_class=HTMLResponse)
    async def compare_page(request: Request) -> HTMLResponse:
        """Compare page: select two models to compare."""
        # List completed scan jobs as candidates
        candidates = []
        for job in job_queue.list_jobs(limit=100):
            if job.status == JobStatus.DONE and job.message != "compare":
                candidates.append(
                    {
                        "job_id": job.job_id,
                        "model_path": job.model_path,
                        "out_dir": job.out_dir,
                    }
                )
        return templates.TemplateResponse(
            request,
            "compare.html",
            {"candidates": candidates},
        )

    @router.post("/api/compare")
    async def create_compare_job(payload: dict[str, str]) -> JSONResponse:
        """Submit a new compare job."""
        dir_a_str = payload.get("dir_a", "")
        dir_b_str = payload.get("dir_b", "")
        if not dir_a_str or not dir_b_str:
            raise HTTPException(status_code=400, detail="dir_a and dir_b required")

        dir_a = Path(dir_a_str).resolve()
        dir_b = Path(dir_b_str).resolve()

        if not dir_a.exists():
            raise HTTPException(status_code=404, detail=f"dir_a not found: {dir_a}")
        if not dir_b.exists():
            raise HTTPException(status_code=404, detail=f"dir_b not found: {dir_b}")

        out_dir = output_root / f"compare_{dir_a.name}_vs_{dir_b.name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        job = job_queue.submit_compare(dir_a, dir_b, out_dir, spec_path)
        return JSONResponse(job.to_dict())

    @router.get("/compare/{job_id}", response_class=HTMLResponse)
    async def compare_report(request: Request, job_id: str) -> HTMLResponse:
        """Compare report page: show delta visualizations and summary metrics."""
        job = job_queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")

        out_dir = Path(job.out_dir)
        spec = AtlasSpec.from_json(Path(job.spec_path))

        # Load compare_summary.json
        summary_path = out_dir / "compare_summary.json"
        compare_summary: dict[str, Any] = {}
        if summary_path.exists():
            with open(summary_path) as f:
                compare_summary = json.load(f)

        # Discover delta PNGs
        delta_pngs = sorted(out_dir.glob("delta_*.png"))
        delta_render_dir = out_dir / "render"
        if delta_render_dir.exists():
            delta_pngs = sorted(delta_render_dir.glob("delta_*.png"))

        return templates.TemplateResponse(
            request,
            "compare_report.html",
            {
                "job": job.to_dict(),
                "spec": {
                    "spec_version": spec.spec_version,
                    "slots": spec.slots,
                    "channels": spec.channels,
                },
                "compare_summary": compare_summary,
                "delta_pngs": [str(p.name) for p in delta_pngs],
                "out_dir": str(out_dir.relative_to(output_root)),
            },
        )

    return router
