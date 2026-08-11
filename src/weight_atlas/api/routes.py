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

        # Load spec from job spec_path, or fallback to default
        spec_path = Path(job.spec_path) if job.spec_path else None
        if spec_path is not None and spec_path.exists():
            spec = AtlasSpec.from_json(spec_path)
        else:
            # Use default spec
            default_spec_path = Path("specs/atlas_spec.v2.1.json")
            if default_spec_path.exists():
                spec = AtlasSpec.from_json(default_spec_path)
            else:
                raise HTTPException(status_code=500, detail="Default spec not found")

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

        # Compute out_dir relative to output_root FIRST (handle imported dirs outside root)
        try:
            out_dir_rel = str(out_dir.relative_to(output_root))
        except ValueError:
            out_dir_rel = str(out_dir)

        # Also check render/ subdirectory for rendered PNGs
        render_dir = out_dir / "render"
        if render_dir.exists():
            render_sheet = sorted(render_dir.glob("*_raw.png"))
            render_terrain = sorted(render_dir.glob("terrain_*.png"))
            sheet_pngs.extend(render_sheet)
            terrain_pngs.extend(render_terrain)
            if not tif_files:
                tif_files = sorted(render_dir.glob("field_*.tif"))

        # Create job_info dict with full path for image URLs
        job_info = job.to_dict()
        job_info['out_dir'] = str(out_dir)
        job_info['out_dir_rel'] = out_dir_rel

        return templates.TemplateResponse(
            request,
            "detail.html",
            {
                "job": job_info,
                "fingerprint": fingerprint,
                "spec": {
                    "spec_version": spec.spec_version,
                    "slots": spec.slots,
                    "channels": spec.channels,
                    "grid": spec.grid,
                    "sheet": spec.sheet,
                },
                "sheet_pngs": [f"render/{p.name}" for p in sheet_pngs],
                "terrain_pngs": [f"render/{p.name}" for p in terrain_pngs],
                "obj_meshes": [str(p.name) for p in obj_meshes],
                "tif_files": [str(p.name) for p in tif_files],
                "out_dir": out_dir_rel,
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

    @router.post("/api/jobs/{job_id}/rescan")
    async def rescan_job(job_id: str) -> JSONResponse:
        """Re-run the full scan pipeline for a job."""
        job = job_queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")

        out_dir = Path(job.out_dir)
        if not out_dir.exists():
            raise HTTPException(status_code=404, detail="output directory not found")

        from weight_atlas.core.types import AtlasSpec
        from weight_atlas.scan import scan as run_scan

        # Load spec from job spec_path, or fallback to default
        spec_path = Path(job.spec_path) if job.spec_path else None
        if spec_path is not None and spec_path.exists():
            spec = AtlasSpec.from_json(spec_path)
        else:
            spec = AtlasSpec.from_json(Path("specs/atlas_spec.v2.1.json"))

        # Re-run scan (overwrites existing artefacts)
        model_path = Path(job.model_path)
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"model path not found: {model_path}")

        artefacts = [str(a) for a in run_scan(model_path, out_dir, spec)]

        # Auto-render sheets after scan
        try:
            render_artefacts = job_queue._auto_render_sheets(out_dir, spec)
            artefacts.extend(render_artefacts)
        except Exception:
            pass  # Rendering is best-effort

        return JSONResponse({"status": "ok", "artefacts": artefacts})

    @router.post("/api/jobs/{job_id}/render/{renderer:path}")
    async def render_job(job_id: str, renderer: str) -> JSONResponse:
        """Trigger rendering for a job."""
        job = job_queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")

        out_dir = Path(job.out_dir)
        if not out_dir.exists():
            raise HTTPException(status_code=404, detail="output directory not found")

        from weight_atlas.core.registry import get_renderer
        from weight_atlas.core.types import AtlasSpec, Field2D
        from weight_atlas.fields.tif_io import read_tif
        from weight_atlas.fields.scaling import apply_scale

        spec = AtlasSpec.from_json(Path("specs/atlas_spec.v2.1.json"))
        renderer_cls = get_renderer(renderer)
        renderer_obj = renderer_cls()

        render_dir = out_dir / "render"
        render_dir.mkdir(exist_ok=True)

        # Discover channels
        channels: set[str] = set()
        for tif in out_dir.glob("field_*.tif"):
            core = tif.name[len("field_"):-len(".tif")]
            if core.endswith("_raw"):
                channels.add(core[:-len("_raw")])
            elif core.endswith("_smooth"):
                channels.add(core[:-len("_smooth")])

        produced: list[str] = []
        for channel in channels:
            smooth_path = out_dir / f"field_{channel}_smooth.tif"
            raw_path = out_dir / f"field_{channel}_raw.tif"
            tif = smooth_path if smooth_path.exists() else raw_path
            if not tif.exists():
                continue

            data = read_tif(tif)
            # Apply channel scaling on-the-fly (same as CLI render)
            ch_spec = spec.channels.get(channel, {})
            if "scale" in ch_spec:
                data = apply_scale(data, ch_spec["scale"])
            n_rows, n_cols = data.shape
            field = Field2D(
                channel=channel,
                data=data,
                row_labels=[str(i) for i in range(n_rows)],
                col_labels=list(spec.slots)[:n_cols] if n_cols <= len(spec.slots) else [str(i) for i in range(n_cols)],
                spec_version=spec.spec_version,
            )
            try:
                paths = renderer_obj.render(field, spec, render_dir)
                produced.extend(str(p.name) for p in paths)
            except Exception as e:
                produced.append(f"Error rendering {channel}: {e}")

        # Also render preview for each channel
        if renderer == "sheet":
            from weight_atlas.render.preview import PreviewRenderer
            preview_renderer = PreviewRenderer()
            for channel in channels:
                smooth_path = out_dir / f"field_{channel}_smooth.tif"
                raw_path = out_dir / f"field_{channel}_raw.tif"
                tif = smooth_path if smooth_path.exists() else raw_path
                if not tif.exists():
                    continue

                data = read_tif(tif)
                # Apply channel scaling on-the-fly (same as CLI render)
                ch_spec_prev = spec.channels.get(channel, {})
                if "scale" in ch_spec_prev:
                    data = apply_scale(data, ch_spec_prev["scale"])
                n_rows, n_cols = data.shape
                field = Field2D(
                    channel=channel,
                    data=data,
                    row_labels=[str(i) for i in range(n_rows)],
                    col_labels=list(spec.slots)[:n_cols] if n_cols <= len(spec.slots) else [str(i) for i in range(n_cols)],
                    spec_version=spec.spec_version,
                )
                try:
                    paths = preview_renderer.render(field, spec, render_dir)
                    produced.extend(str(p.name) for p in paths)
                except Exception as e:
                    produced.append(f"Error rendering preview {channel}: {e}")

        return JSONResponse({"status": "ok", "renderer": renderer, "produced": produced})

    @router.post("/api/import")
    async def import_scan(payload: dict[str, str]) -> JSONResponse:
        """Import an existing scan directory into the job database."""
        scan_dir_str = payload.get("scan_dir", "")
        model_path = payload.get("model_path", "")
        if not scan_dir_str:
            raise HTTPException(status_code=400, detail="scan_dir required")
        scan_dir = Path(scan_dir_str).resolve()
        if not scan_dir.exists():
            raise HTTPException(status_code=404, detail=f"scan_dir not found: {scan_dir}")
        if not (scan_dir / "fingerprint.json").exists():
            raise HTTPException(status_code=400, detail="Not a valid scan directory (missing fingerprint.json)")

        job = job_queue.import_scan(scan_dir, model_path)
        return JSONResponse(job.to_dict())

    # Allowlist of safe file extensions for serving
    _artefact_allowlist = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
                           ".obj", ".mtl", ".stl",
                           ".json", ".txt", ".csv", ".npy", ".tif", ".tiff"}

    @router.get("/api/artefacts/{job_id}/{path:path}")
    async def serve_artefact(job_id: str, path: str) -> Any:
        """Serve an artefact file from a job's output directory.

        Security:
        - Path traversal protection (resolved path must be within out_dir)
        - File extension allowlist (only safe types served)
        - Blocks symlinks that escape out_dir
        """
        from fastapi.responses import FileResponse

        job = job_queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")

        # Check extension allowlist
        artefact_path_lower = path.lower()
        if not any(artefact_path_lower.endswith(ext) for ext in _artefact_allowlist):
            raise HTTPException(
                status_code=403,
                detail=f"File type not allowed: {path}. Allowed: {_artefact_allowlist}"
            )

        # Security: ensure path doesn't escape out_dir
        out_dir = Path(job.out_dir).resolve()
        artefact_path = (out_dir / path).resolve()

        # Check that the resolved path is within out_dir (traversal protection)
        try:
            artefact_path.relative_to(out_dir)
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied: path traversal") from None

        if not artefact_path.exists():
            raise HTTPException(status_code=404, detail=f"Artefact not found: {path}")

        # Additional check: reject symlinks that escape out_dir
        if artefact_path.is_symlink():
            real_path = artefact_path.resolve()
            try:
                real_path.relative_to(out_dir)
            except ValueError:
                raise HTTPException(status_code=403, detail="Access denied: symlink escape") from None

        return FileResponse(artefact_path)

    @router.get("/models/{job_id}/artifacts/{artifact_name:path}")
    async def model_artifact(job_id: str, artifact_name: str) -> Any:
        """Serve a specific artifact file for a model (canonical route per v0.2.0 spec).

        Uses allowlist (.png/.obj inline, .tif as download link).
        Traversal protection via resolved path check.
        """
        job = job_queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")

        # Extension allowlist
        artifact_lower = artifact_name.lower()
        if not any(artifact_lower.endswith(ext) for ext in _artefact_allowlist):
            raise HTTPException(
                status_code=403,
                detail=f"File type not allowed: {artifact_name}"
            )

        out_dir = Path(job.out_dir).resolve()
        artifact_path = (out_dir / artifact_name).resolve()

        try:
            artifact_path.relative_to(out_dir)
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied: path traversal") from None

        if not artifact_path.exists():
            raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_name}")

        if artifact_path.is_symlink():
            real_path = artifact_path.resolve()
            try:
                real_path.relative_to(out_dir)
            except ValueError:
                raise HTTPException(status_code=403, detail="Access denied: symlink escape") from None

        from fastapi.responses import FileResponse
        return FileResponse(artifact_path)

    return router
