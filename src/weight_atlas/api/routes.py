"""HTTP routes for the weight-atlas web UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates

from weight_atlas.api.jobs import JobQueue, JobStatus
from weight_atlas.core.types import AtlasSpec, load_default_spec

# Model-file picker: safe suffixes shown in the browse dialog.
_BROWSE_MODEL_SUFFIXES = {".gguf", ".safetensors"}


def create_router(
    job_queue: JobQueue,
    templates: Jinja2Templates,
    spec_path: Path,
    output_root: Path,
    model_roots: list[Path] | None = None,
) -> APIRouter:
    router = APIRouter()

    async def _read_body(request: Request) -> dict[str, str]:
        """Read a JSON or form-encoded request body (HTMX forms send urlencoded)."""
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise HTTPException(status_code=400, detail="expected a JSON object")
            return {str(k): str(v) for k, v in payload.items()}
        form = await request.form()
        return {key: str(value) for key, value in form.items()}

    def _require_allowed(path: Path) -> None:
        """Reject paths outside the configured allowlist.

        When ``model_roots`` is None the check is a no-op (local-only,
        backward-compatible default). Operators exposing the UI beyond
        loopback should pass an allowlist of model/scan root directories.
        """
        if model_roots is None:
            return
        resolved = path.resolve()
        for root in model_roots:
            try:
                resolved.relative_to(root.resolve())
                return
            except ValueError:
                continue
        raise HTTPException(
            status_code=403,
            detail=f"path outside allowed roots: {path}",
        )

    def _browse_start() -> Path:
        """Default starting directory for the file picker."""
        if model_roots:
            return model_roots[0]
        return Path.home()

    def _dir_has_model(path: Path) -> bool:
        """True if a directory contains model files (HF-style model dir)."""
        try:
            return any(path.glob("*.gguf")) or any(path.glob("*.safetensors"))
        except OSError:
            return False

    def _browse_error(msg: str) -> HTMLResponse:
        return HTMLResponse(
            f'<p class="hint browse-error">{msg}</p>',
            status_code=200,
        )

    @router.get("/api/browse", response_class=HTMLResponse)
    async def browse_files(
        request: Request,
        path: str = "",
        mode: str = "model",
    ) -> HTMLResponse:
        """List a directory for the model file picker (HTMX fragment).

        ``mode="model"``: directories navigate; model files and model
        directories are selectable. ``mode="dir"``: any directory is
        selectable (used by the scan import dialog).
        """
        if mode not in ("model", "dir"):
            mode = "model"

        raw = Path(path) if path else _browse_start()
        try:
            current = raw.expanduser().resolve()
            _require_allowed(current)
        except (OSError, HTTPException) as exc:
            if isinstance(exc, HTTPException) and exc.status_code == 403:
                return _browse_error("path is outside the allowed model roots")
            return _browse_error("path could not be resolved")
        if not current.is_dir():
            return _browse_error(f"not a directory: {current}")

        dirs: list[dict[str, Any]] = []
        files: list[dict[str, Any]] = []
        try:
            entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            entries = []
        for p in entries:
            if p.name.startswith("."):
                continue
            try:
                if p.is_dir():
                    dirs.append(
                        {
                            "name": p.name,
                            "path": str(p),
                            "is_model_dir": _dir_has_model(p),
                        }
                    )
                elif p.suffix.lower() in _BROWSE_MODEL_SUFFIXES:
                    files.append({"name": p.name, "path": str(p)})
            except OSError:
                continue

        parent = str(current.parent) if current != current.parent else ""
        return templates.TemplateResponse(
            request,
            "_file_browser.html",
            {
                "current": str(current),
                "parent": parent,
                "dirs": dirs,
                "files": files,
                "mode": mode,
            },
        )

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
    async def create_job(request: Request) -> JSONResponse:
        """Submit a new scan job."""
        payload = await _read_body(request)
        model_path_str = payload.get("model_path", "")
        if not model_path_str:
            raise HTTPException(status_code=400, detail="model_path required")
        model_path = Path(model_path_str).resolve()
        if not model_path.exists():
            raise HTTPException(
                status_code=404, detail=f"model path not found: {model_path}"
            )
        _require_allowed(model_path)

        # Unique output dir: two different models with the same file name (in
        # different roots) would otherwise share one out_dir and overwrite
        # each other's scans — same convention as compare_* outputs.
        import uuid
        out_dir = output_root / f"{model_path.name}_{uuid.uuid4().hex[:8]}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Checkbox form field: checked sends "on"/"true", unchecked is absent.
        quant_probe_raw = payload.get("quant_probe", "")
        quant_probe = str(quant_probe_raw).lower() in ("on", "true", "1", "yes")

        job = job_queue.submit(model_path, out_dir, spec_path, quant_probe=quant_probe)
        # Keep the job JSON for the API, but have HTMX navigate the browser to
        # the job's live progress page.
        return JSONResponse(
            job.to_dict(),
            headers={"HX-Redirect": f"/jobs/{job.job_id}"},
        )

    @router.get("/jobs", response_class=HTMLResponse)
    async def jobs_list(request: Request) -> HTMLResponse:
        """List every job (scans, compares, renders) with status and progress."""
        return templates.TemplateResponse(
            request,
            "jobs.html",
            {"jobs": job_queue.list_jobs(limit=200)},
        )

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

    def _model_context(job: Any) -> dict[str, Any]:
        """Build the shared context for a model detail sub-page.

        Loads the spec + artefact discovery once per request. Fingerprint is
        loaded separately (it is large) so overview/sheets/terrain pages stay
        light.
        """
        out_dir = Path(job.out_dir)

        spec_path = Path(job.spec_path) if job.spec_path else None
        if spec_path is not None and spec_path.exists():
            spec = AtlasSpec.from_json(spec_path)
        else:
            try:
                spec = load_default_spec()
            except (OSError, RuntimeError) as exc:
                raise HTTPException(status_code=500, detail=f"Default spec not found: {exc}") from exc

        sheet_pngs = sorted(out_dir.glob("*_raw.png"))
        terrain_pngs = sorted(out_dir.glob("terrain_*.png"))
        embed_terrain_pngs = sorted(out_dir.glob("embedding_terrain*.png"))
        treemap_pngs = sorted(out_dir.glob("param_treemap*.png"))
        obj_meshes = sorted(out_dir.glob("*.obj"))
        tif_files = sorted(out_dir.glob("field_*.tif"))

        render_dir = out_dir / "render"
        if render_dir.exists():
            sheet_pngs.extend(sorted(render_dir.glob("*_raw.png")))
            terrain_pngs.extend(sorted(render_dir.glob("terrain_*.png")))
            embed_terrain_pngs.extend(sorted(render_dir.glob("embedding_terrain*.png")))
            treemap_pngs.extend(sorted(render_dir.glob("param_treemap*.png")))
            if not tif_files:
                tif_files = sorted(render_dir.glob("field_*.tif"))

        try:
            out_dir_rel = str(out_dir.relative_to(output_root))
        except ValueError:
            out_dir_rel = str(out_dir)

        job_info = job.to_dict()
        job_info["out_dir"] = str(out_dir)
        job_info["out_dir_rel"] = out_dir_rel

        return {
            "job": job_info,
            "spec": {
                "spec_version": spec.spec_version,
                "slots": spec.slots,
                "channels": spec.channels,
                "grid": spec.grid,
                "sheet": spec.sheet,
            },
            "sheet_pngs": [f"render/{p.name}" for p in sheet_pngs],
            "terrain_pngs": [f"render/{p.name}" for p in terrain_pngs],
            "embed_terrain_pngs": [f"render/{p.name}" for p in embed_terrain_pngs],
            "treemap_pngs": [f"render/{p.name}" for p in treemap_pngs],
            "obj_meshes": [str(p.name) for p in obj_meshes],
            "tif_files": [str(p.name) for p in tif_files],
            "out_dir": out_dir_rel,
        }

    def _load_fingerprint(job: Any) -> dict[str, Any]:
        fp_path = Path(job.out_dir) / "fingerprint.json"
        if fp_path.exists():
            try:
                with open(fp_path) as f:
                    return cast(dict[str, Any], json.load(f))
            except (OSError, ValueError):
                return {}
        return {}

    # ── Scatter + records tabs (alesha-pro P2) ────────────────────────────

    def _model_records(job: Any) -> list[dict[str, Any]]:
        from weight_atlas.api.query import model_records

        return model_records(job)

    def _records_tab_context(job: Any) -> dict[str, Any]:
        from weight_atlas.api.query import RECORD_BOARDS, extreme_records

        records = _model_records(job)
        # Sorted-name index → stats-table page (both lists are name-sorted),
        # so each record links straight to the tensor's table position.
        name_index = {r["tensor_name"]: i for i, r in enumerate(records)}

        def _page(name: str) -> int:
            return name_index.get(name, 0) // stats_per_page + 1

        def _fmt(v: Any) -> str:
            if v is None:
                return "—"
            f = float(v)
            if f == int(f) and abs(f) < 1e6:
                return str(int(f))
            return f"{f:.4g}"

        boards = []
        for metric, direction, label in RECORD_BOARDS:
            tops = extreme_records(records, metric, direction)
            if not tops:
                continue  # metric absent in this fingerprint (e.g. old scan / probe off)
            boards.append({
                "metric": metric,
                "direction": direction,
                "label": label,
                "tops": [
                    {
                        "name": t["tensor_name"],
                        "slot": t["slot"],
                        "value": _fmt(t.get(metric)),
                        "page": _page(t["tensor_name"]),
                    }
                    for t in tops
                ],
            })
        return {"record_boards": boards}

    def _records_viz_context(job: Any) -> dict[str, Any]:
        """Outlier-impact visual context for the records tab.

        Renders three deterministic server-side SVGs (no JS, no client deps):
        a channel-scale-dominance ranking (OCGQuant framing), a per-layer
        depth profile, and percentile strips that place each board record in
        its model-wide distribution.
        """
        from weight_atlas.api.query import (
            distribution_strip,
            extreme_records,
            layer_profile,
            outlier_impact,
        )

        records = _model_records(job)
        impact = outlier_impact(records, limit=10)
        profile = layer_profile(records, "row_amax_ratio")

        # Percentile strip per amax board leader (the two cards where
        # "what does this number mean" needs distribution context most).
        strips: dict[str, dict[str, Any]] = {}
        for metric in ("row_amax_ratio", "col_amax_ratio"):
            tops = extreme_records(records, metric, "max", limit=1)
            if tops:
                strip = distribution_strip(
                    records, metric, float(tops[0][metric])
                )
                if strip is not None:
                    strips[metric] = strip

        return {
            "impact": impact,
            "impact_svg": _impact_svg(impact),
            "profile_svg": _profile_svg(profile),
            "strips": strips,
            "strip_svgs": {
                m: _strip_svg(s) for m, s in strips.items()
            },
        }


    def _scatter_tab_context(job: Any, request: Request, x: str | None, y: str | None) -> dict[str, Any]:
        from weight_atlas.api.query import METRICS, scatter_points

        records = _model_records(job)

        # Only metrics with at least one finite value are selectable axes
        # (deterministic order = METRICS order).
        available = []
        for metric in METRICS:
            if any(r.get(metric) is not None for r in records):
                available.append(metric)
        if x not in available:
            x = "kurtosis" if "kurtosis" in available else available[0]
        if y not in available:
            y = "sqnr_int4_g128" if "sqnr_int4_g128" in available else available[-1]

        data = scatter_points(records, x, y)
        return {
            "scatter": data,
            "scatter_metrics": available,
            "scatter_x": x,
            "scatter_y": y,
            "svg": _scatter_svg(data),
        }

    def _scatter_svg(data: dict[str, Any]) -> str:
        """Deterministic server-side SVG scatter plot (no JS, no client deps).

        p1–p99 clamped axes from scatter_points; plot box 900×560; points are
        2.4px circles in slot-group colors; hover title = tensor name (native
        SVG <title> tooltip).
        """

        w, h = 900, 560
        ml, mt, mr, mb = 64, 14, 14, 46
        pw, ph = w - ml - mr, h - mt - mb
        x_axis, y_axis = data["x_axis"], data["y_axis"]

        slot_colors = {
            "attn": "#4e79a7", "mlp": "#f28e2b", "norm": "#9c755f",
            "embed": "#59a14f", "lm_head": "#76b7b2", "router": "#e15759",
            "expert": "#b07aa1", "ssm": "#edc948", "bdh": "#b07aa1",
            "vision": "#79706e", "other": "#bab0ac",
        }

        def slot_color(slot: str) -> str:
            for prefix in slot_colors:
                if slot.startswith(prefix):
                    return slot_colors[prefix]
            return slot_colors["other"]

        x_lo, x_hi = float(x_axis["lo"]), float(x_axis["hi"])
        y_lo, y_hi = float(y_axis["lo"]), float(y_axis["hi"])

        def tx(v: float) -> float:
            # v is ALREADY axis-transformed (log10 applied by scatter_points'
            # clamp — see query.scatter_points): never transform twice.
            return ml + (v - x_lo) / (x_hi - x_lo) * pw

        def ty(v: float) -> float:
            # v is ALREADY axis-transformed (log10 applied by scatter_points'
            # clamp — see query.scatter_points): never transform twice.
            return mt + (1.0 - (v - y_lo) / (y_hi - y_lo)) * ph

        parts = [
            f'<svg viewBox="0 0 {w} {h}" role="img" class="scatter-svg" '
            f'xmlns="http://www.w3.org/2000/svg">',
            f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="#141414" stroke="#444"/>',
        ]
        # gridlines at 5 positions with axis labels
        for i in range(5):
            f = i / 4
            gx = ml + f * pw
            gy = mt + f * ph
            parts.append(f'<line x1="{gx:.1f}" y1="{mt}" x2="{gx:.1f}" y2="{mt + ph}" stroke="#2a2a2a"/>')
            parts.append(f'<line x1="{ml}" y1="{gy:.1f}" x2="{ml + pw}" y2="{gy:.1f}" stroke="#2a2a2a"/>')
            xv = x_axis["lo"] + f * (x_axis["hi"] - x_axis["lo"])
            yv = y_axis["hi"] - f * (y_axis["hi"] - y_axis["lo"])
            if x_axis["log"]:
                xv = 10 ** xv
            if y_axis["log"]:
                yv = 10 ** yv
            xlbl = f"{xv:.2g}" if abs(xv) < 1e5 else f"{xv:.1e}"
            ylbl = f"{yv:.2g}" if abs(yv) < 1e5 else f"{yv:.1e}"
            parts.append(f'<text x="{gx:.1f}" y="{mt + ph + 16}" text-anchor="middle" font-size="10" fill="#888">{xlbl}</text>')
            parts.append(f'<text x="{ml - 6}" y="{gy + 3:.1f}" text-anchor="end" font-size="10" fill="#888">{ylbl}</text>')
        parts.append(
            f'<text x="{ml + pw / 2:.0f}" y="{h - 8}" text-anchor="middle" font-size="11" fill="#aaa">'
            f'{x_axis["metric"]}{" (log)" if x_axis["log"] else ""}</text>'
        )
        parts.append(
            f'<text x="14" y="{mt + ph / 2:.0f}" text-anchor="middle" font-size="11" fill="#aaa" '
            f'transform="rotate(-90 14 {mt + ph / 2:.0f})">'
            f'{y_axis["metric"]}{" (log)" if y_axis["log"] else ""}</text>'
        )
        for p in data["points"]:
            cx, cy = tx(p["x"]), ty(p["y"])
            name = p["tensor_name"]
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.4" fill="{slot_color(p["slot"])}" '
                f'fill-opacity="0.72"><title>{name} · {p["slot"]} · layer {p["layer"]} · '
                f'{p["numel"]:,} params</title></circle>'
            )
        parts.append("</svg>")
        return "".join(parts)

    model_tabs = ("overview", "sheets", "terrain", "stats", "scatter", "records", "spec")
    stats_per_page = 200

    def _render_model_tab(
        request: Request,
        job: Any,
        tab: str,
        ctx: dict[str, Any],
        page: int = 1,
        x: str | None = None,
        y: str | None = None,
        q: str | None = None,
    ) -> str:
        """Render a model sub-page body (the tab content below the tab bar)."""
        ctx = dict(ctx)
        ctx["active_tab"] = tab
        if tab in ("overview", "stats", "spec"):
            ctx["fingerprint"] = _load_fingerprint(job)
        if tab == "stats":
            tensors = ctx.get("fingerprint", {}).get("tensors", {})
            items = list(tensors.items())
            # name substring filter (case-insensitive) — the only way to
            # find e.g. n-gram records among 74k rows without paging
            q = (q or "").strip()
            if q:
                ql = q.lower()
                items = [(n, s) for n, s in items if ql in n.lower()]
            total = len(items)
            per = stats_per_page
            pages = max(1, -(-total // per))
            page = max(1, min(page, pages))
            start = (page - 1) * per
            ctx.update({
                "stats_rows": items[start:start + per],
                "stats_page": page,
                "stats_pages": pages,
                "stats_total": total,
                "stats_per_page": per,
                "stats_q": q,
            })
        elif tab == "scatter":
            ctx.update(_scatter_tab_context(job, request, x, y))
        elif tab == "records":
            ctx.update(_records_tab_context(job))
            ctx["viz"] = _records_viz_context(job)
        template = templates.env.get_template(f"_model_{tab}.html")
        return template.render(**ctx)

    @router.get("/models/{job_id}", response_class=HTMLResponse)
    async def model_detail(
        request: Request,
        job_id: str,
        tab: str = "overview",
        page: int = 1,
        x: str | None = None,
        y: str | None = None,
        q: str | None = None,
    ) -> HTMLResponse:
        """Model detail view: tabbed sub-pages (overview/sheets/terrain/stats/
        scatter/records/spec).

        The stats table is server-paginated and rendered only on its own
        sub-page, so the overview page stays small even for 74k-tensor models.
        The scatter tab takes ``x``/``y`` metric params (validated server-side,
        falls back to sensible defaults when absent).
        """
        job = job_queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if tab not in model_tabs:
            tab = "overview"
        ctx = _model_context(job)
        ctx["tab_content"] = _render_model_tab(request, job, tab, ctx, page=page, x=x, y=y, q=q)
        ctx["active_tab"] = tab
        # htmx tab switches (hx-get) swap into #model-tab-content: serve the
        # bare fragment, never a full page (a full page would duplicate
        # header/nav/footer inside the tab div on every swap).
        if request.headers.get("HX-Request") == "true":
            return HTMLResponse(ctx["tab_content"])
        return templates.TemplateResponse(
            request,
            "detail.html",
            ctx,
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

    @router.get("/healthz")
    async def healthz() -> JSONResponse:
        """Liveness + database reachability for reverse proxies/monitors.

        200 ``{"status": "ok"}`` when the job store answers; 503 with
        ``{"status": "degraded", ...}`` otherwise. No auth, no secrets,
        safe to poll every few seconds.
        """
        try:
            job_queue.list_jobs(limit=1)
        except Exception as exc:  # noqa: BLE001 - report, don't leak
            return JSONResponse(
                {"status": "degraded", "database": "unreachable",
                 "error": f"{type(exc).__name__}"},
                status_code=503,
            )
        return JSONResponse({"status": "ok", "database": "ok"})

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
        # List completed scan jobs as candidates. Only scan outputs carry the
        # manifest.json + field_*.tif artefacts a comparison consumes; render
        # jobs may point their out_dir at a compare output directory (delta
        # renders), which has no manifest and would fail the comparison.
        candidates = []
        for job in job_queue.list_jobs(limit=100):
            if job.status == JobStatus.DONE and job.job_type == "scan":
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
    async def create_compare_job(request: Request) -> JSONResponse:
        """Submit a new compare job."""
        payload = await _read_body(request)
        dir_a_str = payload.get("dir_a", "")
        dir_b_str = payload.get("dir_b", "")
        if not dir_a_str or not dir_b_str:
            raise HTTPException(status_code=400, detail="dir_a and dir_b required")

        mode = payload.get("mode", "strict")
        if mode not in ("strict", "aligned"):
            raise HTTPException(status_code=400, detail=f"unknown compare mode: {mode}")

        interp = payload.get("interp", "linear")
        if interp not in ("linear", "nearest"):
            raise HTTPException(status_code=400, detail=f"unknown interp: {interp}")

        dir_a = Path(dir_a_str).resolve()
        dir_b = Path(dir_b_str).resolve()

        if not dir_a.exists():
            raise HTTPException(status_code=404, detail=f"dir_a not found: {dir_a}")
        if not dir_b.exists():
            raise HTTPException(status_code=404, detail=f"dir_b not found: {dir_b}")
        _require_allowed(dir_a)
        _require_allowed(dir_b)

        # A comparison consumes scan artefacts (manifest.json + field_*.tif).
        # Compare/render output dirs (delta sheets, etc.) have no manifest and
        # would fail inside the worker; reject them up front with a clear 400.
        for label, d in (("dir_a", dir_a), ("dir_b", dir_b)):
            if not (d / "manifest.json").exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"{label} is not a scan output directory (no manifest.json): {d}",
                )

        # Unique output dir: same-named models in different roots would otherwise
        # overwrite each other's compare results.
        import uuid
        out_dir = output_root / f"compare_{dir_a.name}_vs_{dir_b.name}_{uuid.uuid4().hex[:8]}"
        out_dir.mkdir(parents=True, exist_ok=True)

        job = job_queue.submit_compare(dir_a, dir_b, out_dir, spec_path, mode=mode, interp=interp)
        # Keep the job JSON for the API; HTMX navigates to the compare progress page.
        return JSONResponse(
            job.to_dict(),
            headers={"HX-Redirect": f"/jobs/{job.job_id}"},
        )

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

        # out_dir may live outside output_root (imported scans) — fall back to the
        # absolute path rather than raising ValueError.
        try:
            out_dir_rel = str(out_dir.relative_to(output_root))
        except ValueError:
            out_dir_rel = str(out_dir)

        # Compare job model_path is "dir_a|dir_b"; the dir names are the best
        # human-readable labels (the fingerprint carries no display name).
        model_parts = str(job.model_path).split("|")
        model_a_name = Path(model_parts[0]).name if len(model_parts) > 0 and model_parts[0] else ""
        model_b_name = Path(model_parts[1]).name if len(model_parts) > 1 and model_parts[1] else ""

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
                "out_dir": out_dir_rel,
                "model_a_name": model_a_name,
                "model_b_name": model_b_name,
            },
        )

    @router.post("/api/jobs/{job_id}/rescan")
    async def rescan_job(job_id: str) -> Response:
        """Re-run the full scan pipeline for a job.

        Enqueues a rescan job on the worker thread instead of running the
        (potentially very slow) scan inline on the event loop, so other
        requests keep being served. The HTMX client is redirected to the new
        job's live progress page.
        """
        job = job_queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")

        out_dir = Path(job.out_dir)
        if not out_dir.exists():
            raise HTTPException(status_code=404, detail="output directory not found")

        # Validate the model path is actually scannable before enqueuing. An
        # imported scan's model_path often points at the artefacts directory
        # (not a model), in which case the re-scan would silently fail in the
        # worker and leave a stale fingerprint on disk.
        from weight_atlas.core.types import detect_loader
        model_path = Path(job.model_path)
        try:
            detect_loader(model_path)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"cannot re-scan: {exc}. This job was imported and its model "
                    "path is not a scannable model. Re-import the scan with the "
                    "original model path, or submit a new scan."
                ),
            ) from exc

        new_job = job_queue.submit_rescan(job_id)
        return Response(
            status_code=202,
            headers={"HX-Redirect": f"/jobs/{new_job.job_id}"},
            content="",
        )

    @router.post("/api/jobs/{job_id}/render/{renderer:path}")
    async def render_job(job_id: str, renderer: str, request: Request) -> Response:
        """Trigger rendering for a job (enqueued on the worker thread).

        Redirects the HTMX client to the new job's live progress page.
        Optional sheet overrides (``normalized_depth``, ``drop_empty_cols``)
        come from the UI's checkboxes and are applied on top of the recorded
        spec for this render only.
        """
        job = job_queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")

        out_dir = Path(job.out_dir)
        if not out_dir.exists():
            raise HTTPException(status_code=404, detail="output directory not found")

        from weight_atlas.core.registry import get_renderer
        try:
            get_renderer(renderer)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown renderer: {renderer}") from None

        form = await request.form()
        sheet_knobs: dict[str, str | bool] = {}
        if renderer == "sheet" or renderer == "preview":
            # Only the raster sheet renderers consume these knobs.
            for key in ("normalized_depth", "drop_empty_cols"):
                if key in form:
                    sheet_knobs[key] = True
        elif renderer == "fractal":
            # Fractal-mode toggle (fbm/sdf) overlays onto the recorded spec
            # for this render only.
            fractal_mode = form.get("fractal_mode")
            if fractal_mode in ("fbm", "sdf"):
                sheet_knobs["fractal_mode"] = fractal_mode
        elif renderer == "embedding_terrain":
            # Camera/light/relief knobs for the UMAP landscape render.
            # Values are clamped inside the renderer; unknown keys dropped.
            for key in ("pitch", "yaw", "dist_factor", "lens", "z_scale",
                        "gamma", "sun_alt", "sun_azi", "resolution",
                        "samples"):
                raw = form.get(key)
                if isinstance(raw, str) and raw.strip():
                    sheet_knobs[key] = raw.strip()
            sheet_knobs["labels"] = form.get("labels") is not None
            sheet_knobs["log_height"] = form.get("log_height") is not None

        new_job = job_queue.submit_render(job_id, renderer, sheet_knobs=sheet_knobs)
        return Response(
            status_code=202,
            headers={"HX-Redirect": f"/jobs/{new_job.job_id}"},
            content="",
        )

    @router.post("/api/jobs/{job_id}/model-path")
    async def set_model_path(job_id: str, request: Request) -> Response:
        """Set/update the model path recorded on a job (enables re-scan of imports).

        Imported scans often have ``model_path`` pointing at the artefacts
        directory; this lets the user correct it so "Re-scan Model" can run.
        """
        job = job_queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")

        payload = await _read_body(request)
        model_path_str = payload.get("model_path", "").strip()
        if not model_path_str:
            raise HTTPException(status_code=400, detail="model_path required")

        model_path = Path(model_path_str).resolve()
        from weight_atlas.core.types import detect_loader
        try:
            detect_loader(model_path)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"not a scannable model: {exc}",
            ) from exc
        _require_allowed(model_path)

        job_queue.update_model_path(job_id, str(model_path))
        from html import escape
        return Response(
            status_code=200,
            media_type="text/html",
            content=(
                f'<span class="hint">Model path updated to <code>{escape(str(model_path))}</code>. '
                "You can now Re-scan.</span>"
            ),
        )

    @router.post("/api/import")
    async def import_scan(request: Request) -> JSONResponse:
        """Import an existing scan directory into the job database."""
        from starlette.concurrency import run_in_threadpool

        payload = await _read_body(request)
        scan_dir_str = payload.get("scan_dir", "")
        model_path = payload.get("model_path", "")
        if not scan_dir_str:
            raise HTTPException(status_code=400, detail="scan_dir required")
        scan_dir = Path(scan_dir_str).resolve()
        if not scan_dir.exists():
            raise HTTPException(status_code=404, detail=f"scan_dir not found: {scan_dir}")
        _require_allowed(scan_dir)
        if not (scan_dir / "fingerprint.json").exists():
            raise HTTPException(status_code=400, detail="Not a valid scan directory (missing fingerprint.json)")

        # The import renders sheets synchronously (matplotlib, potentially
        # minutes) — keep that CPU-bound work off the event loop so UI
        # polling and other requests stay responsive.
        job = await run_in_threadpool(job_queue.import_scan, scan_dir, model_path)
        # Import is immediate (job is already done) → go straight to the detail page.
        return JSONResponse(
            job.to_dict(),
            headers={"HX-Redirect": f"/models/{job.job_id}"},
        )

    @router.post("/api/packages/prepare")
    async def prepare_package(request: Request) -> JSONResponse:
        """Export a registered scan as a .wasc share package (Phase 0).

        ``job_id`` selects the scan; ``profile`` is stats|full. Writes
        ``<output_root>/packages/<job_id>.wasc`` deterministically and
        returns its path — the UI shows it for manual sharing. LAN-local
        only; a public download surface is Phase 1 (NOT IMPLEMENTED).
        """
        from starlette.concurrency import run_in_threadpool

        from weight_atlas.sharing.package import PackageError, export_package

        payload = await _read_body(request)
        job_id = str(payload.get("job_id", ""))
        profile = str(payload.get("profile", "stats"))
        if profile not in ("stats", "full"):
            raise HTTPException(status_code=400, detail="profile must be stats|full")
        job = job_queue.get(job_id)
        if job is None or job.status != JobStatus.DONE:
            raise HTTPException(status_code=404, detail="job not found or not done")
        scan_dir = Path(job.out_dir)
        if not (scan_dir / "fingerprint.json").exists():
            raise HTTPException(status_code=404, detail="scan has no fingerprint")

        packages_root = output_root / "packages"
        out_path = packages_root / f"{job_id}.wasc"

        def _export() -> Path:
            return export_package(
                scan_dir, out_path, profile=profile,
                model_name=str(payload.get("model_name", "") or scan_dir.name),
            )

        try:
            pkg = await run_in_threadpool(_export)
        except PackageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(
            {
                "package": str(pkg),
                "bytes": pkg.stat().st_size,
                "profile": profile,
                "note": (
                    "Package written to disk for manual sharing. Public "
                    "upload/download is NOT YET IMPLEMENTED (Phase 1)."
                ),
            }
        )

    @router.post("/api/packages")
    async def upload_package(request: Request) -> JSONResponse:
        """Ingest a .wasc scan package: verify, extract, register.

        Phase 0 scan sharing (docs/2026-09-02_scan-sharing-phase0.md).
        This endpoint is LAN-local by design: it accepts a package PATH on
        this machine (the browser picker uses /api/browse) — NOT a public
        upload surface. Public upload (Phase 1: auth, quotas, archive-bomb
        limits, untrusted-archive hardening) is explicitly NOT IMPLEMENTED;
        a public deployment must not expose this endpoint.
        """
        from starlette.concurrency import run_in_threadpool

        from weight_atlas.sharing.package import PackageError, import_package

        payload = await _read_body(request)
        pkg_str = payload.get("package", "")
        if not pkg_str:
            raise HTTPException(status_code=400, detail="package path required")
        import uuid

        pkg_path = Path(pkg_str).resolve()
        if not pkg_path.exists():
            raise HTTPException(status_code=404, detail=f"package not found: {pkg_path}")
        _require_allowed(pkg_path)
        dest = output_root / f"pkg_{pkg_path.stem}_{uuid.uuid4().hex[:8]}"

        def _extract() -> Path:
            return import_package(pkg_path, dest)

        try:
            scan_dir = await run_in_threadpool(_extract)
        except PackageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        model_name = str(payload.get("model_name", "") or pkg_path.stem)
        job = await run_in_threadpool(job_queue.import_scan, scan_dir, model_name)
        return JSONResponse(
            job.to_dict(),
            headers={"HX-Redirect": f"/models/{job.job_id}"},
        )

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

        # Security: ensure path doesn't escape out_dir. resolve() already
        # canonicalizes symlinks, so a single containment check on the
        # resolved path covers both traversal and symlink escapes.
        out_dir = Path(job.out_dir).resolve()
        artefact_path = (out_dir / path).resolve()

        # Check that the resolved path is within out_dir (traversal protection)
        try:
            artefact_path.relative_to(out_dir)
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied: path traversal") from None

        if not artefact_path.exists():
            raise HTTPException(status_code=404, detail=f"Artefact not found: {path}")

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

        # Resolved containment check covers traversal AND symlink escapes.
        try:
            artifact_path.relative_to(out_dir)
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied: path traversal") from None

        if not artifact_path.exists():
            raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_name}")

        from fastapi.responses import FileResponse
        return FileResponse(artifact_path)

    return router


# ───────────────────────────────────────────────────────────────────────────────────
# Records-tab SVG renderers (deterministic, server-side, no JS).
#
# Framing follows the outlier-channel literature: a channel whose magnitude
# dominates its row/column sets the per-channel quantization scale, so every
# companion channel in the same block inherits that scale as collateral
# quantization error (cf. OCGQuant, arXiv:2609.00066). row_amax_ratio /
# col_amax_ratio in the fingerprint measure exactly this dominance.
# ───────────────────────────────────────────────────────────────────────────────────

_IMPACT_BAR_H = 20
_IMPACT_ROW_H = 26


def _fmt_metric(v: float) -> str:
    """Compact deterministic metric formatting for SVG labels."""
    if v == int(v) and abs(v) < 1e6:
        return str(int(v))
    if abs(v) >= 1e5 or (0 < abs(v) < 1e-3):
        return f"{v:.2e}"
    return f"{v:.4g}"


def _impact_svg(impact: list[dict[str, Any]]) -> str:
    """Horizontal bar chart: channel-scale dominance per tensor.

    One bar per tensor, length = dominance ratio (log scale from 1.0),
    color = slot group, label = ratio; hover title carries row/col split.
    """
    if not impact:
        return ""
    slot_colors = {
        "attn": "#4e79a7", "mlp": "#f28e2b", "norm": "#9c755f",
        "embed": "#59a14f", "lm_head": "#76b7b2", "router": "#e15759",
        "expert": "#b07aa1", "ssm": "#edc948", "bdh": "#b07aa1",
        "vision": "#79706e", "other": "#bab0ac",
    }

    def slot_color(slot: str) -> str:
        for prefix, color in slot_colors.items():
            if slot.startswith(prefix):
                return color
        return slot_colors["other"]

    name_w = 300
    bar_x = name_w + 12
    val_w = 74
    w = bar_x + val_w + 24
    h = len(impact) * _IMPACT_ROW_H + 30
    lo = min(t["dominance"] for t in impact)
    hi = max(t["dominance"] for t in impact)
    lo = min(lo, 1.0)  # ratio scale starts at 1 (max == median)
    log_span = (np.log10(max(hi, lo * 10)) - np.log10(lo)) or 1.0

    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" class="impact-svg" '
        f'xmlns="http://www.w3.org/2000/svg">',
        f'<text x="{bar_x}" y="12" font-size="10" fill="#888">'
        f'max-channel / median-channel ratio (log)</text>',
    ]
    for i, t in enumerate(impact):
        y = 24 + i * _IMPACT_ROW_H
        frac = (np.log10(max(t["dominance"], lo)) - np.log10(lo)) / log_span
        bw = max(4.0, frac * val_w) if hi > lo else val_w * 0.92
        color = slot_color(t["slot"])
        layer_lbl = f"L{t['layer']} · " if t["layer"] >= 0 else ""
        title = (
            f"{t['tensor_name']} · {t['slot']}"
            f" · row {t['row']:.4g} / col {t['col']:.4g}"
            if t["row"] is not None and t["col"] is not None
            else t["tensor_name"]
        )
        parts.extend(
            [
                f'<text x="0" y="{y + 13}" font-size="10" fill="#aaa" '
                f'font-family="monospace">{layer_lbl}{t["tensor_name"][:44]}</text>',
                f'<rect x="{bar_x}" y="{y}" width="{val_w}" height="{_IMPACT_BAR_H}" '
                f'fill="#1d1d1d" stroke="#333" stroke-width="0.5">'
                f'<title>{title}</title></rect>',
                f'<rect x="{bar_x}" y="{y}" width="{bw:.1f}" height="{_IMPACT_BAR_H}" '
                f'fill="{color}" fill-opacity="0.85"><title>{title}</title></rect>',
                f'<text x="{bar_x + val_w + 6}" y="{y + 13}" font-size="10" '
                f'fill="#ddd" font-family="monospace">{_fmt_metric(t["dominance"])}</text>',
            ]
        )
    parts.append("</svg>")
    return "".join(parts)


def _profile_svg(profile: dict[str, Any]) -> str:
    """Line chart: per-layer max row_amax_ratio (outlier depth structure)."""
    layers = profile.get("layers") or []
    if len(layers) < 2:
        return ""
    w, h = 900, 200
    ml, mt, mr, mb = 56, 12, 16, 34
    pw, ph = w - ml - mr, h - mt - mb
    ys = [entry["max"] for entry in layers]
    lo, hi = min(ys), max(ys)
    if hi <= lo * 1.0001:
        hi = lo * 1.5 + 1e-9
    log = lo > 0 and hi / lo >= 100.0

    def tf(v: float) -> float:
        t = float(np.log10(v)) if log else v
        tlo = float(np.log10(lo)) if log else lo
        thi = float(np.log10(hi)) if log else hi
        return mt + (1.0 - (t - tlo) / (thi - tlo)) * ph

    n = len(layers)
    pts = [
        (ml + (i / (n - 1)) * pw, tf(entry["max"]))
        for i, entry in enumerate(layers)
    ]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = (
        f"M {ml},{mt + ph} "
        + " ".join(f"L {x:.1f},{y:.1f}" for x, y in pts)
        + f" L {ml + pw},{mt + ph} Z"
    )
    worst_i = max(range(n), key=lambda i: layers[i]["max"])
    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" class="profile-svg" '
        f'xmlns="http://www.w3.org/2000/svg">',
        f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="#141414" stroke="#444"/>',
        f'<path d="{area}" fill="#4e79a7" fill-opacity="0.12"/>',
        f'<polyline points="{line}" fill="none" stroke="#4e79a7" stroke-width="1.6"/>',
    ]
    for i in range(5):
        f = i / 4
        gy = mt + f * ph
        parts.append(f'<line x1="{ml}" y1="{gy:.1f}" x2="{ml + pw}" y2="{gy:.1f}" stroke="#2a2a2a"/>')
    for i, entry in enumerate(layers):
        x, y = pts[i]
        title = f"layer {entry['layer']} · max {entry['max']:.4g} · {entry['worst_tensor']}"
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="#4e79a7" '
            f'fill-opacity="0.85"><title>{title}</title></circle>'
        )
    # annotate the worst layer
    wx, wy = pts[worst_i]
    parts.append(
        f'<circle cx="{wx:.1f}" cy="{wy:.1f}" r="4.6" fill="none" stroke="#e15759" stroke-width="1.4">'
        f'<title>peak: layer {layers[worst_i]["layer"]} · {layers[worst_i]["max"]:.4g}</title></circle>'
    )
    for i in (0, n // 2, n - 1):
        parts.append(
            f'<text x="{pts[i][0]:.1f}" y="{h - 16}" text-anchor="middle" font-size="10" '
            f'fill="#888">layer {layers[i]["layer"]}</text>'
        )
    parts.append(
        f'<text x="{ml - 6}" y="{mt + 10}" text-anchor="end" font-size="10" fill="#888">'
        f'{_fmt_metric(hi)}</text>'
    )
    parts.append(
        f'<text x="{ml + pw / 2:.0f}" y="{h - 4}" text-anchor="middle" font-size="11" fill="#aaa">'
        f'max row_amax_ratio per layer{" (log)" if log else ""}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _strip_svg(strip: dict[str, Any]) -> str:
    """Percentile strip: where the board leader sits in the metric's model-wide
    distribution (p5–p95 box, min–max whiskers, leader marker).
    """
    w, h = 320, 58
    ml, mr = 10, 10
    pw = w - ml - mr
    y = 26
    lo, hi = strip["min"], strip["max"]
    log = strip["log"]

    def tf(v: float) -> float:
        t = float(np.log10(v)) if log else v
        tlo = float(np.log10(lo)) if log else lo
        thi = float(np.log10(hi)) if log else hi
        if thi - tlo < 1e-12:
            thi = tlo + 1.0
        return ml + (t - tlo) / (thi - tlo) * pw

    def tick(v: float) -> str:
        return _fmt_metric(v)

    q = {k: tf(strip[k]) for k in ("p5", "p25", "p50", "p75", "p95")}
    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" class="strip-svg" '
        f'xmlns="http://www.w3.org/2000/svg">',
        f'<line x1="{tf(lo):.1f}" y1="{y}" x2="{q["p5"]:.1f}" y2="{y}" stroke="#555"/>',
        f'<line x1="{q["p95"]:.1f}" y1="{y}" x2="{tf(hi):.1f}" y2="{y}" stroke="#555"/>',
        f'<rect x="{q["p25"]:.1f}" y="{y - 8}" width="{q["p75"] - q["p25"]:.1f}" height="16" '
        f'fill="#4e79a7" fill-opacity="0.35" stroke="#4e79a7"/>',
        f'<line x1="{q["p50"]:.1f}" y1="{y - 8}" x2="{q["p50"]:.1f}" y2="{y + 8}" stroke="#9c755f" stroke-width="1.6"/>',
        # leader marker
        f'<line x1="{tf(strip["value"]):.1f}" y1="{y - 15}" x2="{tf(strip["value"]):.1f}" y2="{y + 15}" '
        f'stroke="#e15759" stroke-width="2"/><title>leader: {_fmt_metric(strip["value"])} '
        f'({strip["percentile"]:.1f}% of {strip["n"]} tensors)</title>',
        f'<text x="{ml}" y="{h - 6}" font-size="9" fill="#888">{tick(lo)}</text>',
        f'<text x="{w - mr}" y="{h - 6}" text-anchor="end" font-size="9" fill="#888">{tick(hi)}</text>',
        f'<text x="{w / 2:.0f}" y="{h - 6}" text-anchor="middle" font-size="9" fill="#bbb">'
        f'p99 {_fmt_metric(strip["p99"])} · leader at p{strip["percentile"]:.1f}</text>',
    ]
    parts.append("</svg>")
    return "".join(parts)
