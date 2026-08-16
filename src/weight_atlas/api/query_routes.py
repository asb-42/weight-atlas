"""LLM query API routes (spec v0.2) — machine-readable read endpoints.

Mounted under ``/api``. Every response is deterministic JSON; errors use the
spec's ``{error: {code, type, message, hint}}`` envelope via :class:`QueryError`
(handled in ``main.py``).

``model_id`` is a completed scan job's ``job_id`` (stable identifier). A scan
is any DONE job whose out_dir contains ``fingerprint.json``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from weight_atlas.api import query as q
from weight_atlas.api.jobs import JobQueue, JobStatus


def create_query_router(job_queue: JobQueue) -> APIRouter:
    router = APIRouter()

    def _get_scan(model_id: str) -> tuple[Any, dict[str, Any]]:
        """Resolve model_id → (job, fingerprint); raise QueryError otherwise."""
        job = job_queue.get(model_id)
        if job is None:
            raise q.QueryError(
                404,
                "model_not_found",
                f"Model '{model_id}' not found.",
                "GET /api/models to list available scans",
            )
        if job.status != JobStatus.DONE:
            raise q.QueryError(
                409,
                "scan_not_complete",
                f"Scan '{model_id}' is not complete (status: {job.status.value}).",
                "Wait for the scan to finish, or check /api/jobs/{job_id}",
            )
        return job, q._load_fingerprint(job)

    @router.get("/api")
    async def discovery() -> dict[str, Any]:
        """Self-description — an agent's first call (in-band onboarding)."""
        return q.discovery_body()

    @router.get("/api/schema")
    async def schema() -> dict[str, Any]:
        """Machine-readable field-level schema for every response body."""
        return q.schema_body()

    @router.get("/api/models")
    async def models() -> dict[str, Any]:
        """Top-level listing of all completed scans."""
        return q.list_scans(job_queue.list_jobs(limit=1000))

    @router.get("/api/model/{model_id}")
    async def model_metadata(model_id: str) -> dict[str, Any]:
        job, fp = _get_scan(model_id)
        return q.scan_metadata(job, fp)

    @router.get("/api/model/{model_id}/summary")
    async def summary(
        model_id: str,
        group_by: str = "type",
        metrics: str | None = None,
    ) -> dict[str, Any]:
        """Model-wide aggregates, one call = whole model in context size."""
        job, fp = _get_scan(model_id)
        return q.summary_body(job, fp, group_by, _split_list(metrics))

    @router.get("/api/model/{model_id}/layer/{n}")
    async def layer(model_id: str, n: int) -> dict[str, Any]:
        job, fp = _get_scan(model_id)
        return q.layer_body(job, fp, n)

    @router.get("/api/model/{model_id}/anomalies")
    async def anomalies(
        model_id: str,
        metric: str = "spectral_norm",
        threshold: str = "p99",
        method: str = "quantile",
        n: int = 50,
        direction: str = "both",
        type_: str | None = Query(None, alias="type"),
        layer_range: str | None = None,
        fields: str | None = None,
    ) -> dict[str, Any]:
        """Statistically unusual tensors (p99 default)."""
        job, fp = _get_scan(model_id)
        return q.anomalies_body(
            job, fp, metric, threshold, method, n, direction,
            type_, layer_range, _split_list(fields),
        )

    @router.get("/api/model/{model_id}/query")
    async def query(
        model_id: str,
        layer: str | None = None,
        type_: str | None = Query(None, alias="type"),
        metric: str | None = None,
        order: str = "desc",
        fields: str | None = None,
        limit: int = 50,
        offset: int = 0,
        min_: float | None = Query(None, alias="min"),
        max_: float | None = Query(None, alias="max"),
    ) -> dict[str, Any]:
        """Filtered, sorted, paginated tensor list (token-budget friendly)."""
        job, fp = _get_scan(model_id)
        return q.query_body(
            job, fp, layer, type_, metric, order, _split_list(fields),
            limit, offset, min_, max_,
        )

    @router.get("/api/model/{model_id}/compare")
    async def compare(
        model_id: str,
        a: str = Query(..., description="slice, e.g. layer:0 or layer:42.type:mlp.gate_proj"),
        b: str = Query(..., description="slice, e.g. layer:31"),
        metrics: str | None = None,
        fields: str | None = None,
    ) -> dict[str, Any]:
        """Two-slice in-model comparison."""
        job, fp = _get_scan(model_id)
        return q.compare_body(job, fp, a, b, _split_list(metrics), _split_list(fields))

    @router.get("/api/model/{model_id}/histogram")
    async def histogram(
        model_id: str,
        metric: str = "spectral_norm",
        bins: int = 10,
        log: bool = False,
        type_: str | None = Query(None, alias="type"),
        layer_range: str | None = None,
        density: bool = False,
    ) -> dict[str, Any]:
        """Distribution of a metric (CDF/mode/skew analysis)."""
        job, fp = _get_scan(model_id)
        return q.histogram_body(job, fp, metric, bins, log, type_, layer_range, density)

    @router.get("/api/model/{model_id}/tensor/{name}")
    async def tensor(model_id: str, name: str) -> dict[str, Any]:
        """Full detail for one tensor (name is URL-encoded; dots preserved)."""
        job, fp = _get_scan(model_id)
        return q.tensor_body(job, fp, name)

    @router.get("/api/model/{model_id}/delta")
    async def delta(
        model_id: str,
        with_model: str = Query(..., alias="with", description="other scan's model_id"),
        metric: str = "spectral_norm",
        n: int = 50,
        min_change_pct: float = 0.0,
        fields: str | None = None,
    ) -> dict[str, Any]:
        """Cross-scan comparison: weight-space tier when paired/edit artefacts
        exist, else statistic diff."""
        job_a, fp_a = _get_scan(model_id)
        job_b, fp_b = _get_scan(with_model)
        return q.delta_body(
            job_a, fp_a, job_b, fp_b, metric, n, min_change_pct,
            _split_list(fields), job_queue,
        )

    return router


def _split_list(value: str | None) -> list[str] | None:
    """Split a comma-separated param into a list; None stays None."""
    if value is None:
        return None
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items or None


# Re-export for main.py's exception handler registration.
QueryError = q.QueryError
