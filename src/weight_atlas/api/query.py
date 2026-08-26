"""Read-side query engine for the LLM query API (spec v0.2).

Pure functions over a scan's ``fingerprint.json``: tensor records with
layer/slot/type derivation, global + per-type baselines, filtered/sorted
pagination, anomalies, layer views, slice comparison, histograms, tensor
drilldown, and cross-scan deltas. All responses are deterministic: fixed
ordering, no timestamps in analytical output, floats rounded to 4 decimals.

``model_id`` in the API == ``job_id`` in the job database (the stable
identifier the web UI already uses); a scan is any DONE job whose output
directory contains ``fingerprint.json``.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from weight_atlas.api.jobs import Job, JobQueue, JobStatus
from weight_atlas.core.name_map import extract_expert_id, get_moe_slot, map_name

# Core fingerprint metrics (order = canonical column order everywhere).
METRICS = (
    "frobenius",
    "spectral_norm",
    "effective_rank",
    "stable_rank",
    "kurtosis",
    "sparsity",
    "kernel_norm",
)

METRIC_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "frobenius": {
        "description": "Frobenius (L2) norm of the weight matrix",
        "min": 0,
        "expected_range": [0, 500],
    },
    "spectral_norm": {
        "description": "largest singular value (spectral / operator norm)",
        "min": 0,
        "expected_range": [2, 25],
    },
    "effective_rank": {
        "description": "spectral-entropy effective rank (exp of Shannon entropy of the singular-value distribution)",
        "min": 1,
        "expected_range": [1, 64],
    },
    "stable_rank": {
        "description": "log1p((frobenius/spectral_norm)^2) — the tint-channel stat; 1 = rank-1-like spectrum",
        "min": 0,
        "expected_range": [0.5, 4],
    },
    "kurtosis": {
        "description": "excess kurtosis of the weight distribution",
        "min": 0,
        "expected_range": [0, 6],
    },
    "sparsity": {
        "description": "fraction of weights near zero (below threshold)",
        "min": 0,
        "expected_range": [0, 1],
    },
    "kernel_norm": {
        "description": "mean per-output-channel L2 norm of a conv kernel; Frobenius for non-4-D tensors",
        "min": 0,
        "expected_range": [0, 500],
    },
}

# Slot → human-readable HF-style type label. The slot is the authoritative
# grouping (matches raster columns); ``type`` is a derived display label.
_SLOT_TYPE: dict[str, str] = {
    "embed": "embed_tokens",
    "lm_head": "lm_head",
    "norm_attn": "input_layernorm",
    "norm_mlp": "post_attention_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_o": "self_attn.o_proj",
    "mlp_gate": "mlp.gate_proj",
    "mlp_up": "mlp.up_proj",
    "mlp_down": "mlp.down_proj",
    "router": "router",
}


class QueryError(Exception):
    """API error carrying the spec's error envelope (code/type/message/hint)."""

    def __init__(
        self,
        status_code: int,
        error_type: str,
        message: str,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.message = message
        self.hint = hint

    def to_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "error": {
                "code": self.status_code,
                "type": self.error_type,
                "message": self.message,
            }
        }
        if self.hint:
            body["error"]["hint"] = self.hint
        return body


def derive_type(name: str, slot: str) -> str:
    """Human-readable HF-style type label for a tensor.

    Expert tensors get ``expert.{id}.{gate|up|down}_proj``; shared experts get
    ``shared_expert.{gate|up|down}_proj``. Unknown/arch-specific slots fall
    back to the slot id itself.
    """
    if slot == "expert":
        eid = extract_expert_id(name)
        moe = get_moe_slot(name)
        if eid is not None and moe:
            return f"expert.{eid}.{moe}_proj"
        return "expert"
    if slot == "shared_expert":
        moe = get_moe_slot(name)
        if moe:
            return f"shared_expert.{moe}_proj"
        return "shared_expert"
    return _SLOT_TYPE.get(slot, slot)


def _r(x: float) -> float:
    return round(float(x), 4)


def _ratio_str(value: float, base: float) -> str | None:
    """Format value/base as ``N.N×``; None when base is 0/NaN."""
    if base is None or not np.isfinite(base) or base == 0:
        return None
    return f"{value / base:.1f}×"


def _percentile_of(value: float, values_sorted: np.ndarray) -> float | None:
    """Percentile rank of ``value`` within ascending ``values_sorted`` (0..100).

    Uses "weak" percentile-of-score semantics (fraction of values <= value),
    so the maximum maps to p100. The array MUST be ascending —
    np.searchsorted binary-searches and returns arbitrary positions on
    unsorted input.
    """
    if values_sorted.size == 0 or not np.isfinite(value):
        return None
    return _r(
        float(
            np.searchsorted(values_sorted, value, side="right")
            / values_sorted.size
            * 100.0
        )
    )


def _quantile_str(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return _r(float(np.quantile(values, q)))


# ---------------------------------------------------------------------------
# Fingerprint loading + caching
# ---------------------------------------------------------------------------
# Cache parsed fingerprints keyed by (path, mtime_ns, size) so re-scans pick up
# new data without reloading a multi-MB JSON on every request.
_scan_cache: OrderedDict[tuple[str, int, int], dict[str, Any]] = OrderedDict()
_SCAN_CACHE_MAX = 16


def _cache_key(fp_path: Path) -> tuple[str, int, int]:
    st = fp_path.stat()
    return (str(fp_path), st.st_mtime_ns, st.st_size)


def _load_fingerprint(job: Job) -> dict[str, Any]:
    fp_path = Path(job.out_dir) / "fingerprint.json"
    if not fp_path.exists():
        raise QueryError(
            404,
            "model_not_found",
            f"Fingerprint not found for '{job.job_id}' (missing fingerprint.json).",
            "Re-run the scan; this scan has no fingerprint data.",
        )
    key = _cache_key(fp_path)
    cached = _scan_cache.get(key)
    if cached is not None:
        _scan_cache.move_to_end(key)
        return cached
    with open(fp_path) as f:
        fp = cast_dict(json.load(f))
    _scan_cache[key] = fp
    _scan_cache.move_to_end(key)
    while len(_scan_cache) > _SCAN_CACHE_MAX:
        _scan_cache.popitem(last=False)
    return fp


def cast_dict(x: Any) -> dict[str, Any]:
    if not isinstance(x, dict):
        return {}
    return x


# ---------------------------------------------------------------------------
# Tensor records + caching
# ---------------------------------------------------------------------------
# Derived records are expensive to rebuild (map_name runs dozens of regexes
# per tensor name; a 74k-tensor fingerprint costs millions of regex
# evaluations). They are immutable once built (no endpoint mutates them), so
# they are cached under the same invalidation key as the parsed fingerprint.
_records_cache: OrderedDict[tuple[str, int, int], list[dict[str, Any]]] = OrderedDict()
_RECORDS_CACHE_MAX = 16


def _load_records(job: Job, fp: dict[str, Any]) -> list[dict[str, Any]]:
    """Tensor records for one scan; cached alongside the fingerprint cache."""
    try:
        key = _cache_key(Path(job.out_dir) / "fingerprint.json")
    except OSError:
        return _build_records(fp)
    cached = _records_cache.get(key)
    if cached is not None:
        _records_cache.move_to_end(key)
        return cached
    records = _build_records(fp)
    _records_cache[key] = records
    _records_cache.move_to_end(key)
    while len(_records_cache) > _RECORDS_CACHE_MAX:
        _records_cache.popitem(last=False)
    return records


def _build_records(fp: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten ``fingerprint.tensors`` into per-tensor API records.

    Adds the derived ``layer`` (None → -1), ``slot``, ``type`` and ``numel``
    columns; keeps the raw fingerprint metrics verbatim.
    """
    tensors = fp.get("tensors") or {}
    records: list[dict[str, Any]] = []
    for name, info in tensors.items():
        shape = list(info.get("shape") or [])
        numel = 1
        for s in shape:
            numel *= int(s)
        layer, slot = map_name(str(name))
        rec: dict[str, Any] = {
            "tensor_name": str(name),
            "layer": -1 if layer is None else int(layer),
            "slot": str(slot),
            "type": derive_type(str(name), str(slot)),
            "shape": shape,
            "numel": numel,
        }
        for metric in METRICS:
            val = info.get(metric)
            rec[metric] = None if val is None else float(val)
        records.append(rec)
    records.sort(key=lambda r: r["tensor_name"])
    return records


def _metric_array(records: Iterable[dict[str, Any]], metric: str) -> np.ndarray:
    vals = np.array(
        [r[metric] for r in records if r.get(metric) is not None and np.isfinite(r[metric])],
        dtype=np.float64,
    )
    return vals


def _baseline_global(records: list[dict[str, Any]], metrics: Iterable[str]) -> dict[str, Any]:
    """Global mean/median/p95/p99/max per metric over all tensors."""
    out: dict[str, Any] = {}
    for metric in metrics:
        vals = _metric_array(records, metric)
        if vals.size == 0:
            out[metric] = {"mean": None, "median": None, "p95": None, "p99": None, "max": None}
            continue
        out[metric] = {
            "mean": _r(float(np.mean(vals))),
            "median": _r(float(np.median(vals))),
            "p95": _quantile_str(vals, 0.95),
            "p99": _quantile_str(vals, 0.99),
            "max": _r(float(np.max(vals))),
        }
    return out


def _baseline_per_type(
    records: list[dict[str, Any]], metrics: Iterable[str]
) -> dict[str, dict[str, float]]:
    """Per-type means of each metric (type = derived label)."""
    by_type: dict[str, dict[str, list[float]]] = {}
    for rec in records:
        t = rec["type"]
        bucket = by_type.setdefault(t, {})
        for metric in metrics:
            val = rec.get(metric)
            if val is not None and np.isfinite(val):
                bucket.setdefault(metric, []).append(float(val))
    out: dict[str, dict[str, float]] = {}
    for t, bucket in sorted(by_type.items()):
        out[t] = {
            m: _r(float(np.mean(vals))) for m, vals in sorted(bucket.items()) if vals
        }
    return out


def build_baseline(
    records: list[dict[str, Any]], metrics: Iterable[str]
) -> dict[str, Any]:
    return {
        "global": _baseline_global(records, metrics),
        "per_type": _baseline_per_type(records, metrics),
    }


def scan_metadata(job: Job, fp: dict[str, Any]) -> dict[str, Any]:
    """Scan metadata + baseline (section 5.2)."""
    tensors = fp.get("tensors") or {}
    model = fp.get("model") or {}
    moe = model.get("moe") or {}
    quant = fp.get("quantization") or {}
    dominant_q = None
    if quant:
        dominant_q = max(quant.items(), key=lambda kv: kv[1])[0]
    records = _load_records(job, fp)
    metrics = [m for m in METRICS if any(r.get(m) is not None for r in records)]
    return {
        "model_id": job.job_id,
        "model_name": Path(job.model_path).name or Path(job.out_dir).name,
        "huggingface_id": None,
        "n_tensors": int(model.get("n_tensors") or len(tensors)),
        "n_layers": int(model.get("n_layers") or 0),
        "n_experts": int(moe.get("num_experts") or 0),
        "top_k": 0 if not moe else None,
        "arch": f"{fp.get('loader') or 'unknown'}-{'moe' if moe else 'dense'}",
        "quantization": dominant_q or "FP16",
        "author": None,
        "scan_timestamp": job.created_at,
        "tool_version": fp.get("tool_version") or "unknown",
        "metrics": metrics,
        "baseline": build_baseline(records, metrics),
    }


def list_scans(jobs: Iterable[Job]) -> dict[str, Any]:
    """Top-level scan listing (section 5.1)."""
    scans: list[dict[str, Any]] = []
    for job in jobs:
        if job.status != JobStatus.DONE:
            continue
        if not (Path(job.out_dir) / "fingerprint.json").exists():
            continue
        fp = _load_fingerprint(job)
        model = fp.get("model") or {}
        moe = model.get("moe") or {}
        quant = fp.get("quantization") or {}
        dominant_q = max(quant.items(), key=lambda kv: kv[1])[0] if quant else None
        scans.append(
            {
                "model_id": job.job_id,
                "model_name": Path(job.model_path).name or Path(job.out_dir).name,
                "arch": f"{fp.get('loader') or 'unknown'}-{'moe' if moe else 'dense'}",
                "n_tensors": int(model.get("n_tensors") or len(fp.get("tensors") or {})),
                "n_layers": int(model.get("n_layers") or 0),
                "quantization": dominant_q or "FP16",
                "scan_timestamp": job.created_at,
                "tool_version": fp.get("tool_version") or "unknown",
            }
        )
    scans.sort(key=lambda s: s["scan_timestamp"], reverse=True)
    return {"n_scans": len(scans), "scans": scans}


# ---------------------------------------------------------------------------
# Filter / slice parsing
# ---------------------------------------------------------------------------
def _parse_layer_int(expr: str, token: str) -> int:
    """Parse one integer out of a layer expression; 400 on garbage."""
    try:
        return int(token.strip())
    except ValueError:
        raise QueryError(
            400,
            "invalid_param",
            f"invalid layer expression: {expr!r}",
            "42 | >=50 | <=10 | >5 | <8 | 0:31 | 0,2,4",
        ) from None


def _parse_layer_filter(expr: str | None) -> Callable[[int], bool] | None:
    if expr is None or expr.strip() in ("", "all"):
        return None
    expr = expr.strip()
    if expr.startswith(">="):
        v = _parse_layer_int(expr, expr[2:])
        return lambda layer: layer >= v
    if expr.startswith("<="):
        v = _parse_layer_int(expr, expr[2:])
        return lambda layer: layer <= v
    if expr.startswith(">"):
        v = _parse_layer_int(expr, expr[1:])
        return lambda layer: layer > v
    if expr.startswith("<"):
        v = _parse_layer_int(expr, expr[1:])
        return lambda layer: layer < v
    if ":" in expr:
        lo_s, hi_s = expr.split(":", 1)
        lo = _parse_layer_int(expr, lo_s) if lo_s.strip() else 0
        hi = _parse_layer_int(expr, hi_s)
        return lambda layer: lo <= layer <= hi
    if "," in expr:
        vals = {_parse_layer_int(expr, x) for x in expr.split(",") if x.strip()}
        return lambda layer: layer in vals
    v = _parse_layer_int(expr, expr)
    return lambda layer: layer == v


def _match_type(rec_type: str, slot: str, expr: str) -> bool:
    if rec_type == expr or slot == expr:
        return True
    # Prefix semantics: ``self_attn`` matches ``self_attn.q_proj``; ``embed``
    # matches the ``embed`` slot of ``embed_tokens``.
    if expr in ("embed", "embed_tokens"):
        return rec_type == "embed_tokens" or slot == "embed"
    return rec_type.startswith(expr + ".")


def _parse_type_filter(expr: str | None) -> Callable[[dict[str, Any]], bool] | None:
    if expr is None or expr.strip() in ("", "all"):
        return None
    expr = expr.strip()
    return lambda rec: _match_type(rec["type"], rec["slot"], expr)


def _match_slice_pred(rec: dict[str, Any], key: str, value: str) -> bool:
    if key == "layer":
        filt = _parse_layer_filter(value)
        return filt is None or filt(rec["layer"])
    if key == "type":
        return _match_type(rec["type"], rec["slot"], value)
    if key == "slot":
        return str(rec["slot"]) == value
    raise QueryError(400, "invalid_param", f"unknown slice key: {key}", "layer / type / slot")


_SLICE_SPLIT_RE = re.compile(r"\.(?=(?:layer|type|slot):)")


def parse_slice(expr: str) -> list[tuple[str, str]]:
    """Parse ``layer:0``, ``type:mlp.gate_proj``, ``layer:42.type:mlp.gate_proj``.

    Predicates are dot-concatenated ``key:value`` pairs; the splitter only
    breaks on a dot followed by a known key so dotted type values survive.
    """
    if not expr or not expr.strip():
        raise QueryError(400, "invalid_param", "empty slice expression", "layer:N, type:X, or dot-concatenated")
    preds: list[tuple[str, str]] = []
    for part in _SLICE_SPLIT_RE.split(expr):
        part = part.strip().rstrip(".")
        if not part:
            continue
        if ":" not in part:
            raise QueryError(400, "invalid_param", f"malformed slice predicate: {part}", "key:value")
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key not in ("layer", "type", "slot"):
            raise QueryError(400, "invalid_param", f"unknown slice key: {key}", "layer / type / slot")
        preds.append((key, value))
    if not preds:
        raise QueryError(400, "invalid_param", f"empty slice expression: {expr}", "layer:N, type:X, or dot-concatenated")
    return preds


def _slice_records(records: list[dict[str, Any]], expr: str) -> list[dict[str, Any]]:
    preds = parse_slice(expr)
    return [r for r in records if all(_match_slice_pred(r, k, v) for k, v in preds)]


def _slice_aggregate(
    records: list[dict[str, Any]], metrics: Iterable[str]
) -> dict[str, Any]:
    vals: dict[str, list[float]] = {}
    for rec in records:
        for metric in metrics:
            val = rec.get(metric)
            if val is not None and np.isfinite(val):
                vals.setdefault(metric, []).append(float(val))
    agg: dict[str, Any] = {"n_tensors": len(records)}
    for metric in metrics:
        arr = vals.get(metric) or []
        if not arr:
            agg[metric] = None
        else:
            agg[metric] = _r(float(np.mean(arr)))
    return agg


# ---------------------------------------------------------------------------
# Endpoint bodies
# ---------------------------------------------------------------------------
def summary_body(
    job: Job, fp: dict[str, Any], group_by: str, metrics: list[str] | None
) -> dict[str, Any]:
    """Model-wide aggregates (section 5.3)."""
    if group_by not in ("type", "layer", "none"):
        raise QueryError(400, "invalid_param", f"unknown group_by: {group_by}", "type / layer / none")
    records = _load_records(job, fp)
    use_metrics = [m for m in (metrics or []) if m in METRICS]
    if not use_metrics:
        use_metrics = list(METRICS)

    baseline = build_baseline(records, use_metrics)
    global_means = baseline["global"]

    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        if group_by == "type":
            key = rec["type"]
        elif group_by == "layer":
            key = str(rec["layer"])
        else:
            key = "all"
        groups.setdefault(key, []).append(rec)

    def _group_rows(key: str, grp: list[dict[str, Any]]) -> dict[str, Any]:
        agg = _slice_aggregate(grp, use_metrics)
        row: dict[str, Any] = {"n_tensors": len(grp)}
        for metric in use_metrics:
            val = agg.get(metric)
            if val is None:
                row[metric] = {"mean": None, "p95": None, "max": None}
            else:
                arr = _metric_array(grp, metric)
                row[metric] = {
                    "mean": val,
                    "p95": _quantile_str(arr, 0.95),
                    "max": _r(float(np.max(arr))) if arr.size else None,
                }
        row["note"] = _summary_note(key, agg, global_means)
        if group_by == "type":
            row["type"] = key
        elif group_by == "layer":
            row["layer"] = int(key)
        return row

    rows = [_group_rows(k, v) for k, v in sorted(groups.items())]

    # Outliers: global p99 on the first metric.
    first = use_metrics[0]
    all_vals = _metric_array(records, first)
    p99 = _quantile_str(all_vals, 0.99)
    anomaly_recs = [r for r in records if r.get(first) is not None and r[first] > (p99 or np.inf)]
    mean = float(np.mean(all_vals)) if all_vals.size else 0.0
    std = float(np.std(all_vals)) if all_vals.size else 1.0
    top_outliers = sorted(anomaly_recs, key=lambda r: r[first], reverse=True)[:5]
    outliers_body = [
        {
            "tensor_name": r["tensor_name"],
            "layer": r["layer"],
            "type": r["type"],
            first: r[first],
            "zscore": _r((r[first] - mean) / std) if std else None,
        }
        for r in top_outliers
    ]

    return {
        "model_id": job.job_id,
        "group": group_by,
        "rows": rows,
        "anomaly_count": len(anomaly_recs),
        "top_outliers": outliers_body,
    }


def _summary_note(key: str, agg: dict[str, Any], global_means: dict[str, Any]) -> str:
    """One-line server-side interpretation for a summary group."""
    sp = agg.get("spectral_norm")
    gsp = (global_means.get("spectral_norm") or {}).get("mean")
    ratio = _ratio_str(sp, gsp) if sp is not None and gsp else None
    base = "global mean"
    if key == "lm_head":
        return "Output projection — high spectral norm expected"
    if key == "embed_tokens":
        return "Input embedding — high spectral norm expected"
    if key == "router":
        return "MoE router — 1-D, tiny, near-unity spectral norm expected"
    if ratio:
        return f"{ratio} {base}"
    return "n/a"


def layer_body(job: Job, fp: dict[str, Any], n: int) -> dict[str, Any]:
    """All tensors in one layer with intra-layer comparison (section 5.4)."""
    records = _load_records(job, fp)
    layer_recs = [r for r in records if r["layer"] == n]
    if not layer_recs:
        raise QueryError(
            404,
            "layer_not_found",
            f"Layer {n} not found in model '{job.job_id}' (n_layers={fp.get('model', {}).get('n_layers')}).",
            f"GET /model/{job.job_id}/query?layer={n} to confirm the index",
        )

    all_metrics = [m for m in METRICS if any(r.get(m) is not None for r in records)]
    global_means = build_baseline(records, all_metrics)["global"]

    layer_stats: dict[str, Any] = {}
    for metric in all_metrics:
        arr = _metric_array(layer_recs, metric)
        if arr.size == 0:
            continue
        layer_stats[f"{metric}_mean"] = _r(float(np.mean(arr)))
        layer_stats[f"{metric}_median"] = _r(float(np.median(arr)))
        layer_stats[f"{metric}_max"] = _r(float(np.max(arr)))

    rows: list[dict[str, Any]] = []
    # Precompute the full-model metric arrays once; _layer_flag used to
    # rebuild them per row (O(rows x model) per request).
    flag_arrays = {m: np.sort(_metric_array(records, m)) for m in ("kurtosis", "spectral_norm")}
    for rec in layer_recs:
        row: dict[str, Any] = {
            "tensor_name": rec["tensor_name"],
            "type": rec["type"],
            "slot": rec["slot"],
            "shape": rec["shape"],
        }
        for metric in all_metrics:
            val = rec.get(metric)
            row[metric] = None if val is None else _r(val)
            lm = layer_stats.get(f"{metric}_mean")
            gm = (global_means.get(metric) or {}).get("mean")
            row[f"vs_layer_mean_{metric}"] = _ratio_str(val, lm) if val is not None and lm else None
            row[f"vs_global_mean_{metric}"] = _ratio_str(val, gm) if val is not None and gm else None
        row["flag"] = _layer_flag(rec, flag_arrays)
        rows.append(row)

    return {
        "model_id": job.job_id,
        "layer": n,
        "n_tensors": len(layer_recs),
        "layer_stats": layer_stats,
        "global_context": {
            m: (global_means.get(m) or {}).get("mean") for m in all_metrics
        },
        "rows": rows,
    }


def _layer_flag(
    rec: dict[str, Any], metric_arrays: dict[str, np.ndarray]
) -> str | None:
    """Flag unusual tensors within a layer (kurtosis / spectral outliers).

    ``metric_arrays`` holds the ascending-sorted full-model arrays per metric,
    precomputed by the caller.
    """
    for metric in ("kurtosis", "spectral_norm"):
        val = rec.get(metric)
        if val is None or not np.isfinite(val):
            continue
        pct = _percentile_of(val, metric_arrays[metric])
        if metric == "kurtosis" and pct is not None and pct >= 99.5:
            return f"kurtosis outlier (p{pct:.2f})"
        if metric == "spectral_norm" and pct is not None and pct >= 95:
            return f"spectral_norm p{pct:.0f}"
    return None


def anomalies_body(
    job: Job,
    fp: dict[str, Any],
    metric: str,
    threshold: str,
    method: str,
    n: int,
    direction: str,
    type_filter: str | None,
    layer_range: str | None,
    fields: list[str] | None,
) -> dict[str, Any]:
    """Statistically unusual tensors (section 5.5)."""
    if metric not in METRICS:
        raise QueryError(400, "invalid_param", f"unknown metric: {metric}", f"one of {', '.join(METRICS)}")
    if method not in ("quantile", "IQR", "zscore"):
        raise QueryError(400, "invalid_param", f"unknown method: {method}", "quantile / IQR / zscore")
    if direction not in ("high", "low", "both"):
        raise QueryError(400, "invalid_param", f"unknown direction: {direction}", "high / low / both")

    records = _load_records(job, fp)
    if type_filter:
        tf = _parse_type_filter(type_filter)
        records = [r for r in records if tf is None or tf(r)]
    if layer_range:
        lf = _parse_layer_filter(layer_range)
        records = [r for r in records if lf is None or lf(r["layer"])]

    vals = _metric_array(records, metric)
    if vals.size < 2:
        raise QueryError(400, "invalid_param", "too few tensors after filtering to compute anomalies")

    resolved: float | None = None
    if method == "quantile":
        qmap = {"p95": 0.95, "p99": 0.99, "p999": 0.999}
        if threshold in qmap:
            resolved = _quantile_str(vals, qmap[threshold])
        else:
            try:
                resolved = float(threshold)
            except ValueError:
                raise QueryError(
                    400, "invalid_param", f"unknown threshold: {threshold}", "p95 / p99 / p999 or a numeric value"
                ) from None
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    q1 = float(np.quantile(vals, 0.25))
    q3 = float(np.quantile(vals, 0.75))
    iqr = q3 - q1

    def _is_anomaly(val: float) -> bool:
        high = False
        low = False
        if method == "quantile":
            high = val >= (resolved or np.inf)
            low = val <= (resolved or -np.inf)
        elif method == "IQR":
            high = val >= q3 + 1.5 * iqr
            low = val <= q1 - 1.5 * iqr
        else:  # zscore
            high = val >= mean + 3 * std
            low = val <= mean - 3 * std
        if direction == "high":
            return high
        if direction == "low":
            return low
        return high or low

    rows: list[dict[str, Any]] = []
    vals_sorted = np.sort(vals)
    for rec in records:
        val = rec.get(metric)
        if val is None or not np.isfinite(val) or not _is_anomaly(val):
            continue
        pct = _percentile_of(val, vals_sorted)
        row: dict[str, Any] = {
            "tensor_name": rec["tensor_name"],
            "layer": rec["layer"],
            "type": rec["type"],
            "slot": rec["slot"],
            metric: _r(val),
            "zscore": _r((val - mean) / std) if std else None,
            "percentile": pct,
            "vs_global_mean": _ratio_str(val, mean),
            "context": _anomaly_context(rec["type"], val, mean, std),
        }
        if fields:
            row = _trim_fields(row, fields)
        rows.append(row)

    rows.sort(key=lambda r: (r.get("zscore") or 0), reverse=True)
    rows = rows[:max(0, n)]
    return {
        "model_id": job.job_id,
        "metric": metric,
        "threshold": {"method": method, "value": threshold, "resolved": resolved},
        "n_anomalies": len(rows),
        "rows": rows,
    }


def _anomaly_context(t: str, val: float, mean: float, std: float) -> str:
    z = (val - mean) / std if std else 0
    if t == "lm_head":
        return "Output projection — high norm expected, but extreme here"
    if t == "embed_tokens":
        return "Input embedding — high norm expected, but extreme here"
    if z > 3:
        return f"More than 3σ above the model mean (z={z:.1f})"
    if z < -3:
        return f"More than 3σ below the model mean (z={z:.1f})"
    return "Outside the typical range for this tensor type"


def query_body(
    job: Job,
    fp: dict[str, Any],
    layer: str | None,
    type_filter: str | None,
    metric: str | None,
    order: str,
    fields: list[str] | None,
    limit: int,
    offset: int,
    min_val: float | None,
    max_val: float | None,
) -> dict[str, Any]:
    """Filtered, sorted, paginated tensor list (section 5.6)."""
    limit = max(0, min(limit, 500))
    offset = max(0, offset)
    records = _load_records(job, fp)

    if type_filter:
        tf = _parse_type_filter(type_filter)
        records = [r for r in records if tf is None or tf(r)]
    if layer:
        lf = _parse_layer_filter(layer)
        records = [r for r in records if lf is None or lf(r["layer"])]
    if min_val is not None:
        records = [r for r in records if _any_metric(r, metric) >= min_val]
    if max_val is not None:
        records = [r for r in records if _any_metric(r, metric) <= max_val]

    sort_metric = metric if metric in METRICS else None
    if sort_metric:
        def _sort_key(r: dict[str, Any]) -> float:
            val = r.get(sort_metric)
            return float(val) if val is not None else -np.inf
        records.sort(key=_sort_key, reverse=(order == "desc"))
    else:
        records.sort(key=lambda r: r["tensor_name"])

    total = len(records)
    page = records[offset:offset + limit]
    has_more = offset + len(page) < total
    next_offset = offset + len(page) if has_more else None

    metrics = [m for m in METRICS if any(r.get(m) is not None for r in page)] or list(METRICS)
    rows = []
    for rec in page:
        row: dict[str, Any] = {
            "tensor_name": rec["tensor_name"],
            "layer": rec["layer"],
            "type": rec["type"],
            "slot": rec["slot"],
            "shape": rec["shape"],
            "numel": rec["numel"],
        }
        for m in metrics:
            val = rec.get(m)
            row[m] = None if val is None else _r(val)
        if fields:
            row = _trim_fields(row, fields)
        rows.append(row)

    baseline: dict[str, Any] = {}
    if type_filter and rows:
        baseline = {"type": type_filter}
        for m in metrics:
            arr = _metric_array(page, m)
            baseline[f"{m}_mean"] = _r(float(np.mean(arr))) if arr.size else None
            baseline[f"{m}_p95"] = _quantile_str(arr, 0.95)
        # Keep the _baseline small: only the requested/filtered subset.
        baseline = {"type": type_filter, **{k: v for k, v in baseline.items() if k != "type"}}
    else:
        baseline = build_baseline(page, metrics)

    return {
        "model_id": job.job_id,
        "n_results": total,
        "offset": offset,
        "limit": limit,
        "has_more": has_more,
        "next_offset": next_offset,
        "_baseline": baseline,
        "rows": rows,
    }


def _any_metric(rec: dict[str, Any], metric: str | None) -> float:
    if metric and metric in METRICS:
        val = rec.get(metric)
        if val is not None:
            return float(val)
    return -np.inf


def _trim_fields(row: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """Trim a row to the requested columns; always keep tensor_name."""
    keep = set(fields) | {"tensor_name"}
    return {k: v for k, v in row.items() if k in keep}


def compare_body(
    job: Job,
    fp: dict[str, Any],
    a: str,
    b: str,
    metrics: list[str] | None,
    fields: list[str] | None,
) -> dict[str, Any]:
    """Two-slice in-model comparison (section 5.7)."""
    records = _load_records(job, fp)
    use_metrics = [m for m in (metrics or []) if m in METRICS] or list(METRICS)
    a_recs = _slice_records(records, a)
    b_recs = _slice_records(records, b)
    if not a_recs:
        raise QueryError(404, "slice_not_found", f"slice 'a' matched no tensors: {a}")
    if not b_recs:
        raise QueryError(404, "slice_not_found", f"slice 'b' matched no tensors: {b}")
    a_agg = _slice_aggregate(a_recs, use_metrics)
    b_agg = _slice_aggregate(b_recs, use_metrics)

    delta: dict[str, Any] = {}
    for metric in use_metrics:
        av = a_agg.get(metric)
        bv = b_agg.get(metric)
        if av is None or bv is None:
            continue
        abs_change = float(bv) - float(av)
        pct = (abs_change / av * 100.0) if av else None
        delta[metric] = {
            "a": av,
            "b": bv,
            "abs": _r(abs_change),
            "pct": _r(pct) if pct is not None else None,
            "interpretation": _delta_interpretation(metric, av, bv, pct),
        }

    return {
        "model_id": job.job_id,
        "a": {"slice": a, "n_tensors": len(a_recs), **{k: v for k, v in a_agg.items() if k != "n_tensors"}},
        "b": {"slice": b, "n_tensors": len(b_recs), **{k: v for k, v in b_agg.items() if k != "n_tensors"}},
        "delta": delta,
    }


def _delta_interpretation(
    metric: str, a: float, b: float, pct: float | None
) -> str:
    if metric == "spectral_norm":
        if pct is not None and pct > 15:
            return f"{pct:.0f}% higher mean spectral norm in slice b"
        if pct is not None and pct < -15:
            return f"{-pct:.0f}% lower mean spectral norm in slice b"
        return "similar spectral norm between slices"
    if metric == "kurtosis":
        return "kurtosis dominated by a single heavy-tailed tensor (e.g. a layernorm)" if abs(a - b) > a * 2 else "similar kurtosis"
    return "n/a"


def histogram_body(
    job: Job,
    fp: dict[str, Any],
    metric: str,
    bins: int,
    log: bool,
    type_filter: str | None,
    layer_range: str | None,
    density: bool,
) -> dict[str, Any]:
    """Distribution of a metric (section 5.9)."""
    if metric not in METRICS:
        raise QueryError(400, "invalid_param", f"unknown metric: {metric}", f"one of {', '.join(METRICS)}")
    bins = max(1, min(int(bins), 200))
    records = _load_records(job, fp)
    if type_filter:
        tf = _parse_type_filter(type_filter)
        records = [r for r in records if tf is None or tf(r)]
    if layer_range:
        lf = _parse_layer_filter(layer_range)
        records = [r for r in records if lf is None or lf(r["layer"])]

    vals = _metric_array(records, metric)
    if vals.size == 0:
        raise QueryError(400, "invalid_param", "no values after filtering to histogram")
    if log:
        positive = vals[vals > 0]
        if positive.size == 0:
            raise QueryError(400, "invalid_param", "log=true but no positive values")
        lo = float(np.log10(np.min(positive)))
        hi = float(np.log10(np.max(positive)))
        counts, edges_log = np.histogram(np.log10(positive), bins=bins, range=(lo, hi))
        edges = 10.0 ** edges_log
    else:
        counts, edges = np.histogram(vals, bins=bins)

    total = int(np.sum(counts))
    bin_rows: list[dict[str, Any]] = []
    for i in range(len(counts)):
        bin_rows.append(
            {
                "range": [_r(edges[i]), _r(edges[i + 1])],
                "count": int(counts[i]),
                "pct": _r(float(counts[i]) / total * 100.0),
            }
        )
    skew = float(((vals - np.mean(vals)) ** 3).mean() / (np.std(vals) ** 3)) if np.std(vals) else 0.0
    dist = {
        "mean": _r(float(np.mean(vals))),
        "median": _r(float(np.median(vals))),
        "skew": _r(skew),
        "kurtosis": _r(float(((vals - np.mean(vals)) ** 4).mean() / (np.std(vals) ** 4) - 3)) if np.std(vals) else None,
    }
    # Density normalisation.
    if density and len(bin_rows):
        for br in bin_rows:
            width = br["range"][1] - br["range"][0]
            br["density"] = _r(br["pct"] / 100.0 / width) if width else None

    return {
        "model_id": job.job_id,
        "metric": metric,
        "bins": bin_rows,
        "total": total,
        "distribution": dist,
        "shape_description": _shape_description(bin_rows, dist, vals),
    }


def _shape_description(bin_rows: list[dict[str, Any]], dist: dict[str, Any], vals: np.ndarray) -> str:
    skew = dist.get("skew")
    mode = max(bin_rows, key=lambda b: b["count"]) if bin_rows else None
    side = "Right-skewed" if skew and skew > 0.5 else ("Left-skewed" if skew and skew < -0.5 else "Roughly symmetric")
    mode_str = f"{mode['range'][0]:g}-{mode['range'][1]:g}" if mode else "n/a"
    tail = f"long tail to {float(np.max(vals)):g}" if np.isfinite(vals).any() and skew and abs(float(skew)) > 0.5 else "no extreme tail"
    return f"{side}, mode at {mode_str}, {tail}"


def tensor_body(job: Job, fp: dict[str, Any], name: str) -> dict[str, Any]:
    """Full detail for one tensor (section 5.10)."""
    records = _load_records(job, fp)
    by_name = {r["tensor_name"]: r for r in records}
    rec = by_name.get(name)
    if rec is None:
        raise QueryError(
            404,
            "tensor_not_found",
            f"Tensor '{name}' not found in model '{job.job_id}'.",
            f"GET /model/{job.job_id}/query?type={name.rsplit('.', 1)[0] if '.' in name else 'all'} to list matching tensors",
        )
    metrics = [m for m in METRICS if rec.get(m) is not None]

    type_recs = [r for r in records if r["type"] == rec["type"]]
    global_means = build_baseline(records, metrics)["global"]
    # Hoist the per-metric arrays (each was previously rebuilt twice per
    # metric); percentile needs them ascending.
    type_arrays = {m: _metric_array(type_recs, m) for m in metrics}
    model_arrays = {m: np.sort(_metric_array(records, m)) for m in metrics}

    ctx: dict[str, Any] = {"vs_type_mean": {}, "vs_global_mean": {}, "percentile_in_model": {}}
    for metric in metrics:
        val = rec.get(metric)
        t_arr = type_arrays[metric]
        tmean = float(np.mean(t_arr)) if t_arr.size else None
        gmean = (global_means.get(metric) or {}).get("mean")
        ctx["vs_type_mean"][metric] = _ratio_str(val, tmean) if val is not None and tmean else None
        ctx["vs_global_mean"][metric] = _ratio_str(val, gmean) if val is not None and gmean else None
        ctx["percentile_in_model"][metric] = (
            _percentile_of(val, model_arrays[metric]) if val is not None else None
        )

    return {
        "model_id": job.job_id,
        "tensor_name": rec["tensor_name"],
        "layer": rec["layer"],
        "slot": rec["slot"],
        "type": rec["type"],
        "shape": rec["shape"],
        "numel": rec["numel"],
        "metrics": {m: _r(rec[m]) for m in metrics},
        "context": ctx,
        "interpretation": _tensor_interpretation(rec, ctx),
    }


def _tensor_interpretation(rec: dict[str, Any], ctx: dict[str, Any]) -> str:
    sp = rec.get("spectral_norm")
    sp_pct = (ctx.get("percentile_in_model") or {}).get("spectral_norm")
    kurt_pct = (ctx.get("percentile_in_model") or {}).get("kurtosis")
    parts: list[str] = []
    if sp_pct is not None and sp_pct >= 95:
        parts.append("spectral norm is a p95+ outlier")
    if kurt_pct is not None and kurt_pct >= 99.5:
        parts.append("kurtosis is extreme (p99.5+)")
    if not parts:
        parts.append("no anomalies")
    type_note = ""
    if rec["type"] == "lm_head":
        type_note = " Output projection — high norm expected."
    elif rec["type"] == "embed_tokens":
        type_note = " Input embedding — high norm expected."
    elif sp is not None and ctx.get("vs_global_mean", {}).get("spectral_norm"):
        type_note = f" Spectral norm is {ctx['vs_global_mean']['spectral_norm']} the model mean."
    return f"{rec['type']}: " + "; ".join(parts) + type_note


def delta_body(
    job_a: Job,
    fp_a: dict[str, Any],
    job_b: Job,
    fp_b: dict[str, Any],
    metric: str,
    n: int,
    min_change_pct: float,
    fields: list[str] | None,
    jobs: JobQueue,
) -> dict[str, Any]:
    """Cross-scan delta (section 5.8). Tier 1 uses weight-space paired/edit
    artefacts when available; tier 2 diffs fingerprint statistics."""
    if metric not in METRICS:
        raise QueryError(400, "invalid_param", f"unknown metric: {metric}", f"one of {', '.join(METRICS)}")
    n = max(0, min(int(n), 500))
    min_change_pct = float(min_change_pct)

    # Tier 1: find a DONE paired/edit job pairing these two scans.
    tier1 = _find_edit_job(jobs, job_a, job_b)
    if tier1 is not None:
        return tier1

    records_a = _load_records(job_a, fp_a)
    b_by_name = {r["tensor_name"]: r for r in _load_records(job_b, fp_b)}
    rows: list[dict[str, Any]] = []
    for ra in records_a:
        rb = b_by_name.get(ra["tensor_name"])
        if rb is None:
            continue
        av = ra.get(metric)
        bv = rb.get(metric)
        if av is None or bv is None or not np.isfinite(av) or not np.isfinite(bv):
            continue
        if av == 0:
            continue
        pct = (float(bv) - float(av)) / float(av) * 100.0
        if abs(pct) < min_change_pct:
            continue
        rows.append(
            {
                "tensor_name": ra["tensor_name"],
                "layer": ra["layer"],
                "type": ra["type"],
                "a": _r(float(av)),
                "b": _r(float(bv)),
                "abs_change": _r(float(bv) - float(av)),
                "pct_change": _r(pct),
                "context": f"{pct:+.1f}% change in {metric}",
            }
        )
    rows.sort(key=lambda r: abs(r["pct_change"]), reverse=True)
    rows = rows[:n]
    if fields:
        rows = [_trim_fields(r, fields) for r in rows]

    changes = [r["pct_change"] for r in rows]
    summary: dict[str, Any] = {
        "mean_change_pct": _r(float(np.mean(changes))) if changes else None,
        "max_change_pct": _r(float(max(changes, key=abs))) if changes else None,
    }
    if rows:
        by_type: dict[str, int] = {}
        for r in rows:
            by_type[r["type"]] = by_type.get(r["type"], 0) + 1
        summary["most_affected_type"] = max(by_type.items(), key=lambda kv: kv[1])[0]
        layers = sorted({r["layer"] for r in rows if r["layer"] >= 0})
        summary["most_affected_layer_range"] = (
            f"{min(layers)}-{max(layers)}" if layers else None
        )

    return {
        "model_a": job_a.job_id,
        "model_b": job_b.job_id,
        "tier": "statistic_diff",
        "metric": metric,
        "n_compared": len(records_a),
        "n_changed_above_5pct": len([r for r in rows if abs(r["pct_change"]) >= 5]),
        "summary": summary,
        "rows": rows,
    }


def _find_edit_job(
    jobs: JobQueue, job_a: Job, job_b: Job
) -> dict[str, Any] | None:
    """Tier-1 detection: a DONE paired job pairing these two scans.

    Looks for a compare-type job whose ``model_path`` encodes ``dir_a|dir_b``
    matching the two scans' out dirs and whose ``compare_summary.json`` carries
    ``preset == "edit"`` (the edit-signature artefacts). Returns the weight-space
    delta body when found, else None.
    """
    dirs = {Path(job_a.out_dir).resolve(), Path(job_b.out_dir).resolve()}
    for job in jobs.list_jobs(limit=200):
        if job.status != JobStatus.DONE or job.job_type != "compare":
            continue
        parts = job.model_path.split("|")
        if len(parts) != 2:
            continue
        pair = {Path(p).resolve() for p in parts}
        if pair != dirs:
            continue
        summary_path = Path(job.out_dir) / "compare_summary.json"
        if not summary_path.exists():
            continue
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except (OSError, ValueError):
            continue
        if summary.get("preset") != "edit":
            continue
        body: dict[str, Any] = {
            "model_a": job_a.job_id,
            "model_b": job_b.job_id,
            "tier": "weight_space",
            "n_compared": (summary.get("edit_signature") or {}).get("n_tensors")
            or summary.get("alignment", {}).get("n_pairs")
            or None,
            "edit_signature": summary.get("edit_signature"),
            "noise_floor": summary.get("noise_floor"),
            "warnings": summary.get("warnings"),
            "rows": [
                {
                    "layer": h.get("layer"),
                    "slot": h.get("slot"),
                    "rel_l2": h.get("rel_l2"),
                    "name_a": h.get("name_a"),
                    "name_b": h.get("name_b"),
                }
                for h in (summary.get("edit_signature") or {}).get("hotspot_ranking_rel_l2") or []
            ],
        }
        return body
    return None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discovery_body() -> dict[str, Any]:
    """GET /api — in-band self-description for synthetic consumers."""
    endpoints = [
        {"method": "GET", "path": "/api", "purpose": "self-description (this document)", "params": []},
        {"method": "GET", "path": "/api/schema", "purpose": "machine-readable field schema", "params": []},
        {"method": "GET", "path": "/api/models", "purpose": "list all scans", "params": []},
        {"method": "GET", "path": "/api/model/{model_id}", "purpose": "scan metadata + baseline", "params": []},
        {"method": "GET", "path": "/api/model/{model_id}/summary", "purpose": "model-wide aggregates", "params": ["group_by", "metrics"]},
        {"method": "GET", "path": "/api/model/{model_id}/layer/{n}", "purpose": "all tensors in one layer, intra-layer comparison", "params": []},
        {"method": "GET", "path": "/api/model/{model_id}/anomalies", "purpose": "statistically unusual tensors", "params": ["metric", "threshold", "method", "n", "direction", "type", "layer_range", "fields"]},
        {"method": "GET", "path": "/api/model/{model_id}/query", "purpose": "filtered, sorted, paginated tensor list", "params": ["layer", "type", "metric", "order", "fields", "limit", "offset", "min", "max"]},
        {"method": "GET", "path": "/api/model/{model_id}/compare", "purpose": "two slices within one model", "params": ["a", "b", "metrics", "fields"]},
        {"method": "GET", "path": "/api/model/{model_id}/histogram", "purpose": "distribution of a metric", "params": ["metric", "bins", "log", "type", "layer_range", "density"]},
        {"method": "GET", "path": "/api/model/{model_id}/tensor/{name}", "purpose": "full detail for one tensor", "params": []},
        {"method": "GET", "path": "/api/model/{model_id}/delta", "purpose": "cross-scan comparison (weight-space tier preferred)", "params": ["with", "metric", "n", "min_change_pct", "fields"]},
    ]
    return {
        "api_version": "0.2",
        "docs": "/docs",
        "endpoints": endpoints,
        "metrics": METRIC_DESCRIPTIONS,
        "types": sorted(
            set(_SLOT_TYPE.values())
            | {"expert.{id}.{gate|up|down}_proj", "shared_expert.{gate|up|down}_proj"}
        ),
        "filter_grammar": {
            "layer": "42 | >=50 | 0:31 | 0,2,4 | -1 (non-layer)",
            "type": "mlp.gate_proj | self_attn | embed | lm_head",
            "threshold": "p95 | p99 | p999 | 15.0",
            "slice": "layer:42.type:mlp.gate_proj",
        },
        "pagination": {"limit_max": 500, "offset": "int", "has_more": "bool", "next_offset": "int|null"},
    }


def schema_body() -> dict[str, Any]:
    """GET /api/schema — OpenAPI-style field-level schema for response bodies."""
    return {
        "api_version": "0.2",
        "response_schemas": {
            "models": {
                "n_scans": "int",
                "scans": [{
                    "model_id": "string (job_id)", "model_name": "string",
                    "arch": "string", "n_tensors": "int", "n_layers": "int",
                    "quantization": "string", "scan_timestamp": "string (ISO-8601)",
                    "tool_version": "string",
                }],
            },
            "model": {
                "model_id": "string", "model_name": "string", "huggingface_id": "string|null",
                "n_tensors": "int", "n_layers": "int", "n_experts": "int", "top_k": "int|null",
                "arch": "string", "quantization": "string", "author": "string|null",
                "scan_timestamp": "string", "tool_version": "string", "metrics": "[string]",
                "baseline": {"global": "metric->{mean,median,p95,p99,max}", "per_type": "type->metric->mean"},
            },
            "summary": {
                "model_id": "string", "group": "type|layer|none",
                "rows": [{"n_tensors": "int", "type|layer": "string|int", "metric": "{mean,p95,max}", "note": "string"}],
                "anomaly_count": "int", "top_outliers": "[{tensor_name, layer, type, metric, zscore}]",
            },
            "layer": {
                "model_id": "string", "layer": "int", "n_tensors": "int",
                "layer_stats": "metric_{mean,median,max}", "global_context": "metric->mean",
                "rows": [{"tensor_name", "type", "slot", "shape", "metric", "vs_layer_mean_*", "vs_global_mean_*", "flag"}],
            },
            "anomalies": {
                "model_id": "string", "metric": "string",
                "threshold": {"method", "value", "resolved"},
                "n_anomalies": "int",
                "rows": [{"tensor_name", "layer", "type", "slot", "metric", "zscore", "percentile", "vs_global_mean", "context"}],
            },
            "query": {
                "model_id": "string", "n_results": "int", "offset": "int", "limit": "int",
                "has_more": "bool", "next_offset": "int|null", "_baseline": "object",
                "rows": ["tensor record (fields trims columns)"],
            },
            "compare": {
                "model_id": "string",
                "a": {"slice", "n_tensors", "metric->mean"}, "b": "same",
                "delta": "metric->{a, b, abs, pct, interpretation}",
            },
            "histogram": {
                "model_id": "string", "metric": "string",
                "bins": [{"range": "[lo, hi]", "count": "int", "pct": "float"}],
                "total": "int", "distribution": {"mean", "median", "skew", "kurtosis"},
                "shape_description": "string",
            },
            "tensor": {
                "model_id": "string", "tensor_name": "string", "layer": "int", "slot": "string",
                "type": "string", "shape": "[int]", "numel": "int", "metrics": "metric->float",
                "context": {"vs_type_mean", "vs_global_mean", "percentile_in_model"},
                "interpretation": "string",
            },
            "delta": {
                "model_a": "string", "model_b": "string", "tier": "weight_space|statistic_diff",
                "metric": "string (tier 2)", "n_compared": "int",
                "edit_signature": "object (tier 1)", "noise_floor": "object (tier 1)",
                "summary": {"mean_change_pct", "max_change_pct", "most_affected_type", "most_affected_layer_range"},
                "rows": ["tensor-level change rows"],
            },
        },
        "error": {"error": {"code": "int", "type": "string", "message": "string", "hint": "string|null"}},
    }
