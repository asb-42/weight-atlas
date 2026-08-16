# Weight Atlas — LLM Query API Specification

**Version:** 0.1 (draft)
**Date:** 2026-08-16
**Status:** Proposal

---

## 1. Purpose

Weight Atlas scans LLM weight files and extracts per-tensor statistics into a structured dataset. This API exposes that dataset through targeted query endpoints, enabling both human users (via web UI) and LLM agents (via REST) to analyse model weight topology without loading the full 25 MB table into context.

**Design principle:** The LLM is an analyst, not a reader. It asks targeted questions and receives 20–200 rows per response. The full table remains available for human visual inspection (TIFF, HTML, topographic renders).

---

## 2. Data Model

### 2.1 Tensor Record

Each row in the statistics table represents one tensor:

| Field | Type | Description |
|-------|------|-------------|
| `tensor_name` | string | Full dotted path, e.g. `language_model.model.layers.42.mlp.gate_proj.weight` |
| `layer` | int | Layer index (0-based). `-1` for non-layer tensors (embed, lm_head, final norm). |
| `slot` | int | Column index in the raster (position in the model's parameter layout). |
| `shape` | list[int] | Tensor dimensions, e.g. `[33792, 7168]`. |
| `numel` | int | Total number of elements. |
| `frobenius` | float | Frobenius norm. |
| `spectral_norm` | float | Spectral norm (largest singular value). |
| `effective_rank` | float | Stable / effective rank (spectral entropy-based). |
| `kurtosis` | float | Excess kurtosis of the weight distribution. |
| `sparsity` | float | Fraction of weights below a threshold (e.g. 0.01). |

### 2.2 Scan Metadata

| Field | Type | Description |
|-------|------|-------------|
| `scan_id` | string | Unique identifier for this scan. |
| `model_name` | string | Human-readable model name, e.g. `Qwen3.8-27B Q4_K_M`. |
| `model_id` | string | HuggingFace-style ID or local path. |
| `n_tensors` | int | Total number of scanned tensors. |
| `n_layers` | int | Number of transformer layers. |
| `n_experts` | int | Number of MoE experts (0 if dense). |
| `top_k` | int | Experts activated per token (0 if dense). |
| `hidden_dim` | int | Hidden size. |
| `arch` | string | Architecture tag, e.g. `qwen3-dense`, `kimi-k3-moe`. |
| `quantization` | string | Quantization method, e.g. `Q4_K_M`, `FP16`. |
| `scan_timestamp` | string | ISO-8601 UTC timestamp. |
| `waste_version` | string | Weight Atlas engine version. |

---

## 3. Endpoints

Base URL: `http://<host>:<port>/api`

All responses are `application/json`.

### 3.1 `GET /meta`

Returns scan metadata.

**Response:**
```json
{
  "scan_id": "k3-2026-08-15-a1b2c3",
  "model_name": "Kimi K3",
  "model_id": "kimi/k3",
  "n_tensors": 249924,
  "n_layers": 93,
  "n_experts": 896,
  "top_k": 16,
  "hidden_dim": 7168,
  "arch": "kimi-k3-moe",
  "quantization": "Q4_K_M",
  "scan_timestamp": "2026-08-15T22:00:00Z",
  "waste_version": "0.2.0",
  "metrics": ["frobenius", "spectral_norm", "effective_rank", "kurtosis", "sparsity"]
}
```

---

### 3.2 `GET /summary`

Aggregated statistics. Returns global, per-layer, and per-tensor-type summaries.

**Query parameters (optional):**

| Param | Values | Default |
|-------|--------|---------|
| `group_by` | `layer`, `type`, `none` | `none` |
| `metric` | any metric name | all metrics |
| `agg` | `mean,median,p50,p90,p95,p99,max,min,std` | `mean,median,p95,max` |

**Response (group_by=layer, metric=spectral_norm, first 3 of 93):**
```json
{
  "group": "layer",
  "metric": "spectral_norm",
  "rows": [
    {"key": 0,  "mean": 12.3, "median": 11.8, "p95": 15.2, "max": 20.1},
    {"key": 1,  "mean": 13.1, "median": 12.5, "p95": 16.0, "max": 21.4},
    {"key": 2,  "mean": 12.8, "median": 12.1, "p95": 15.5, "max": 19.8}
  ],
  "n_rows": 93
}
```

**Response (group_by=type):**
```json
{
  "group": "type",
  "rows": [
    {"key": "embed_tokens", "n_tensors": 1, "mean_spectral_norm": 49.7, "max_kurtosis": 0.40},
    {"key": "lm_head",      "n_tensors": 1, "mean_spectral_norm": 204.5, "max_kurtosis": 0.43},
    {"key": "input_layernorm", "n_tensors": 93, "mean_spectral_norm": 14.0, "max_kurtosis": 290.9},
    {"key": "mlp.gate_proj",   "n_tensors": 93, "mean_spectral_norm": 13.4, "max_kurtosis": 2.19},
    {"key": "mlp.up_proj",     "n_tensors": 93, "mean_spectral_norm": 13.1, "max_kurtosis": 2.10},
    {"key": "mlp.down_proj",   "n_tensors": 93, "mean_spectral_norm": 10.2, "max_kurtosis": 2.04},
    {"key": "self_attn.q_proj", "n_tensors": 93, "mean_spectral_norm": 14.2, "max_kurtosis": 1.8},
    {"key": "self_attn.k_proj", "n_tensors": 93, "mean_spectral_norm": 13.9, "max_kurtosis": 1.7},
    {"key": "self_attn.v_proj", "n_tensors": 93, "mean_spectral_norm": 14.1, "max_kurtosis": 1.9},
    {"key": "self_attn.o_proj", "n_tensors": 93, "mean_spectral_norm": 13.5, "max_kurtosis": 1.6}
  ],
  "n_rows": 10
}
```

---

### 3.3 `GET /query`

Filtered, sorted, paginated tensor list.

**Query parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `layer` | int or range | Filter by layer, e.g. `42` or `>=50` or `50:70` |
| `type` | string | Filter by tensor type, e.g. `mlp.gate_proj`, `self_attn` |
| `metric` | string | Sort by this metric |
| `order` | `asc` or `desc` | Sort direction |
| `limit` | int | Max rows (default 50, max 500) |
| `offset` | int | Pagination offset |
| `min` | float | Minimum value for the sorted metric |
| `max` | float | Maximum value for the sorted metric |

**Response:**
```json
{
  "n_results": 93,
  "offset": 0,
  "limit": 5,
  "rows": [
    {
      "tensor_name": "language_model.model.layers.91.mlp.gate_proj.weight",
      "layer": 91,
      "shape": [33792, 7168],
      "frobenius": 354.8,
      "spectral_norm": 13.4,
      "effective_rank": 15.7,
      "kurtosis": 2.19,
      "sparsity": 0.217
    }
  ]
}
```

---

### 3.4 `GET /anomalies`

Returns the N most anomalous tensors for a given metric, using a configurable detection method.

**Query parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `metric` | string | Metric to detect anomalies on (default: `spectral_norm`) |
| `method` | `IQR`, `zscore`, `mad` | Detection method (default: `IQR`) |
| `n` | int | Number of results (default: 50) |
| `direction` | `high`, `low`, `both` | Which side to detect (default: `both`) |

**Response:**
```json
{
  "metric": "spectral_norm",
  "method": "IQR",
  "threshold_high": 18.2,
  "threshold_low": 8.1,
  "n_anomalies": 12,
  "rows": [
    {
      "tensor_name": "language_model.model.layers.42.mlp.gate_proj.weight",
      "layer": 42,
      "spectral_norm": 19.7,
      "zscore": 3.8,
      "deviation": "above_high"
    },
    {
      "tensor_name": "language_model.lm_head.weight",
      "layer": -1,
      "spectral_norm": 204.5,
      "zscore": 28.1,
      "deviation": "above_high"
    }
  ]
}
```

---

### 3.5 `GET /compare`

Side-by-side comparison of two slices (layer, type, or range).

**Query parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `a` | string | Slice A, e.g. `layer:0` or `type:mlp.gate_proj` |
| `b` | string | Slice B, e.g. `layer:92` or `type:mlp.down_proj` |

**Response:**
```json
{
  "a": {"slice": "layer:0", "n_tensors": 10, "mean_spectral_norm": 12.3, "max_kurtosis": 290.9},
  "b": {"slice": "layer:92", "n_tensors": 10, "mean_spectral_norm": 14.8, "max_kurtosis": 2.5},
  "delta": {
    "spectral_norm": {"a": 12.3, "b": 14.8, "abs": 2.5, "pct": 20.3},
    "kurtosis": {"a": 290.9, "b": 2.5, "abs": -288.4, "pct": -99.1}
  }
}
```

---

### 3.6 `GET /delta`

Comparison between two scans (e.g. base vs. abliterated model).

**Query parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `model_a` | string | Scan ID or model name for baseline |
| `model_b` | string | Scan ID or model name for comparison |
| `metric` | string | Metric to diff (default: `spectral_norm`) |
| `n` | int | Number of most-changed tensors (default: 50) |
| `min_change_pct` | float | Minimum % change to include (default: 0) |

**Response:**
```json
{
  "model_a": "k3-base",
  "model_b": "k3-abliterated",
  "n_compared": 249924,
  "n_changed_above_threshold": 3421,
  "rows": [
    {
      "tensor_name": "language_model.model.layers.42.mlp.gate_proj.weight",
      "a": 13.4,
      "b": 2.1,
      "abs_change": -11.3,
      "pct_change": -84.3
    }
  ]
}
```

---

### 3.7 `GET /histogram`

Distribution of a metric as binned counts. Useful for CDF analysis, mode detection, skew.

**Query parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `metric` | string | Metric (default: `spectral_norm`) |
| `bins` | int | Number of bins (default: 20, max: 100) |
| `filter` | string | Optional filter, same syntax as `/query` |

**Response:**
```json
{
  "metric": "spectral_norm",
  "bins": [
    {"range": [0.0, 1.0], "count": 1204},
    {"range": [1.0, 2.0], "count": 8930},
    {"range": [2.0, 3.0], "count": 15200},
    {"range": [3.0, 4.0], "count": 22400},
    {"range": [4.0, 5.0], "count": 18700}
  ],
  "total": 249924,
  "mean": 8.2,
  "median": 7.1,
  "skew": 1.4,
  "kurtosis": 6.2
}
```

---

### 3.8 `GET /tensor/{tensor_name}`

Full detail for a single tensor. The tensor name is URL-encoded.

**Response:**
```json
{
  "tensor_name": "language_model.model.layers.42.mlp.gate_proj.weight",
  "layer": 42,
  "slot": 178340,
  "type": "mlp.gate_proj",
  "shape": [33792, 7168],
  "numel": 242074880,
  "frobenius": 354.8126,
  "spectral_norm": 13.3818,
  "effective_rank": 15.7137,
  "kurtosis": 2.1868,
  "sparsity": 0.2170,
  "percentiles": {
    "spectral_norm": {"p1": 11.2, "p5": 12.0, "p25": 12.8, "p50": 13.4, "p75": 14.1, "p95": 15.8, "p99": 17.2}
  }
}
```

---

### 3.9 `POST /upload`

Upload a new scan (zip of Weight Atlas output: TIFF, JSON, HTML, fingerprint).

**Request:** `multipart/form-data` with fields:

| Field | Type | Description |
|-------|------|-------------|
| `file` | binary | ZIP archive |
| `model_name` | string | Display name |
| `model_id` | string | Optional HuggingFace ID |
| `author` | string | Uploader name / handle |

**Response:**
```json
{
  "scan_id": "qwen3-27b-2026-08-16-d4e5f6",
  "status": "processing",
  "n_tensors": 249924,
  "url": "/scan/qwen3-27b-2026-08-16-d4e5f6"
}
```

---

## 4. Error Handling

All errors use a consistent JSON shape:

```json
{
  "error": {
    "code": 404,
    "type": "tensor_not_found",
    "message": "Tensor 'language_model.model.layers.99.mlp.gate_proj.weight' not found in scan 'k3-2026-08-15-a1b2c3'."
  }
}
```

| HTTP Code | Type | Example |
|-----------|------|---------|
| 400 | `invalid_param` | Malformed filter expression |
| 404 | `tensor_not_found` | Tensor name doesn't exist |
| 404 | `scan_not_found` | Scan ID doesn't exist |
| 409 | `model_mismatch` | `/delta` called with scans of different architectures |
| 413 | `payload_too_large` | Upload exceeds size limit |
| 429 | `rate_limited` | Too many requests |
| 500 | `internal_error` | Server-side failure |

---

## 5. Design Principles

1. **Small responses.** Every endpoint returns at most 500 rows. The LLM never needs to ingest 25 MB.
2. **Deterministic.** Same query, same response, every time. No timestamps in output (except `scan_timestamp` in `/meta`).
3. **Filterable, not searchable.** No free-text search. Structured filters only: layer, type, metric range.
4. **Metric-agnostic.** All endpoints accept any metric from the scan as a parameter. Adding a new metric requires no API change.
5. **Human + AI co-use.** The same API serves the web UI (for humans) and REST clients (for LLMs). The web UI adds visual context (TIFF, topographic renders) that the API doesn't need.
6. **Stateless.** No session, no auth for public scans. Uploads require a token.
7. **Progressive drilldown.** Start with `/meta` → `/summary` → `/anomalies` → `/query` → `/tensor/{name}`. Each step narrows focus.

---

## 6. Typical LLM Workflow

```
Step 1:  GET /meta
         → Model: Kimi K3, 93 layers, 896 experts, 249924 tensors

Step 2:  GET /summary?group_by=type&metric=spectral_norm
         → lm_head: 204.5, embed: 49.7, mlp.gate: 13.4, attn: 14.1

Step 3:  GET /anomalies?metric=spectral_norm&n=20
         → Top 20 outliers: mostly in layers 40–60, gate_proj and up_proj

Step 4:  GET /query?layer=42&type=mlp.gate_proj&sort=spectral_norm&order=desc&limit=10
         → 10 rows from layer 42, sorted by spectral norm

Step 5:  GET /tensor/language_model.model.layers.42.mlp.gate_proj.weight
         → Full detail: shape, all metrics, percentiles

Step 6:  GET /compare?a=layer:0&b=layer:92
         → Layer 0 has kurtosis 290.9 (layernorm), Layer 92 has 2.5

Step 7:  GET /delta?model_a=k3-base&model_b=k3-abliterated&n=20
         → Top 20 most-changed tensors after abliteration

→ LLM synthesises a report with findings, comparisons, and recommendations.
```

---

## 7. Storage & Performance

| Aspect | Approach |
|--------|----------|
| **Storage format** | SQLite (one DB file per scan) or Parquet. Indexed on `(scan_id, layer, type, tensor_name)`. |
| **In-memory** | Not needed. 249,924 rows fit in <10 MB SQLite. Queries are indexed, sub-millisecond. |
| **Upload** | ZIP → unpack → parse JSON → insert into DB → index. ~5–15 s for 250k tensors. |
| **API server** | FastAPI (async) or Flask. Single process is sufficient for <100 concurrent users. |
| **Scaling** | Multiple scans = multiple DB files. No cross-scan queries except `/delta`, which reads two DBs. |
| **Size** | 249,924 tensors × ~8 metrics × 8 bytes ≈ 16 MB raw. SQLite with indexes ≈ 25–35 MB per scan. |

---

## 8. Security

- Public reads: no auth required for published scans.
- Uploads: require a bearer token or OAuth2 client credential.
- Rate limiting: 60 req/min per IP for reads, 5 req/min for uploads.
- File validation: ZIP must contain a manifest JSON. Reject files > 500 MB.
- No PII in tensor names. Model names are user-supplied, displayed as-is.

---

## 9. Roadmap

| Phase | Scope |
|-------|-------|
| **v0.1** | `/meta`, `/summary`, `/query`, `/tensor/{name}` — core read API over one scan |
| **v0.2** | `/anomalies`, `/histogram`, `/compare` — analysis endpoints |
| **v0.3** | `/delta` — cross-scan comparison |
| **v0.4** | `/upload` — user-contributed scans, public gallery |
| **v0.5** | Web UI with embedded topographic renders + interactive filter panel |
| **v0.6** | LLM agent endpoint: `POST /agent/analyse` with natural-language query → structured answer |

---

*End of specification.*
