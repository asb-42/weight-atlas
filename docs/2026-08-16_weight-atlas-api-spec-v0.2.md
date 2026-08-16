# Weight Atlas — LLM Query API Specification

**Version:** 0.2 (draft)
**Date:** 2026-08-16
**Status:** Proposal

---

## 1. Purpose

Weight Atlas scans LLM weight files and extracts per-tensor statistics into a structured dataset. This API exposes that dataset through targeted query endpoints, enabling both human users (via web UI) and LLM agents (via REST) to analyse model weight topology without loading the full 25 MB table into context.

**Design principle:** The LLM is an analyst, not a reader. It asks targeted questions and receives 20–200 rows per response. Every response includes **baseline context** so the consumer can interpret values without a second call.

**Synthetic-user principle:** An agent cannot read this document. It reads `GET /api` and `GET /api/schema` — a machine-readable manifest of endpoints, metric vocabulary, taxonomy, and filter grammar, with one worked example per endpoint. Everything an agent needs to use the API safely is discoverable at runtime, in-band.

The full table remains available for human visual inspection (TIFF, HTML, topographic renders). The API is the machine-readable layer.

---

## 2. URL Structure

All endpoints are nested under the model resource:

```
GET  /api                                 → self-description (endpoints, metrics, taxonomy, grammar)
GET  /api/schema                          → machine-readable metric/type vocabulary
GET  /models                              → list all scans
GET  /model/{model_id}                    → scan metadata
GET  /model/{model_id}/summary            → model-wide aggregates
GET  /model/{model_id}/layer/{n}          → all tensors in one layer, with intra-layer comparison
GET  /model/{model_id}/anomalies          → statistically unusual tensors
GET  /model/{model_id}/query              → filtered, sorted, paginated tensors
GET  /model/{model_id}/compare            → two slices within one model
GET  /model/{model_id}/histogram          → distribution of a metric
GET  /model/{model_id}/tensor/{name}      → full detail for one tensor
GET  /model/{model_id}/delta?with={other} → cross-scan comparison
POST /upload                              → register a new scan
```

Base URL: `http://<host>:<port>/api`

All responses are `application/json`.

---

## 3. Data Model

### 3.0 Discovery (agent onboarding)

`GET /api` is the entry point for any synthetic consumer. It returns everything
needed to use the API without reading docs: the endpoint list, the metric
vocabulary (names + what they mean + typical ranges), the tensor-type
taxonomy, and the filter grammar.

```json
{
  "api_version": "0.2",
  "docs": "/docs",
  "endpoints": [
    {"method": "GET", "path": "/model/{model_id}/summary", "purpose": "model-wide aggregates", "params": ["group_by", "metrics"]},
    {"method": "GET", "path": "/model/{model_id}/anomalies", "purpose": "statistically unusual tensors", "params": ["metric", "threshold", "method", "n"]}
  ],
  "metrics": {
    "spectral_norm": {"description": "largest singular value", "min": 0, "expected_range": [2, 25]},
    "kurtosis": {"description": "excess kurtosis of the weight distribution", "min": 0, "expected_range": [0, 6]}
  },
  "types": ["embed_tokens", "lm_head", "input_layernorm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "post_attention_layernorm", "router"],
  "filter_grammar": {
    "layer": "42 | >=50 | 0:31 | 0,2,4",
    "type": "mlp.gate_proj | self_attn",
    "threshold": "p95 | p99 | p999 | 15.0",
    "slice": "layer:42.type:mlp.gate_proj"
  },
  "pagination": {"limit_max": 500, "offset": "int"}
}
```

`GET /api/schema` additionally returns the OpenAPI-style field schema for every
response body so a code-generating agent can type-check results.

### 3.1 Tensor Record

| Field | Type | Description |
|-------|------|-------------|
| `tensor_name` | string | Full dotted path, e.g. `language_model.model.layers.42.mlp.gate_proj.weight` |
| `layer` | int | Layer index (0-based). `-1` for non-layer tensors. |
| `slot` | string | Engine slot taxonomy from `map_name` (matches raster columns): `embed`, `lm_head`, `attn_q`, `attn_k`, `attn_v`, `attn_o`, `mlp_gate`, `mlp_up`, `mlp_down`, `norm_attn`, `norm_mlp`, `router`, `expert`, `other`, vision slots (`mm_projector`, …). This is the *authoritative* grouping — the human-readable dotted `type` below is a derived label. |
| `type` | string | Normalised tensor type: `embed_tokens`, `lm_head`, `input_layernorm`, `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`, `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`, `post_attention_layernorm`, `router`, `expert.{id}.gate_proj`, etc. Derived from the tensor name; the coarse `slot` is what the pipeline groups on. |
| `shape` | list[int] | Tensor dimensions. |
| `numel` | int | Total number of elements. |
| `frobenius` | float | Frobenius norm. |
| `spectral_norm` | float | Spectral norm (largest singular value). |
| `effective_rank` | float | Stable / effective rank (spectral entropy-based). |
| `stable_rank` | float | log1p((frobenius/spectral_norm)²) — the tint channel stat. |
| `kurtosis` | float | Excess kurtosis of the weight distribution. |
| `sparsity` | float | Fraction of weights below threshold. |

### 3.2 Scan Metadata

| Field | Type | Description |
|-------|------|-------------|
| `scan_id` | string | Unique identifier for this scan. |
| `model_name` | string | Display name, e.g. `Qwen3.8-27B Q4_K_M`. |
| `model_id` | string | Stable identifier used in URLs, e.g. `qwen3-27b-2026-08-16-d4e5f6`. |
| `huggingface_id` | string or null | Optional HF repo ID. |
| `n_tensors` | int | Total scanned tensors. |
| `n_layers` | int | Number of transformer layers. |
| `n_experts` | int | MoE experts (0 if dense). |
| `top_k` | int | Experts activated per token (0 if dense). |
| `hidden_dim` | int | Hidden size. |
| `arch` | string | `qwen3-dense`, `kimi-k3-moe`, etc. |
| `quantization` | string | `Q4_K_M`, `FP16`, etc. |
| `author` | string | Uploader. |
| `scan_timestamp` | string | ISO-8601 UTC. |
| `tool_version` | string | Weight Atlas engine version (from `fingerprint.json`). |

---

## 4. Baseline Context Convention

**Every response that returns tensor values includes a `_baseline` object** so the consumer can interpret magnitudes without a follow-up call.

```json
{
  "_baseline": {
    "model": "qwen3-27b-2026-08-16-d4e5f6",
    "n_tensors": 249924,
    "global": {
      "spectral_norm": {"mean": 8.2, "median": 7.1, "p95": 15.8, "p99": 22.4},
      "kurtosis":      {"mean": 1.9, "median": 1.4, "p95": 4.2,  "p99": 12.8}
    },
    "per_type": {
      "mlp.gate_proj": {"spectral_norm_mean": 13.4, "kurtosis_mean": 2.19},
      "self_attn.q_proj": {"spectral_norm_mean": 14.2, "kurtosis_mean": 1.8}
    }
  },
  "rows": [ ... ]
}
```

This means when an LLM sees `spectral_norm: 19.7` on a `gate_proj` tensor, it immediately knows that's 1.5× the type mean (13.4) and above the p95 (15.8) — no second call needed.

---

## 5. Endpoints

### 5.1 `GET /models`

**Purpose:** Top-level listing. The entry point for any consumer that doesn't yet know which models are available.

**Response:**
```json
{
  "n_scans": 4,
  "scans": [
    {
      "model_id": "kimi-k3-2026-08-15-a1b2c3",
      "model_name": "Kimi K3",
      "arch": "kimi-k3-moe",
      "n_tensors": 249924,
      "n_layers": 93,
      "quantization": "Q4_K_M",
      "scan_timestamp": "2026-08-15T22:00:00Z"
    },
    {
      "model_id": "qwen3-27b-2026-08-16-d4e5f6",
      "model_name": "Qwen3.8-27B",
      "arch": "qwen3-dense",
      "n_tensors": 249924,
      "n_layers": 32,
      "quantization": "Q4_K_M",
      "scan_timestamp": "2026-08-16T04:10:00Z"
    }
  ]
}
```

---

### 5.2 `GET /model/{model_id}`

**Purpose:** Full metadata for one scan.

**Response:**
```json
{
  "scan_id": "kimi-k3-2026-08-15-a1b2c3",
  "model_name": "Kimi K3",
  "huggingface_id": "moonshotai/kimi-k3",
  "n_tensors": 249924,
  "n_layers": 93,
  "n_experts": 896,
  "top_k": 16,
  "hidden_dim": 7168,
  "arch": "kimi-k3-moe",
  "quantization": "Q4_K_M",
  "author": "asb-42",
  "scan_timestamp": "2026-08-15T22:00:00Z",
  "tool_version": "0.2.0",
  "metrics": ["frobenius", "spectral_norm", "effective_rank", "kurtosis", "sparsity"],
  "baseline": {
    "spectral_norm": {"mean": 8.2, "median": 7.1, "p95": 15.8, "p99": 22.4, "max": 204.5},
    "kurtosis": {"mean": 1.9, "median": 1.4, "p95": 4.2, "p99": 12.8, "max": 290.9}
  }
}
```

---

### 5.3 `GET /model/{model_id}/summary`

**Purpose:** One call, entire model in context-size. The LLM's first substantive look.

**Query parameters:**

| Param | Values | Default |
|-------|--------|---------|
| `group_by` | `layer`, `type`, `none` | `type` |
| `metrics` | comma-separated list | all |

**Response (group_by=type, default metrics):**
```json
{
  "model_id": "qwen3-27b-2026-08-16-d4e5f6",
  "group": "type",
  "rows": [
    {
      "type": "lm_head",
      "n_tensors": 1,
      "spectral_norm": {"mean": 204.5, "p95": 204.5, "max": 204.5},
      "kurtosis": {"mean": 0.43, "max": 0.43},
      "note": "1.8× global mean — expected for output projection"
    },
    {
      "type": "embed_tokens",
      "n_tensors": 1,
      "spectral_norm": {"mean": 49.7, "p95": 49.7, "max": 49.7},
      "kurtosis": {"mean": 0.40, "max": 0.40},
      "note": "4.8× global mean — expected for input embedding"
    },
    {
      "type": "mlp.gate_proj",
      "n_tensors": 32,
      "spectral_norm": {"mean": 13.4, "p95": 15.2, "max": 18.1},
      "kurtosis": {"mean": 2.19, "p95": 3.8, "max": 5.2},
      "note": "1.6× global mean — typical FFN gate"
    },
    {
      "type": "self_attn.q_proj",
      "n_tensors": 32,
      "spectral_norm": {"mean": 14.2, "p95": 16.0, "max": 19.4},
      "kurtosis": {"mean": 1.8, "p95": 3.1, "max": 4.5},
      "note": "1.7× global mean — typical attention query"
    }
  ],
  "anomaly_count": 12,
  "top_outliers": [
    {"tensor_name": "language_model.lm_head.weight", "spectral_norm": 204.5, "zscore": 28.1},
    {"tensor_name": "language_model.model.layers.42.mlp.gate_proj.weight", "spectral_norm": 19.7, "zscore": 3.8}
  ]
}
```

The `note` field is server-generated and gives the LLM (or human) an immediate interpretation anchor.

---

### 5.4 `GET /model/{model_id}/layer/{n}`

**Purpose:** All tensors in one layer, with intra-layer comparison. The semantic unit of a transformer.

**Path parameter:** `n` — layer index (0-based). Use `-1` for non-layer tensors (embed, lm_head, final norm).

**Response:**
```json
{
  "model_id": "qwen3-27b-2026-08-16-d4e5f6",
  "layer": 42,
  "n_tensors": 10,
  "layer_stats": {
    "spectral_norm_mean": 13.8,
    "spectral_norm_median": 13.5,
    "spectral_norm_max": 19.7,
    "kurtosis_mean": 2.4,
    "kurtosis_max": 5.2
  },
  "global_context": {
    "spectral_norm_mean_all_layers": 8.2,
    "this_layer_vs_global": "1.7×"
  },
  "rows": [
    {
      "tensor_name": "language_model.model.layers.42.input_layernorm.weight",
      "type": "input_layernorm",
      "shape": [7168],
      "spectral_norm": 14.04,
      "kurtosis": 290.94,
      "vs_layer_mean": "1.0×",
      "vs_global_mean": "1.7×",
      "flag": "kurtosis outlier (p99.97)"
    },
    {
      "tensor_name": "language_model.model.layers.42.mlp.gate_proj.weight",
      "type": "mlp.gate_proj",
      "shape": [33792, 7168],
      "spectral_norm": 19.7,
      "kurtosis": 5.2,
      "vs_layer_mean": "1.4×",
      "vs_global_mean": "2.4×",
      "flag": "spectral_norm p97"
    },
    {
      "tensor_name": "language_model.model.layers.42.self_attn.o_proj.weight",
      "type": "self_attn.o_proj",
      "shape": [7168, 7168],
      "spectral_norm": 12.1,
      "kurtosis": 1.6,
      "vs_layer_mean": "0.9×",
      "vs_global_mean": "1.5×",
      "flag": null
    }
  ]
}
```

**Key design point:** Every row has `vs_layer_mean` and `vs_global_mean` — the LLM never has to do its own division.

---

### 5.5 `GET /model/{model_id}/anomalies`

**Purpose:** Statistically unusual tensors. The LLM's primary "what's weird here" query.

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `metric` | string | `spectral_norm` | Metric to detect on |
| `threshold` | string | `p99` | Cutoff: `p95`, `p99`, `p999`, or explicit value like `15.0` |
| `method` | string | `quantile` | `quantile`, `IQR`, `zscore` |
| `n` | int | 50 | Max results |
| `direction` | string | `both` | `high`, `low`, `both` |
| `type` | string | all | Filter to one tensor type |
| `layer_range` | string | all | e.g. `0:31` or `>=50` |
| `fields` | string | all | Comma-separated subset of columns (token budget) |

**Response:**
```json
{
  "model_id": "qwen3-27b-2026-08-16-d4e5f6",
  "metric": "spectral_norm",
  "threshold": {"method": "quantile", "value": "p99", "resolved": 22.4},
  "n_anomalies": 12,
  "rows": [
    {
      "tensor_name": "language_model.lm_head.weight",
      "layer": -1,
      "type": "lm_head",
      "spectral_norm": 204.5,
      "zscore": 28.1,
      "percentile": 99.9996,
      "vs_global_mean": "25×",
      "context": "Output projection — high norm expected, but 25× is extreme"
    },
    {
      "tensor_name": "language_model.model.layers.42.mlp.gate_proj.weight",
      "layer": 42,
      "type": "mlp.gate_proj",
      "spectral_norm": 19.7,
      "zscore": 3.8,
      "percentile": 97.2,
      "vs_global_mean": "2.4×",
      "context": "gate_proj in mid-layer — above p95 for this type"
    }
  ]
}
```

---

### 5.6 `GET /model/{model_id}/query`

**Purpose:** General-purpose filtered, sorted, paginated tensor list.

**Query parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `layer` | int or range | `42`, `>=50`, `0:31` |
| `type` | string | `mlp.gate_proj`, `self_attn`, `embed`, `lm_head` |
| `metric` | string | Sort by this metric |
| `order` | `asc` / `desc` | Sort direction |
| `fields` | string | Comma-separated subset of metrics/columns to return (token budget) |
| `limit` | int | Max rows (default 50, max 500) |
| `offset` | int | Pagination |
| `min` | float | Minimum value for sorted metric |
| `max` | float | Maximum value for sorted metric |

`fields` trims each row to the requested columns — for token-constrained
consumers, `fields=tensor_name,spectral_norm,kurtosis` is far cheaper than
always receiving the full record.

**Response:**
```json
{
  "model_id": "qwen3-27b-2026-08-16-d4e5f6",
  "n_results": 32,
  "offset": 0,
  "limit": 5,
  "has_more": true,
  "next_offset": 5,
  "_baseline": {"type": "mlp.gate_proj", "spectral_norm_mean": 13.4, "spectral_norm_p95": 15.2},
  "rows": [
    {
      "tensor_name": "language_model.model.layers.91.mlp.gate_proj.weight",
      "layer": 91,
      "type": "mlp.gate_proj",
      "shape": [33792, 7168],
      "spectral_norm": 18.1,
      "kurtosis": 4.8,
      "effective_rank": 14.2,
      "sparsity": 0.198
    }
  ]
}
```

`has_more`/`next_offset` give the consumer a cheap continuation contract: page
by offsetting until `has_more` is false, or ask for more with a larger `limit`
(capped at 500).

---

### 5.7 `GET /model/{model_id}/compare?a={slice}&b={slice}`

**Purpose:** Side-by-side comparison of two slices within one model.

**Slice syntax:** `layer:0`, `type:mlp.gate_proj`, `layer:42.type:mlp.gate_proj` (dot-concatenated predicates, each `key:value`).

**Query parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `a` | slice | First slice (required) |
| `b` | slice | Second slice (required) |
| `metrics` | string | Comma-separated subset (default all) |
| `fields` | string | Comma-separated subset of output fields |

**Response:**
```json
{
  "model_id": "qwen3-27b-2026-08-16-d4e5f6",
  "a": {"slice": "layer:0", "n_tensors": 10, "spectral_norm_mean": 12.3, "kurtosis_max": 290.9},
  "b": {"slice": "layer:31", "n_tensors": 10, "spectral_norm_mean": 14.8, "kurtosis_max": 2.5},
  "delta": {
    "spectral_norm": {"a": 12.3, "b": 14.8, "abs": 2.5, "pct": 20.3, "interpretation": "Last layer has 20% higher mean spectral norm — typical depth gradient"},
    "kurtosis": {"a": 290.9, "b": 2.5, "abs": -288.4, "pct": -99.1, "interpretation": "First layer dominated by layernorm outlier (290.9)"}
  }
}
```

---

### 5.8 `GET /model/{model_id}/delta?with={other_model_id}`

**Purpose:** Cross-scan comparison. The "what did abliteration change" endpoint.

Two tiers (server picks the richest available):

1. **Weight-space delta** (both scans expose paired weights / a
   `paired`-preset run): returns the edit-preset fields — per-tensor `rel_l2`,
   `cos`, `dspec`, `delta_stable_rank`, `spectral_share` — plus the
   `edit_signature` (classification `low_rank_localized`/`diffuse`/…, bands,
   `hotspot_ranking_rel_l2`). This is the honest answer for "what changed in
   the weights".
2. **Statistic diff** (fingerprints only): diffs the recorded `metric`
   (default `spectral_norm`) per tensor, as below.

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `with` | string | required | Other scan's model_id |
| `metric` | string | `spectral_norm` | Metric to diff (tier-2 only) |
| `n` | int | 50 | Most changed tensors |
| `min_change_pct` | float | 0 | Minimum % change |
| `fields` | string | all | Comma-separated subset of columns (token budget) |

**Response (tier 1, weight-space):**
```json
{
  "model_a": "kimi-k3-2026-08-15-a1b2c3",
  "model_b": "kimi-k3-abliterated-2026-08-16-x1y2z3",
  "tier": "weight_space",
  "n_compared": 249924,
  "edit_signature": {
    "classification": "low_rank_localized",
    "stats": {"median_rel_l2": 0.42, "median_delta_stable_rank": 1.1},
    "bands": [{"start_layer": 40, "end_layer": 60, "slots": ["mlp_down"]}],
    "hotspot_ranking_rel_l2": [
      {"layer": 42, "slot": "mlp.gate_proj", "rel_l2": 0.87, "name_a": "…", "name_b": "…"}
    ]
  },
  "noise_floor": {"policy": "identical", "warning": null},
  "rows": [
    {
      "tensor_name": "language_model.model.layers.42.mlp.gate_proj.weight",
      "layer": 42,
      "rel_l2": 0.87,
      "cos": 0.42,
      "dspec": 11.3,
      "delta_stable_rank": 1.05,
      "spectral_share": 0.94,
      "context": "Top rel-L2 change — within the low-rank localized band (layers 40-60, mlp_down)"
    }
  ]
}
```

**Response (tier 2, statistic diff):**
```json
{
  "model_a": "kimi-k3-2026-08-15-a1b2c3",
  "model_b": "kimi-k3-abliterated-2026-08-16-x1y2z3",
  "tier": "statistic_diff",
  "metric": "spectral_norm",
  "n_compared": 249924,
  "n_changed_above_5pct": 3421,
  "summary": {
    "mean_change_pct": -12.4,
    "max_change_pct": -84.3,
    "most_affected_type": "mlp.gate_proj",
    "most_affected_layer_range": "40-60"
  },
  "rows": [
    {
      "tensor_name": "language_model.model.layers.42.mlp.gate_proj.weight",
      "a": 13.4,
      "b": 2.1,
      "abs_change": -11.3,
      "pct_change": -84.3,
      "context": "One of the largest reductions — gate_proj in the hot zone (layers 40-60)"
    }
  ]
}
```

---

### 5.9 `GET /model/{model_id}/histogram?metric={name}&bins={n}`

**Purpose:** Distribution shape. Useful for CDF, mode, skew analysis.

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `metric` | string | `spectral_norm` | Metric to histogram |
| `bins` | int | 10 | Number of bins |
| `log` | bool | `false` | Log-scale the x axis (heavy-tailed metrics like `kurtosis` render better) |
| `type` | string | all | Filter to one tensor type |
| `layer_range` | string | all | Filter to a layer range |
| `density` | bool | `false` | Normalise counts to probability density |

**Response:**
```json
{
  "model_id": "qwen3-27b-2026-08-16-d4e5f6",
  "metric": "spectral_norm",
  "bins": [
    {"range": [0.0, 2.0], "count": 1204, "pct": 0.5},
    {"range": [2.0, 4.0], "count": 8930, "pct": 3.6},
    {"range": [4.0, 6.0], "count": 22400, "pct": 9.0},
    {"range": [6.0, 8.0], "count": 45200, "pct": 18.1},
    {"range": [8.0, 10.0], "count": 62800, "pct": 25.1},
    {"range": [10.0, 12.0], "count": 58400, "pct": 23.4},
    {"range": [12.0, 15.0], "count": 31200, "pct": 12.5},
    {"range": [15.0, 20.0], "count": 8200, "pct": 3.3},
    {"range": [20.0, 50.0], "count": 2100, "pct": 0.8},
    {"range": [50.0, 210.0], "count": 390, "pct": 0.2}
  ],
  "total": 249924,
  "distribution": {"mean": 8.2, "median": 7.1, "skew": 1.4, "kurtosis": 6.2},
  "shape_description": "Right-skewed, unimodal with mode at 8-10, long tail to 204.5 (lm_head)"
}
```

The `shape_description` is server-generated — the LLM doesn't need to interpret raw bins.

---

### 5.10 `GET /model/{model_id}/tensor/{tensor_name}`

**Purpose:** Full detail for a single tensor. The drilldown endpoint.

**Path:** `tensor_name` is URL-encoded. Dots are preserved.

**Response:**
```json
{
  "model_id": "qwen3-27b-2026-08-16-d4e5f6",
  "tensor_name": "language_model.model.layers.42.mlp.gate_proj.weight",
  "layer": 42,
  "slot": "mlp_gate",
  "type": "mlp.gate_proj",
  "shape": [33792, 7168],
  "numel": 242074880,
  "metrics": {
    "frobenius": 354.8126,
    "spectral_norm": 13.3818,
    "stable_rank": 1.9023,
    "effective_rank": 15.7137,
    "kurtosis": 2.1868,
    "sparsity": 0.2170
  },
  "context": {
    "vs_type_mean": {"spectral_norm": "1.0×", "kurtosis": "1.0×"},
    "vs_global_mean": {"spectral_norm": "1.6×", "kurtosis": "1.15×"},
    "percentile_in_model": {"spectral_norm": 52.3, "kurtosis": 48.1}
  },
  "interpretation": "Typical gate_proj tensor. No anomalies. Slightly above global mean in spectral norm (expected for FFN)."
}
```

---

### 5.11 `POST /upload`

**Purpose:** Register a new scan (ZIP of Weight Atlas output).

**Request:** `multipart/form-data`

| Field | Required | Description |
|-------|----------|-------------|
| `file` | yes | ZIP archive (TIFF, JSON, HTML, fingerprint) |
| `model_name` | yes | Display name |
| `author` | yes | Uploader |
| `huggingface_id` | no | HF repo ID |

**Response:**
```json
{
  "model_id": "qwen3-27b-2026-08-16-d4e5f6",
  "status": "ready",
  "n_tensors": 249924,
  "url": "/api/model/qwen3-27b-2026-08-16-d4e5f6"
}
```

---

## 6. Error Handling

```json
{
  "error": {
    "code": 404,
    "type": "model_not_found",
    "message": "Model 'foo-bar' not found. Available: kimi-k3-2026-08-15, qwen3-27b-2026-08-16",
    "hint": "GET /models to list available scans"
  }
}
```

| HTTP | Type | Example |
|------|------|---------|
| 400 | `invalid_param` | Malformed filter expression |
| 404 | `model_not_found` | model_id doesn't exist |
| 404 | `layer_not_found` | Layer index out of range |
| 404 | `tensor_not_found` | Tensor name doesn't exist in this scan |
| 409 | `arch_mismatch` | `/delta` called across incompatible architectures |
| 413 | `payload_too_large` | Upload > 500 MB |
| 429 | `rate_limited` | Too many requests |
| 500 | `internal_error` | Server-side failure |

---

## 7. Design Principles

1. **Small responses.** Max 500 rows per call. The LLM never ingests 25 MB.
2. **Baseline context in every response.** Every tensor value comes with `vs_layer_mean`, `vs_type_mean`, `vs_global_mean` — the consumer never does its own division.
3. **Server-side interpretation.** Where possible, the response includes a one-line `note` or `interpretation` field. The LLM verifies rather than constructs.
4. **Resource-nested URLs.** `/model/{id}/...` — clear ownership, easy to extend, RESTful.
5. **Deterministic.** Same query, same response. No timestamps in analytical output.
6. **Filterable, not searchable.** Structured filters only. No free-text.
7. **Metric-agnostic.** All endpoints accept any metric. Adding a new metric requires no API change.
8. **Human + AI co-use.** Same API serves web UI and REST clients.
9. **Stateless.** No session. Public reads. Token-gated uploads.
10. **Progressive drilldown.** `/models` → `/model/{id}` → `/summary` → `/layer/{n}` → `/tensor/{name}`. Each step narrows focus.
11. **Discoverable in-band.** `GET /api` and `GET /api/schema` describe every endpoint, metric, type and filter grammar at runtime, so an agent never needs out-of-band docs.
12. **Token budget is a first-class param.** Every list endpoint supports `fields` to trim columns. Interpreters on a budget drop `context`/`note` fields and keep the numbers.
13. **Weight-space honesty.** Cross-scan `/delta` prefers the real weight-space paired metrics (`rel_l2`, `dspec`, edit classification) over statistic diffs whenever paired artefacts exist — statistic diffs of a *quantized* scan can mislead.

---

## 8. Typical LLM Workflow

```
Step 1:  GET /models
         → 4 scans available: kimi-k3, qwen3-27b, ...

Step 2:  GET /model/qwen3-27b-2026-08-16-d4e5f6
         → 32 layers, dense, Q4_K_M, baseline: spectral_norm mean 8.2, p99 22.4

Step 3:  GET /model/qwen3-27b-2026-08-16-d4e5f6/summary
         → 10 types, lm_head=204.5 (25× mean), gate_proj=13.4 (1.6×), 12 anomalies

Step 4:  GET /model/qwen3-27b-2026-08-16-d4e5f6/anomalies?metric=spectral_norm
         → Top 12 outliers, each with zscore, percentile, vs_global_mean

Step 5:  GET /model/qwen3-27b-2026-08-16-d4e5f6/layer/42
         → 10 tensors, layer mean 13.8 (1.7× global), gate_proj flagged at 19.7

Step 6:  GET /model/qwen3-27b-2026-08-16-d4e5f6/tensor/language_model.model.layers.42.mlp.gate_proj.weight
         → Full detail, percentiles, interpretation: "Slightly above mean, no anomaly"

Step 7:  GET /model/kimi-k3-base/delta?with=kimi-k3-abliterated
         → 3421 tensors changed >5%, most affected: gate_proj in layers 40-60

→ LLM synthesises a report with findings, comparisons, and recommendations.
```

---

## 9. Storage & Performance

| Aspect | Approach |
|--------|----------|
| **Storage** | SQLite (one DB file per scan). Indexed on `(model_id, layer, type, tensor_name)`. |
| **Size** | 249,924 tensors × 8 metrics ≈ 16 MB raw. SQLite + indexes ≈ 25-35 MB per scan. |
| **Query speed** | Sub-millisecond for indexed lookups. Full-table scans < 50 ms. |
| **Upload** | ZIP → unpack → parse JSON → insert → index. 5-15 s for 250k tensors. |
| **Server** | FastAPI (async) or Flask. Single process sufficient for < 100 concurrent users. |
| **Scaling** | N scans = N DB files. `/delta` reads two. No cross-DB joins needed. |

---

## 10. Security

- Public reads: no auth for published scans.
- Uploads: bearer token or OAuth2.
- Rate limiting: 60 req/min/IP reads, 5 req/min uploads.
- File validation: ZIP must contain manifest JSON. Reject > 500 MB.
- No PII in tensor names. Model names are user-supplied, displayed as-is.

---

## 11. Roadmap

| Phase | Scope |
|-------|-------|
| **v0.1** | `/models`, `/model/{id}`, `/summary`, `/query`, `/tensor/{name}`, `GET /api` — core read API + discovery |
| **v0.2** | `/layer/{n}` with intra-layer comparison, `/anomalies` with p99 default, `/histogram`, `/compare`, `fields` params, pagination `has_more` |
| **v0.3** | `/delta` — cross-scan comparison (weight-space tier via paired/edit artefacts) |
| **v0.4** | `/upload` — user-contributed scans, public gallery |
| **v0.5** | Web UI with embedded topographic renders + interactive filter panel |
| **v0.6** | `POST /model/{id}/analyse` — natural-language query → structured answer (server-side LLM synthesis) |

---

## 12. Changelog

### v0.2 → v0.1
- All endpoints nested under `/model/{id}/`
- Added `GET /models` (top-level listing)
- Added `GET /model/{id}/layer/{n}` with intra-layer comparison
- Added baseline context convention (`vs_layer_mean`, `vs_global_mean`, `vs_type_mean` in all responses)
- Added server-side `interpretation` / `note` / `context` fields
- `threshold=p99` as default for `/anomalies` (simpler than IQR)
- Renamed `/delta` to `/model/{id}/delta?with={other}` (cleaner REST semantics)
- Added `type` field to tensor records (was implicit)
- Added `huggingface_id` to metadata

### v0.2 review (synthetic-user pass, 2026-08-16)
- Added `GET /api` (in-band discovery: endpoints, metrics, taxonomy, grammar) and `GET /api/schema` (field-level schema) — agents must not need out-of-band docs.
- `slot` is now the engine's authoritative raster slot string (embed/lm_head/attn_q/…); `type` is the human-readable derived label. Added `stable_rank` to the record.
- Fixed `waste_version` → `tool_version`.
- `/delta` is now tiered: prefers weight-space paired artefacts (rel_l2/cos/dspec/`edit_signature` classification, bands) when they exist, falls back to statistic diff.
- Every list endpoint accepts `fields` (column subset for token budget); `/query` adds `has_more`/`next_offset` continuation.
- `/histogram` adds `log`, `type`, `layer_range`, `density` params.
- `/compare` documents the slice grammar and `metrics`/`fields` params.
- Design principles extended with in-band discovery, token-budget, and weight-space honesty.

---

*End of specification.*
