# Code Review — 2026-08-26

Deep-dive review of `src/weight_atlas` (~13k LoC): api (jobs/routes/query),
scan/cli, core, compare, render, paired, loaders, fields, stats, activity.
Scope: architecture/design, security, performance, concurrency, resilience.
Formatting, naming, and docstring nits deliberately excluded.

---

## 1. Executive Summary

The codebase is architecturally sound where it matters most: the plugin
registry is clean, determinism contracts are honored end-to-end, the Blender
subprocess handling is textbook (argv-list, timeout, traceback sniffing,
temp-dir hygiene), and the NaN-means-absent discipline holds across the
pipeline. However, there are three production-grade defects: a **wrong Q8_K
dequant layout** that corrupts or crashes scans of Q8_K-quantized models, a
**misaligned cosine similarity** in the compare engine that silently reports
wrong headline numbers whenever the two fields have different NaN footprints,
and a **paired-pipeline memory blowup** that accumulates both models fully in
RAM. On top of that, `serve` defaults to `0.0.0.0` with no authentication and
an inactive path allowlist, which turns the web UI into an unauthenticated
LAN-accessible file-scan-and-serve service. None of these are hard to fix;
all fixes below are local.

---

## 2. Critical & High Severity Issues (Must Fix)

### 2.1 Q8_K dequantization uses a fabricated block layout — Security/Data Corruption

- **Location:** `src/weight_atlas/loaders/gguf_dequant.py:311-325`
  (`_dequant_q8_k`), dispatch at `gguf_dequant.py:149-202`
- **The Problem:** The implementation assumes a 258-byte Q8_K block
  (`[f16 scale][256×int8]`). The real llama.cpp/GGUF Q8_K block is **292
  bytes**: `[f32 d: 4B][256×int8 qs][16×int16 bsums: 32B]`
  (`gguf.GGML_QUANT_SIZES[Q8_K] = (256, 292)` in this repo's own venv). Two
  independent errors: wrong total block size *and* f16 vs f32 scale. For real
  tensors the element-count mismatch usually surfaces as a confusing
  `ValueError` deep in `reshape`; for payloads where block counts coincidentally
  agree it silently decodes garbage into fingerprints. Q8_K is advertised in
  `SUPPORTED_TYPES`, so any user scanning a Q8_K model gets corrupted
  statistics with no diagnostic. Ironically, `_dequant_with_gguf` already
  derives layouts authoritatively from `GGML_QUANT_SIZES` — Q8_K just isn't
  routed to it.
- **The Fix:** Route Q8_K through the gguf library (it implements Q8_K), or if
  the pure-numpy path must stay, implement the true layout with truncation
  guards:

```python
_GGUF_ONLY = {
    GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_1,
    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_K,          # <-- real layout lives in gguf; stop hand-rolling it
    GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, GGML_TYPE_MXFP4, GGML_TYPE_NVFP4,
}
```

  If the numpy fallback must remain (gguf-less installs), pin it to reality:

```python
def _dequant_q8_k(data: bytes) -> np.ndarray:
    """Q8_K: [f32 scale:4B][256 x int8][16 x int16 bsums:32B] = 292B/block."""
    block_size = 292
    if len(data) % block_size:
        raise ValueError(f"truncated Q8_K payload: {len(data)} % {block_size} != 0")
    d = np.frombuffer(data, np.uint8).reshape(-1, block_size)
    scale = np.ascontiguousarray(d[:, :4]).view("<f4").astype(np.float32)
    qs = np.ascontiguousarray(d[:, 4:260]).view(np.int8).astype(np.float32)
    return (qs * scale[:, None]).ravel()
```

  Add a fixture quantized by `gguf.quants.quantize(data, Q8_K)` (the pattern
  `tests/test_gguf.py` already uses for other types) — today Q8_K has zero test
  coverage.

### 2.2 cosine_sim compares misaligned vectors when NaN footprints differ — Data Corruption

- **Location:** `src/weight_atlas/compare/delta.py:70-74` (independent masks)
  and `delta.py:132-150` (`_compute_cosine_sim` zero-padding)
- **The Problem:** `raw_flat_a`/`raw_flat_b` are masked **independently**, so
  whenever A and B have different NaN positions — guaranteed in aligned mode
  where the narrower field is padded with NaN columns (`align.py:187-188`),
  and common in strict mode with missing slots — the two flats have different
  lengths and different element orderings. `_compute_cosine_sim` then zero-pads
  the shorter and takes a row-major dot product: every element after the first
  divergent NaN position is dotted against the **wrong element**. The result
  lands in `compare_summary.json` as the headline similarity metric, plausibly
  close to 1.0 because both prefixes are large. Note the internal
  inconsistency: `_compute_rel_l2` (lines 107-129) correctly intersects both
  masks — only cosine is broken.
- **The Fix:** Intersect the masks exactly like `_compute_rel_l2`; the padding
  branch disappears entirely:

```python
# delta.py, compute_compare_summary metric section
both = np.isfinite(data_a) & np.isfinite(data_b)
cosine_sim = _compute_cosine_sim(
    data_a[both].ravel(),   # same positions, same length — alignment preserved
    data_b[both].ravel(),
)
```

```python
def _compute_cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of position-aligned finite elements."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
```

### 2.3 Paired pipeline holds both full models in RAM until completion — Performance/Memory

- **Location:** `src/weight_atlas/paired/paired.py:956-1002` (`_one` never
  clears; bulk clear only at lines 999-1002)
- **The Problem:** `TensorHandle.load()` memoizes its float32 payload
  (`core/types.py:45-50`). `run_paired` opens both models and processes all
  pairs without ever clearing handles mid-run; the clear loop runs **after**
  the whole thread pool finishes. Since every pair touches both sides, by the
  end of measurement the union of all paired tensors from **both models** is
  resident simultaneously (~4 bytes/param × 2). Two 7B snapshots ≈ ~56 GB
  before the first clear executes; GGUF shared-expert parents (released via
  `on_clear`) also stay resident. `scan.py:179-185` clears per-tensor — paired
  is the outlier, contradicting the module's "memory stays bounded" claim.
- **The Fix:** Clear inside the worker. Keys are unique per side and GGUF's
  refcounted parent release makes this safe:

```python
def _one(pair: tuple[TensorHandle, TensorHandle]) -> TensorImpact:
    ha, hb = pair
    try:
        return _pair_metrics(ha, hb, spec, ref_side=ref_side,
                             compute_spectrum=compute_spectrum,
                             want_u1=want_u1, chunk_size=chunk_size)
    finally:
        ha.clear()
        hb.clear()
```

  Peak RAM drops to `jobs_n × 2` tensors plus one shared GGUF parent per
  concurrently processed expert family. Keep the trailing clear loop for
  skipped handles.

### 2.4 Unauthenticated LAN exposure by default: `serve` binds 0.0.0.0 with inactive path allowlist — Security

- **Location:** `src/weight_atlas/cli.py:106-109` (`--host` default
  `0.0.0.0`), `api/main.py:26-51` (`model_roots=None` default),
  `api/routes.py:40-59` (`_require_allowed` no-op when roots is None)
- **The Problem:** Out of the box, `weight-atlas serve` exposes, with **no
  authentication and no CSRF protection**, on all interfaces:
  - `POST /api/jobs` → scans **any process-readable file path** (arbitrary
    absolute path accepted when `model_roots is None`) and publishes derived
    artefacts.
  - `POST /api/import` + `/api/artefacts/{id}/{path}` → imports any directory
    containing `fingerprint.json` and serves `.json/.txt/.csv/.npy/...` files
    from it — a file-exfil channel bounded only by the fingerprint.json
    requirement.
  - `POST /api/jobs/{id}/render/{renderer}` → spawns Blender subprocesses per
    request (resource-exhaustion DoS).
  
  The stderr warning in `_cmd_serve` is honest but doesn't change the default.
  Even bound to loopback, the absence of CSRF tokens / Host validation means a
  malicious web page can drive state-changing POSTs against
  `http://localhost:8000` via DNS rebinding or plain form POSTs.
- **The Fix:** Make the safe configuration the default and fail closed:

```python
# cli.py — serve subparser
serve.add_argument("--host", default="127.0.0.1",
                   help="Interface to bind (default: 127.0.0.1 localhost-only; "
                        "pass 0.0.0.0 to expose to the LAN)")
```

```python
# main.py — refuse unsafe combinations instead of warning-by-docstring
if model_roots is None:
    # Loopback-only default keeps this safe; explicit opt-in required beyond it.
    pass  # allowed: local trust domain
# When binding non-loopback, require an allowlist at the factory level:
def create_app(..., bind_host: str = "127.0.0.1", ...):
    if bind_host not in ("127.0.0.1", "::1", "localhost") and model_roots is None:
        raise RuntimeError(
            "create_app(bind_host=%r) without model_roots would expose "
            "arbitrary local paths to the network; pass an explicit allowlist."
            % bind_host
        )
```

  Minimal hardening for the CSRF/DNS-rebinding angle regardless of host:
  reject requests whose `Host` header is not the expected origin (FastAPI
  middleware) — browsers always send it, cross-site rebinding does not.

### 2.5 bfloat16 capture mode cannot work; global torch state leaks process-wide — Resilience

- **Location:** `src/weight_atlas/activity/capture.py:61-71` (global mutations),
  `capture.py:83-97` (bf16 crash), `cli.py:94` (`--dtype bfloat16` advertised)
- **The Problem:**
  1. With `dtype="bfloat16"`, hook tensors are bf16; `.numpy()` raises
     `TypeError: Got unsupported ScalarType BFloat16` — NumPy has no bf16. The
     advertised option fails on the first captured layer.
  2. `torch.set_num_threads(1)`, `torch.use_deterministic_algorithms(True)`,
     `torch.manual_seed(...)` mutate process-global state and are never
     restored. On CUDA ≥ 10.2, deterministic mode additionally makes the first
     cuBLAS matmul raise unless `CUBLAS_WORKSPACE_CONFIG=:4096:8` was set
     before torch import — i.e. `--device cuda`, a primary intended mode,
     fails with an unrelated-looking error.
  3. Hook unwrapping `output[0] if isinstance(output, tuple) else output`
     leaves modern `transformers.ModelOutput` returns unwrapped (not a tuple),
     crashing at `hidden ** 2`.
- **The Fix:**

```python
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must precede torch import
...
old_threads = torch.get_num_threads()
old_det = torch.are_deterministic_algorithms_enabled()
old_rng = torch.random.get_rng_state()
try:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    ...  # capture
finally:
    torch.set_num_threads(old_threads)
    torch.use_deterministic_algorithms(old_det)
    torch.random.set_rng_state(old_rng)
```

```python
def _first(hidden_out: Any) -> Any:
    if isinstance(hidden_out, tuple):
        return hidden_out[0]
    if hasattr(hidden_out, "to_tuple"):       # transformers ModelOutput
        return hidden_out.to_tuple()[0]
    return hidden_out

rms = torch.sqrt((hidden.float() ** 2).mean(dim=-1)).detach().cpu().numpy()  # .float() upcasts bf16
```

### 2.6 Degeneration warnings are computed after fingerprint.json is written — Data Integrity

- **Location:** `src/weight_atlas/scan.py:228-231` (write) vs
  `scan.py:423-432` (warnings merged into the in-memory dict afterwards)
- **The Problem:** `json.dump(fingerprint, ...)` runs at line 230; the
  degeneration report is merged into `fingerprint["warnings"]` at line 432 —
  after the file was flushed. The warnings channel documented for fingerprints
  therefore **never fires**: degraded/degenerate-field diagnostics are computed
  and silently dropped. Verified: `scan.py` contains exactly one
  `json.dump(fingerprint...)`, nothing rewrites the file afterwards.
- **The Fix:** Compute diagnostics before serialization, or rewrite after
  merging:

```python
# move the degeneration check ahead of the write (preferred: single write)
_report(0.42, "Building fingerprint...")
fingerprint = _build_fingerprint(stats, spec, loader_id, handles)
degen_warnings = _collect_degeneration_warnings(out, spec)   # reads raw tifs once written
if degen_warnings:
    fingerprint["warnings"] = fingerprint.get("warnings", []) + degen_warnings
with open(fp_path, "w") as f:
    json.dump(fingerprint, f, indent=2, sort_keys=True)
    f.write("\n")
```

  (If reordering is undesirable because TIFFs don't exist yet, do a second
  atomic write of `fp_path` after line 432 — but prefer the single-write order.)

---

## 3. Architectural & Design Improvements (Should Fix)

### 3.1 Compare orchestration duplicated between CLI and API — and already behaviorally diverged — Architecture

- **Location:** `cli.py:204-362` (`_cmd_compare`) vs
  `api/jobs.py:461-599` (`JobQueue._run_compare`)
- **The Problem:** ~150 lines of identical logic (channel discovery, delta
  TIFFs, M9 artefacts, summary assembly) exist twice and have **already
  drifted**: the CLI passes `row_labels_a/b` (real layer labels from
  fingerprints) and `noise_floor_dir`; the API worker passes neither. Identical
  inputs compared via CLI vs web UI therefore produce different summaries and
  renders — quietly violating the project's own parity expectations. Any future
  fix (e.g. §2.2) must be applied twice.
- **The Fix:** Extract one service function; make both entrypoints thin shells:

```python
# weight_atlas/compare/pipeline.py
def run_compare(dir_a: Path, dir_b: Path, out: Path, spec: AtlasSpec, *,
                mode: str = "strict", interp: str | None = None,
                noise_floor_dir: Path | None = None,
                progress: ProgressFn | None = None) -> list[Path]:
    ...  # single copy of the current _cmd_compare body

# cli.py
artefacts = run_compare(args.dir_a, args.dir_b, args.out, spec,
                        mode=args.mode, interp=args.interp,
                        noise_floor_dir=args.noise_floor)

# jobs.py _run_compare
artefacts = run_compare(dir_a, dir_b, out, spec, mode=mode, interp=interp,
                        progress=progress_cb)
# (then decide explicitly whether the API should also forward row labels /
#  noise floor — today's silent divergence becomes a visible choice)
```

### 3.2 Query engine rebuilds all tensor records on every request, on the event loop — Architecture/Performance

- **Location:** `src/weight_atlas/api/query.py:215-242` (`_build_records`,
  called by every `*_body` function); `query_routes.py` endpoints are `async def`
- **The Problem:** The parsed fingerprint is cached (`_scan_cache`, keyed on
  `(path, mtime_ns, size)` — good design), but records are rebuilt per request:
  `map_name()` alone runs ~50 regex `search()` calls per tensor name, so a
  74k-tensor model costs millions of regex evaluations **per API call**, plus a
  full sort — synchronously inside `async def` handlers, freezing the event
  loop that also serves the UI polling every 2 s. The LLM query API is exactly
  the workload that hammers these endpoints.
- **The Fix:** Cache the derived records under the same invalidation key, and
  keep handlers off the hot rebuild path:

```python
_records_cache: OrderedDict[tuple[str, int, int], list[dict[str, Any]]] = OrderedDict()

def _build_records_cached(fp_path: Path, fp: dict[str, Any]) -> list[dict[str, Any]]:
    key = _cache_key(fp_path)
    cached = _records_cache.get(key)
    if cached is not None:
        _records_cache.move_to_end(key)
        return cached
    records = _build_records(fp)
    _records_cache[key] = records
    while len(_records_cache) > _SCAN_CACHE_MAX:
        _records_cache.popitem(last=False)
    return records
```

  Pass `fp_path` alongside `job` into the `*_body` functions (or resolve it
  from `job.out_dir` inside). Related O(n²) in the same layer:
  `_layer_flag` (query.py:619-631) recomputes `_metric_array(records, ...)`
  over the whole model **per row** — hoist the arrays out of the loop and pass
  them in; `tensor_body` (query.py:1000-1007) calls `_metric_array(type_recs, metric)`
  twice per metric — compute once.

### 3.3 Percentile ranks are computed with `searchsorted` on unsorted data — Correctness

- **Location:** `src/weight_atlas/api/query.py:156-160` (`_percentile_of`),
  used by `_layer_flag`, `anomalies_body`, `tensor_body`
- **The Problem:** `np.searchsorted(values, value)` requires `values` sorted;
  `_metric_array` preserves tensor-name order. Result: percentile figures in
  anomaly rows, layer flags ("pNN"), and tensor drilldown context are
  effectively arbitrary — outliers get labeled p42, silently undermining the
  API's core purpose.
- **The Fix:** Sort once per metric array (or use the fact you often already
  compute quantiles):

```python
def _percentile_of(value: float, values_sorted: np.ndarray) -> float | None:
    """values_sorted MUST be ascending; caller sorts once per metric."""
    if values_sorted.size == 0 or not np.isfinite(value):
        return None
    return _r(float(np.searchsorted(values_sorted, value) / values_sorted.size * 100.0))

# call sites
vals = _metric_array(records, metric)
vals_sorted = np.sort(vals)     # hoist; reuse for all rows in the request
pct = _percentile_of(val, vals_sorted)
```

### 3.4 Invalid query-API filter input produces HTTP 500 instead of the error envelope — Resilience

- **Location:** `src/weight_atlas/api/query.py:361-386` (`_parse_layer_filter`;
  bare `int(...)`), reached via `layer=`/`layer_range=` params and slice values
- **The Problem:** `GET /api/model/{id}/query?layer=abc` raises an unhandled
  `ValueError` → 500 with FastAPI's generic body, violating the spec's
  `{error:{code,type,message,hint}}` envelope contract that everything else in
  the router honors.
- **The Fix:**

```python
def _parse_layer_filter(expr: str | None) -> Callable[[int], bool] | None:
    ...
    def _int(s: str) -> int:
        try:
            return int(s.strip())
        except ValueError:
            raise QueryError(
                400, "invalid_param",
                f"invalid layer expression: {expr!r}",
                "42 | >=50 | <=10 | 0:31 | 0,2,4",
            ) from None
    if expr.startswith(">="):
        v = _int(expr[2:])
        ...
```

### 3.5 In-band discovery documents routes that don't exist — Architecture

- **Location:** `src/weight_atlas/api/query.py:1185-1217` (`discovery_body`)
  vs `api/query_routes.py:43-165` (actual mounts)
- **The Problem:** The self-description endpoint is the first thing an LLM
  agent calls, and it advertises `/models`, `/model/{model_id}`, … while the
  actual routes are `/api/models`, `/api/model/{model_id}`, …. Agents following
  the in-band doc get 404s. Hint strings elsewhere in the same module correctly
  say `/api/models` (query.py:32) — the inconsistency is internal evidence of
  drift.
- **The Fix:** Derive paths from one constant set:

```python
_API_BASE = "/api"
endpoints = [
    {"method": "GET", "path": f"{_API_BASE}", ...},
    {"method": "GET", "path": f"{_API_BASE}/schema", ...},
    {"method": "GET", "path": f"{_API_BASE}/models", ...},
    {"method": "GET", "path": f"{_API_BASE}/model/{{model_id}}", ...},
    ...
]
```

### 3.6 Expert-panel scaling silently skipped due to slot/channel misparse — Design

- **Location:** `src/weight_atlas/fields/rasterizer.py:84-99`
  (`load_channel_field`); twin misparse in `render/matplotlib_sheet.py:213`
- **The Problem:** Channel strings derived from filenames look like
  `expert_mlp_gate_height`. The parse `channel[7:].split("_", 1)` yields
  `("mlp", "gate_height")`; `spec.expert_channels.get("gate_height")` misses
  (keys are `height/tint/rough`), so `ch_spec == {}` and the raw-field
  `apply_scale` fallback **never runs** whenever the smooth TIFF is absent
  (old scans, partial dirs, `prefer_smooth=False`). Panels then render from
  unscaled frobenius/kurtosis magnitudes — flat/garbage sheets, no warning.
- **The Fix:** Split structurally on the last segment (channel names have no
  underscores):

```python
elif is_expert:
    parts = channel[len("expert_"):].rsplit("_", 1)
    _panel_slot, base_channel = parts if len(parts) == 2 else (parts[0], "")
    if not base_channel:
        raise ValueError(f"cannot parse expert channel name: {channel!r}")
```

  Longer term, stop re-parsing display names entirely: carry `(slot, channel)`
  pairs through the job metadata instead of encoding them into filenames and
  decoding them at every consumer.

### 3.7 `except KeyError` around renderer calls swallows real renderer failures — Resilience

- **Location:** `cli.py:280-296`, `api/jobs.py:541-559` (call sites);
  `compare/render/delta_sheet.py:180,256` (`spec.sheet["dpi"]`)
- **The Problem:** Both call sites wrap the entire `renderer.render(...)`
  call in `try/except KeyError: pass  # delta renderer not registered`. But
  `DeltaSheet.render` itself raises `KeyError` when `sheet.dpi` is missing from
  a spec, and any future dict lookup bug lands in the same net. Net effect: a
  compare job completes DONE with **no delta PNGs, no error, no warning**.
- **The Fix:** Narrow the guard to registration lookup only; tolerate old specs
  in the renderer:

```python
# callers (cli.py / jobs.py)
from weight_atlas.core.registry import get_renderer
try:
    renderer_cls = get_renderer("delta")
except KeyError:
    renderer_cls = None                      # genuinely not registered
if renderer_cls is not None:
    rendered = renderer_cls().render(...)    # real failures must propagate

# delta_sheet.py
dpi = int(spec.sheet.get("dpi", 150))
```

### 3.8 Reflected HTML injection in `set_model_path` response — Security

- **Location:** `src/weight_atlas/api/routes.py:610-617`
- **The Problem:** The user-supplied path is interpolated raw into an HTML
  response served as `text/html` (and swapped into the DOM by HTMX). A path
  containing `<img src=x onerror=...>` executes script in the app's origin.
  Exploitation requires luring a request (no CSRF protections exist — see
  §2.4), which is exactly why the two issues compound.
- **The Fix:**

```python
from html import escape
content = (
    f'<span class="hint">Model path updated to '
    f'<code>{escape(str(model_path))}</code>. You can now Re-scan.</span>'
)
```

### 3.9 Worker failures lose tracebacks; error strings pollute the artefacts list — Resilience

- **Location:** `src/weight_atlas/api/jobs.py:454-457` (`job.error = str(e)`),
  `jobs.py:637-651` (`produced.append(f"Error rendering {channel}: {e}")`)
- **The Problem:** The persisted error is the exception message only — for a
  GUI tool where users file bugs from the job page, losing the traceback makes
  failures undebuggable. Separately, render errors are appended to the
  artefacts list, which is elsewhere a list of file paths and is rendered as
  such in the UI — type confusion between "file produced" and "error text".
- **The Fix:**

```python
import logging, traceback
log = logging.getLogger("weight_atlas.jobs")

except Exception as e:
    log.exception("job %s (%s) failed", job.job_id, job.job_type)
    job.status = JobStatus.FAILED
    job.error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=8)}"
```

```python
failed: list[str] = []
...
except Exception as e:
    log.warning("render of %s/%s failed", out_dir.name, channel, exc_info=True)
    failed.append(channel)
...
return {"produced": produced, "failed_channels": failed}   # or store on Job
```

### 3.10 `_cmd_diagnose` defined after the `__main__` block — Design hygiene

- **Location:** `src/weight_atlas/cli.py:505-510`
- **The Problem:** `sys.exit(main())` sits above `_cmd_diagnose`. Console-script
  entry works (module fully loads before `main()` is called), but
  `python -m weight_atlas.cli diagnose ...` raises `NameError`. It also invites
  the next function to land in the dead zone.
- **The Fix:** Move `_cmd_diagnose` above `main()`; put the
  `if __name__ == "__main__":` guard last in the file.

---

## 4. Performance & Resilience Optimizations (Nice to Have)

- **Per-block Python loops in Q8_0/Q4_0/Q8_K dequant**
  (`loaders/gguf_dequant.py:278-308`): these types always take the numpy path
  (not in `_GGUF_ONLY`), looping per 18–34-byte block in the interpreter — a
  1 GB Q8_0 tensor costs tens of millions of iterations (minutes). Vectorize
  (identical output, ~100× faster); `_dequant_q1_0` already shows the pattern:

  ```python
  def _dequant_q8_0(data: bytes) -> np.ndarray:
      d = np.frombuffer(data, np.uint8).reshape(-1, 34)
      scale = np.ascontiguousarray(d[:, :2]).view("<f2").astype(np.float32)
      q = np.ascontiguousarray(d[:, 2:]).view(np.int8).astype(np.float32)
      return (q * scale[:, None]).ravel()
  ```

  Also add `len(data) % block_size` truncation guards to all block decoders
  (today trailing bytes are silently dropped → short plausible tensors from
  corrupt files).

- **Whole-array copy before dequantization** (`loaders/gguf_loader.py:88,201`):
  `tensor.data.tobytes()` duplicates the mmap-backed payload on the heap per
  materialization, multiplied by parallel workers — precisely the OOM regime
  scan.py documents. Every consumer accepts buffer objects; pass `data`
  directly.

- **SVD lock over-serialization** (`stats/spectrum.py:68-94,106-109,122-136`):
  the rSVD's heavy GEMMs (`y = m @ omega`, power iterations) run inside
  `_spectrum_lock`; only the LAPACK `qr`/`svd` calls need exclusion. Moving the
  matmuls outside shrinks the critical section seconds→milliseconds for large
  matrices. Determinism unaffected (results depend only on inputs+seed).
  Also: with `u1_coherence` enabled, `paired.py` computes the Δ-spectrum and
  u1 as **two separate locked decompositions of the same matrix** — add a
  combined `spectrum_and_u1(m, seed)` helper returning `(s, u1)` from one pass.

- **`list_scans` parses N multi-MB fingerprints to list names**
  (`api/query.py:329-355`): first `/api/models` call after restart reads every
  fingerprint just to extract seven scalar fields (and evicts the 16-slot
  cache). Either accept it (cache warms) or persist the small header fields
  into the jobs DB at scan time.

- **Expert-panel compare discards real expert IDs and ignores the spec's
  interp key** (`compare/panel.py:75-77`): comment says "Override col_labels
  with expert IDs" but assigns `str(i)`; `align(...)` is called without
  `interp=`, bypassing `compare.aligned_interp`. One-liners:

  ```python
  aligned.col_labels = list(panel_a.col_labels) or [str(i) for i in range(panel_a.data.shape[1])]
  summary = align(..., mode=mode, interp=(spec.compare.get("aligned_interp") or "linear"))
  ```

- **Silent 64-row upsampling in aligned mode** (`compare/align.py:165-166,
  197-201`): `n_rows_common = max(n_rows_max, 64)` resamples two identical
  12-layer models onto 64 rows without the resample warning (which fires only
  on `n_rows_a != n_rows_b`). Emit the warning whenever `n_rows_common !=
  n_rows_a`.

- **Infinite loop on bad spec value** (`render/fractal/mosaic.py:63-76`):
  `fractal.sdf.max_cells = 0` floors row/col counts at 1 so the decimation
  loop can never satisfy `row_count * col_count <= max_cells` — the single API
  worker spins forever (until the sweeper falsely re-queues it). Guard:
  `max_cells = max(1, int(max_cells))`.

- **Paired retry masks root cause and re-runs everything**
  (`paired/paired.py:968-984`): the broad `except RuntimeError` wraps the whole
  measurement loop, so any runtime error triggers a complete second pass with
  limits disabled and triplicates the loop body. Scope the fallback to
  `threadpool_limits(limits=1)` construction only, and factor the iterate/report
  loop into one `_measure(ex)` closure. Similarly, `Executor.map` submits
  eagerly: a shape-mismatch `ValueError` on pair #1 still waits for the entire
  queue before propagating — consider `submit` + `as_completed` with early
  cancellation (index results to preserve determinism).

- **Paired qtype map can poison the manifest** (`paired/render.py:195-197` +
  `paired.py:1069`): when no qtype field was written (zero matched pairs),
  `_render_qtype_map` returns a path to a file that doesn't exist; appending it
  to artefacts makes `_sha256` raise after all expensive work completed.
  Return `None` when nothing was produced and skip the append.

- **Protocol spec resolution breaks in wheel installs, silently**
  (`activity/protocol.py:83-85`; same latent pattern in
  `core/types.py:171-173`): the repo-relative `specs/` lookup fails in a normal
  pip install (hatch packaging ships only `src/weight_atlas`), leaving an empty
  protocol registry and a misleading `Unknown protocol version: 'v1'` far from
  the cause. Ship the JSON inside the package and load via
  `importlib.resources.files("weight_atlas")...`, or fail loudly at startup
  when expected files are missing.

- **Padded positions pollute residual-RMS captures**
  (`activity/capture.py:125-142`): `padding="max_length"` guarantees full-length
  states and the hooks ignore `attention_mask`, so pad tokens contribute to RMS
  and the NaN-masking machinery for absent positions can never fire. Mask in
  the hook: `rms = rms * attention_mask` (write NaN where mask == 0).

- **Embedding density uses a Python loop over vocab points**
  (`embedding/pca.py:134-135`): replace with
  `np.add.at(density, (y_bins, x_bins), 1.0)` (~50× faster, identical output).

- **Scan output dir collision** (`api/routes.py:177-178`): `out_dir =
  output_root / model_path.name` — scanning `a/model.safetensors` then
  `b/model.safetensors` overwrites the first scan's outputs (the compare
  endpoint already fixed this pattern with a uuid suffix; scans didn't).

- **Blocking work on the event loop** (`api/routes.py:634`,
  `api/jobs.py:773-833`): `POST /api/import` renders sheets synchronously
  inside the async handler (matplotlib, seconds-to-minutes) — the exact problem
  that motivated offloading rescan/render onto the worker. Enqueue import as a
  job, or at minimum `await run_in_threadpool(...)`.

- **Non-monotonic progress reporting in scan** (`scan.py:236-330`): expert
  panels report 0.93 before vision (≈0.80) and embedding (0.80-0.84) phases
  run; UI progress jumps backwards. Renumber phases to match actual execution
  order.

- **Dead config surface: `seeds.svd` ignored by the scan pipeline**
  (`stats/norms.py:49-57` docstring vs `scan.py:56-63` which constructs stats
  with default seed): changing `seeds.svd` in the spec reseeds paired Δ-spectra
  while scan spectra stay at seed 0, breaking the "same seeded rSVD" contract
  between pipelines. Thread the spec seed into `_make_handles` or delete the
  parameter.

- **Safetensors header trust** (`loaders/safetensors_loader.py:67-79`): the
  8-byte length prefix is trusted for `f.read(size)` (unbounded allocation on a
  crafted/garbage file) and `data_offsets` are used without bounds checks
  (negative offsets read header bytes as weights). Cap the header size (e.g.
  512 MB) and validate `0 <= start <= end <= len(file) - data_offset` with a
  named error.

- **Redundant symlink re-check** (`api/routes.py:683-688,722-727`):
  `artefact_path` is already `.resolve()`d before the containment check, so the
  subsequent `is_symlink()` branch re-verifies what `resolve()` proved. Harmless
  but dead weight — delete or replace the first check with
  `Path.is_relative_to` for readability.

### Components where no major issues were found

Explicitly verified robust — do not "fix" these:

- **Blender subprocess handling** (`render/blender/blender_wrapper.py`,
  `render_terrain.py`, `render_sdf.py`): argv-list invocation, `timeout=600`
  with kill/reap, stderr-traceback failure sniffing (correct given Blender's
  exit-0-on-script-crash behavior), `tempfile.TemporaryDirectory` handoff,
  env-var binary validated before exec. No injection or zombie-process paths.
- **Determinism machinery**: Agg forced pre-import in all matplotlib renderers;
  PNG tEXt metadata epoch-pinned/stripped (correct PNG chunk walk); OBJ writers
  emit fixed-width floats; compare JSON `sort_keys=True`; fractal value-noise
  hash is RNG-free splitmix64. Byte-determinism contract holds.
- **Q4_0 nibble layout and MXFP4 math** (`gguf_dequant.py`, `mxfp4.py`):
  bit-exact against the reference library; pinned by tests.
- **fields/smoothing.py, fields/tif_io.py, fields/degenerations.py** (one dead
  branch aside), **stats/shape_moments.py, stats/stable_rank.py**,
  **core/registry.py**, **core/types.py** (TensorHandle lifecycle is race-free
  as used), **render/preview.py**, **render/fractal/fbm.py**,
  **render/fractal/sdf.py**: clean.
- **Job queue SQLite usage**: parameterized queries throughout, WAL +
  busy-timeout, explicit connection closing (fd-leak fix well documented),
  idempotent recovery paths (`start()`, sweeper, done-job skip) are mutually
  consistent for the single-worker design.
- **Paired numeric guards**: SQNR/cosine/band/classification branches handle
  zero-divisors, empty arrays, silent references, and NaN/inf references
  thoroughly.

---

## 5. Clarifying Questions for the Author

1. **Deployment posture for `serve`:** Is LAN exposure (`0.0.0.0` default, no
   auth) an intended primary use case, or developer convenience? If LAN use is
   real, should the roadmap include a token/auth gate and Host-header
   validation (§2.4)? This decides whether §2.4 is a defaults change or needs
   real auth design.
2. **Q8_K provenance:** Was `_dequant_q8_k`'s 258-byte layout ever validated
   against a llama.cpp-produced Q8_K file, or inferred? Are there real-world
   Q8_K models you need to support via the pure-numpy path (gguf-less installs),
   or can Q8_K simply route through the `gguf` library (§2.1)?
3. **Multi-process expectations for the job queue:** The queue assumes exactly
   one worker process (in-memory `_queue`/`_enqueued`/`_current_job_id`,
   single-worker sweeper). Should `weight-atlas serve --workers N` (or two
   processes sharing `data/jobs.db`) be supported — meaning recovery/sweeper
   logic needs DB-level leases — or is single-process a contract worth
   asserting/enforcing at startup?
