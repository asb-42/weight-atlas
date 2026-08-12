# Weight Atlas — Production-Grade Code Review

**Date:** 2026-08-12
**Scope:** full repository at `1a787b5` (Kimi K3 dequant pipeline), 6,267 LOC Python
**Method:** static review + execution tracing + targeted verification (242/242 tests pass)

Repository purpose (from README/ARCHITECTURE): LLM weight fingerprinting — scan a model
(safetensors/GGUF) into per-tensor statistics, rasterize into 2D fields, render topographic
sheets, and compare two scans quantitatively. Three core guarantees are claimed:
(1) renderer-independent artifacts, (2) render/compare never read weights, (3) deterministic,
byte-identical outputs. A FastAPI web UI runs an in-process worker for scan/compare jobs.

Findings are ordered by production impact, not by file. Each is given a severity, confidence,
and a concrete failure mechanism. Stylistic issues are excluded.

---

## 1. Spec version/slot divergence between CLI and Web UI breaks the "spec_version mismatch" contract

- **Severity:** Critical
- **Confidence:** High (verified empirically)
- **Files:** `src/weight_atlas/cli.py:73,131,182` vs `src/weight_atlas/api/main.py:33`, `api/routes.py:99,288,322`, `api/jobs.py:405`
- **Technical explanation:** The CLI loads `specs/atlas_spec.v2.3.json` (`spec_version: 3`, 54 slots), while the web API loads `specs/atlas_spec.v2.1.json` (`spec_version: 2`, 15 slots). Both files exist and are silently inconsistent. The slot sets differ by 39 entries (v2.3 adds Kimi K3 MLA/vision/KDA slots). Because `scan.py` writes `spec.spec_version` into `fingerprint.json` and `compare/align.py:39` **hard-rejects** `spec_version` mismatches, a model scanned via the CLI (v3) can never be compared against a model scanned via the web UI (v2) — the comparison raises `ValueError` even when the two models are actually the same architecture.
- **Failure scenario:** A researcher scans a model with `weight-atlas scan` (CLI, v3 fingerprint), later imports or rescans the same/ablated model through the web UI (v2 fingerprint), then runs `compare` — it throws "spec_version mismatch: A=3, B=2".
- **Impact:** Silent cross-tool incompatibility of the core artifact format; the "deterministic canonical spec" guarantee is violated in practice because there are two live specs. Users get inconsistent fingerprints (v3 has ~3.6× the columns) with no warning at scan time.
- **Remediation:** Single source of truth. Have `create_app` default to the same spec the CLI uses (or better, resolve both through one function that asserts the loaded file's `spec_version`). Add a startup assertion that every shipped `specs/atlas_spec*.json` used in code has one canonical `spec_version`/slot list, or delete the stale files. Route all spec resolution through `AtlasSpec.from_json` with a version check.
- **Tradeoffs:** Picking one spec version is a breaking change for whichever entrypoint changes; document it in CHANGELOG and bump the tool version so `check_compatibility` warns rather than hard-fails during the transition.

---

## 2. Each tensor is loaded/dequantized 7× per scan (no memoization on `TensorHandle.load`)

- **Severity:** High
- **Confidence:** High
- **Files:** `src/weight_atlas/scan.py:36-48` (`_make_handles`), `core/types.py:34-36` (`TensorHandle.load`), `stats/norms.py`, `stats/shape_moments.py`, `stats/stable_rank.py`, `loaders/safetensors_loader.py:105-109`, `loaders/gguf_loader.py:112-115`
- **Technical explanation:** `_make_handles` instantiates six statistic objects, each of which independently calls `t.handle.load()`. `StableRank.compute` re-derives Frobenius + spectral itself. The net effect is **7 materializations per tensor**: Frobenius, Spectral, EffectiveRank, StableRank(→Frobenius+Spectral), Kurtosis, Sparsity. `TensorHandle.load()` has no cache, and each safetensors load re-reads and re-parses the JSON header (`_read_header_full`) plus re-reads the raw byte payload from disk; each GGUF load re-runs `data.tobytes()` (full byte copy) plus full `dequantize`.
- **Failure scenario:** Scanning a multi-GB model is ~7× slower than necessary and does ~7× the I/O. For GGUF MoE, the expert sub-handle loader (`gguf_loader.py:117-129`) dequantizes the **entire 3D stacked tensor** and slices it for every expert sub-handle, so 7 stats × `n_experts` × full-3D-dequant per layer — the Kimi K3 pipeline this repo targets has ~100 routed experts per layer, making this quadratic in practice.
- **Impact:** Orders-of-magnitude runtime/memory waste on the primary scan path; de-facto limits the tool to small models or makes MoE scans prohibitively slow.
- **Remediation:** Cache the materialized array on the handle (`functools.cached_property` or an internal `_arr` guard in `load()`), or restructure stats to load once and pass the array through (`Statistic.compute(array)`). At minimum, have `_make_handles` load once and reuse the array across all six stats. For GGUF experts, dequantize the 3D tensor once and slice views rather than per-expert full dequant.
- **Tradeoffs:** Memoization holds one float32 tensor in memory at a time (bounded peak, already the case today) — negligible vs. the current cost; determinism is unaffected since loaders are pure.

---

## 3. GGUF MoE expert panels are silently dropped (`extract_expert_id` returns None)

- **Severity:** High
- **Confidence:** High (verified: `extract_expert_id("blk.0.ffn_gate_exps.weight[3]")` → `None`)
- **Files:** `core/name_map.py:251-266` (`extract_expert_id`), `fields/rasterizer.py:71-141` (`rasterize_expert_panels`), `fields/rasterizer.py:144-173` (`detect_moe`), `loaders/gguf_loader.py:80-98`
- **Technical explanation:** The GGUF loader correctly splits 3D expert tensors into per-expert sub-handles and sets `TensorHandle.expert_id` (which flows into `TensorStats.expert_id` via `_make_handles`). But `rasterize_expert_panels` and `detect_moe` do **not** read `ts.expert_id`; they re-derive the id with `extract_expert_id(ts.name)`, whose regexes only match HF (`mlp.experts.\d+`, `block_sparse_moe.experts.\d+`) and explicitly do not handle the GGUF `...[N]` suffix. Result: `expert_id is None → continue` for every GGUF expert, so no `field_expert_*` TIFFs are written and `detect_moe` returns `{}` (no `model.moe` block) for GGUF MoE models.
- **Failure scenario:** Scanning a GGUF Mixtral/Qwen-MoE model produces a fingerprint and main raster but **no expert panels and no MoE metadata**, with no warning. Expert-level comparison and the `--field expert_mlp_down` render path silently produce nothing.
- **Impact:** Core M6/MoE feature is effectively broken for the GGUF loader, contradicting ARCHITECTURE.md's "GGUF 3D expert split" design. Silent data loss — worst kind for a measurement tool.
- **Remediation:** Make `rasterize_expert_panels`/`detect_moe` prefer `ts.expert_id` when set, falling back to `extract_expert_id(name)`. Add a regression test that runs a GGUF MoE fixture through `scan()` and asserts expert panel TIFFs and `model.moe` exist (existing `test_moe.py` only tests HF naming and only checks the loader's sub-handles, not the rasterizer).
- **Tradeoffs:** None — using the already-populated `expert_id` field is strictly more correct.

---

## 4. Web UI blocks the event loop with synchronous scan/render and has no auth or resource limits

- **Severity:** High
- **Confidence:** High
- **Files:** `api/routes.py:269-304` (`rescan_job`), `306-400` (`render_job`), `402-416` (`import_scan`), `46-62` (`create_job`), `208-228` (`create_compare_job`)
- **Technical explanation:** Three endpoints are declared `async def` but perform CPU/IO-heavy synchronous work inline — `rescan_job` calls `run_scan()` (full model scan, see finding #2), `render_job` renders every channel with matplotlib, `import_scan` auto-renders. FastAPI runs these on the single event-loop thread, so one request blocks all other requests (including HTMX polling and job submission) for the duration. Additionally, none of these routes have authentication or rate limiting, and `create_job`/`import_scan`/`create_compare_job` accept arbitrary filesystem paths (`model_path`, `scan_dir`, `dir_a`, `dir_b`) with no confinement to `output_root`.
- **Failure scenario:** A user (or any client that can reach the port) POSTs `/api/jobs/{id}/rescan` on a large model, or `/api/import` with a large directory; the server becomes unresponsive for minutes, effectively a self-DoS. `import_scan` with an arbitrary `scan_dir` containing a `fingerprint.json` registers a job whose `out_dir` is that directory, after which `/api/artefacts/{id}/{path}` serves allowlisted extensions (`.json`, `.npy`, `.txt`, `.csv`) from it — an arbitrary-file-read primitive if the app is ever exposed beyond localhost.
- **Impact:** Availability loss under trivial conditions; latent data-exfiltration path if the app is bound to a non-loopback interface (the README's `uvicorn ... --reload` defaults to 127.0.0.1, so exposure is operator-dependent).
- **Remediation:** Offload the heavy work to the existing worker thread (submit a job like the normal scan path instead of inline execution). Add a middleware/route check that imported/created paths resolve inside `output_root` (or a configurable allowlist of model roots). If network exposure is intended, add auth and rate limiting; otherwise document that the UI is local-only.
- **Tradeoffs:** Moving rescan/render through the queue serializes behind other jobs (correct, since the worker is single-threaded); path confinement needs a decision on where imported scans may legitimately live.

---

## 5. Job queue does not recover QUEUED jobs after a restart despite SQLite persistence

- **Severity:** Medium
- **Confidence:** High
- **Files:** `api/jobs.py:139-159` (`start`/`_run`), `319-339` (`submit`), `70-88` (`_init_db`)
- **Technical explanation:** `_run` only consumes `job_id` from the in-memory `self._queue`. `submit` persists the job to SQLite (status `queued`) *and* enqueues the id. On restart, the in-memory queue is empty, so any job persisted as `queued` is never re-enqueued and never runs. The ARCHITECTURE.md claim "job state survives server restarts" is only true for *completed* jobs; queued-but-unstarted jobs are stranded in `queued` forever.
- **Failure scenario:** Server restarts while a scan is queued (not yet picked up by the worker); after restart the job is stuck at "queued" with no mechanism to resume or fail it.
- **Impact:** Reliability bug that silently loses submitted work across restarts.
- **Remediation:** On `start()`, query SQLite for `status='queued'` jobs and enqueue them before starting the worker. Also reset `running` jobs to `queued` (or mark them `failed` with a "interrupted by restart" message) on startup.
- **Tradeoffs:** Re-enqueueing is safe given job idempotence (scan overwrites its own `out_dir`); resetting `running` → `queued` requires the worker to tolerate re-running a partial scan, which it does since output files are overwritten.

---

## 6. CLI mapping-coverage warning is dead code and re-runs the entire scan

- **Severity:** Medium
- **Confidence:** High
- **Files:** `cli.py:79-95` (`_cmd_scan`)
- **Technical explanation:** Two defects. (a) The fingerprint's `mapping_coverage` block uses keys `in_slots`, `in_other`, `unmapped`, `unmapped_tensors` (`scan.py:311-316`), but `_cmd_scan` reads `mc.get("ratio", 1.0)` and `mc.get("total", 0)` — neither exists, so `ratio` is always `1.0` and the `< 0.8` warning can never fire (and would print "0/0" if it did). (b) To produce this dead warning, `_cmd_scan` re-opens the loader and recomputes **all** statistics a second time (lines 83-87), doubling the scan cost from finding #2, purely to build a fingerprint that `scan()` already wrote to disk.
- **Failure scenario:** Models with poor name-mapping coverage (the exact case the warning exists for) scan without any warning, so users don't know their fingerprints are mostly unmapped tensors.
- **Impact:** A guardrail intended to catch calibration failures is non-functional, and it costs a full extra scan pass.
- **Remediation:** Read `fingerprint.json` from `args.out` (already produced by `run_scan`) and check `mapping_coverage["in_slots"] < 0.8`; delete the re-open/recompute block.
- **Tradeoffs:** None — strictly removes redundant work and fixes the warning.

---

## 7. `gguf_dequant.dequantize` swallows the gguf library's exceptions but its "fallbacks" are not self-contained

- **Severity:** Medium
- **Confidence:** High
- **Files:** `loaders/gguf_dequant.py:136-220`
- **Technical explanation:** The primary path wraps the `gguf` library call in `except (ImportError, AttributeError, Exception)` (i.e., bare `Exception`) and silently falls through to custom implementations. But several of those "fallback" functions themselves start with `import gguf` and call `gguf.dequantize` directly (`_dequant_q4_1`, `_dequant_q5_0`, `_dequant_q5_1`, `_dequant_q8_1`, `_dequant_tq1_0`, `_dequant_tq2_0`, `_dequant_mxfp4`, `_dequant_nvfp4`). If `gguf` is absent or its dequantizer fails, these types raise `ImportError`/`AttributeError` — there is no genuine fallback for them, contradicting the module's stated design. Conversely, if `gguf` is present but produces a wrong shape/partial result, the bare `except` masks the error and may hand back garbage rather than failing loudly.
- **Failure scenario:** Scanning a Q5_0/Q8_1/TQ/MXFP4 GGUF on a machine without `gguf` (or a `gguf` version mismatch) fails with a bare `ImportError` inside the fallback instead of the intended clear "unsupported/install gguf" message; or a corrupt quantized block is silently "dequantized" wrong.
- **Impact:** Misleading error handling and a false sense of fallback coverage; potential silent numeric corruption on quantized models.
- **Remediation:** Split into explicit branches: `try/except ImportError` around a single `import gguf` at module load, then either use `gguf` for everything it supports or the pure-numpy paths; remove per-function `import gguf`. Re-raise genuine dequantization errors instead of swallowing them.
- **Tradeoffs:** Requires deciding which types are "gguf-only"; the custom Q8_0/Q4_0/F16/BF16/F32/Q1_0 paths are already self-contained and should be the true fallbacks.

---

## 8. GGUF 3D expert slice depends on a fragile `tensor.shape` vs `data.shape` convention

- **Severity:** Medium
- **Confidence:** Medium (current tests pass, but correctness rests on an undocumented invariant)
- **Files:** `loaders/gguf_loader.py:74-89` (`open`), `117-129` (`_load_expert_tensor`)
- **Technical explanation:** `open()` computes `n_experts = actual_shape[0]` and `expert_shape = actual_shape[1:]` from `data.shape` (experts in **dim 0**), but `_load_expert_tensor` reshapes to the GGUF-reported `shape` and slices `arr_3d[:, :, expert_id]` (experts in the **last dim**). This only works because the `gguf` library reports `tensor.shape` with dimensions in a different order than `tensor.data.shape` (the code comment acknowledges "GGUF may report shape differently from memory layout"). The passing test fixture happens to satisfy this; any `gguf` library update or a model whose header shape differs will produce silently transposed/sliced expert weights.
- **Failure scenario:** A `gguf` upgrade changes shape ordering; expert sub-handles return the wrong 2D slice (or crash on reshape) with no error — expert panel statistics become silently wrong.
- **Impact:** Latent correctness risk in a code path the repo is actively extending (Kimi K3).
- **Remediation:** Derive everything from a single source (`data.shape`), slice along the dimension that is actually the expert axis (e.g., `np.take`/`[..., expert_id]` consistent with `actual_shape`), and assert `len(actual_shape) == 3` plus expected orientation. Add a test that asserts a specific expert slice equals the known input.
- **Tradeoffs:** Requires confirming the true memory layout for real GGUF MoE files (one-time verification against a real Mixtral/Qwen GGUF).

---

## 9. Latent `NameError` in compare path when no channels are discovered

- **Severity:** Low
- **Confidence:** High
- **Files:** `api/jobs.py:249-283` (`_run_compare`), `cli.py:209-287` (`_cmd_compare`)
- **Technical explanation:** `summary` is assigned inside the `for channel in channels:` loop and then referenced after the loop (`summary.model_a`, `summary.model_b`). If `channels` is empty (no `field_*_raw.tif` keys in the manifest), `summary` is unbound and the code raises `NameError` (in the worker, caught by the broad `except Exception` and recorded as a generic failure; in the CLI, an unhandled traceback).
- **Failure scenario:** A compare between two directories whose `manifest.json` lacks TIFF entries (e.g., an activity-only or partial scan) crashes instead of reporting "nothing to compare".
- **Impact:** Minor crash path; the worker's broad `except` hides the real cause.
- **Remediation:** Guard `if not summary_channels: return []` (or raise a clear `ValueError`) before building `compare_summary`, and capture per-model metadata from the fingerprints directly instead of the loop-scoped `summary`.
- **Tradeoffs:** None.

---

## 10. `detect_loader` maps directories/unreadable files to `safetensors`

- **Severity:** Low
- **Confidence:** High
- **Files:** `core/types.py:124-137`
- **Technical explanation:** `detect_loader` opens the path and reads 4 bytes; on any `OSError` (including `IsADirectoryError` for a directory, or `PermissionError`) it returns `"safetensors"`. A directory of `.gguf` files therefore auto-detects as safetensors, and `SafetensorsLoader._discover_files` finds no `.safetensors` and raises "no .safetensors files" — a confusing error. Users must know to pass `--loader gguf`.
- **Failure scenario:** `weight-atlas scan ./models/my_gguf_dir` fails with a misleading error despite the directory being full of valid GGUF files.
- **Impact:** UX/correctness gap for the common "directory of shards" workflow.
- **Remediation:** If the path is a directory, inspect its contents (e.g., check the first `.gguf`/`.safetensors` file) or report an explicit "cannot detect loader for directory; specify --loader" error.
- **Tradeoffs:** None.

---

## Additional lower-severity observations (grouped)

- **Spec path resolution is inconsistent** — `cli.py` and `api/routes.py:99,288,322`, `api/jobs.py:405` resolve `specs/atlas_spec*.json` as a *relative* path (depends on CWD), while `api/main.py:33` uses an absolute `base / ...` path. Launching uvicorn from another directory breaks the relative lookups (500s / `FileNotFoundError`). Single absolute-resolution helper recommended. **Low.**
- **SQLite access pattern** — `api/jobs.py` opens a new connection per operation with default isolation and no WAL/busy-timeout. Concurrent worker writes + request reads can hit `database is locked` under load. `PRAGMA journal_mode=WAL` and an explicit busy timeout are cheap. **Low.**
- **`import_scan` / `_auto_render_sheets` swallow all exceptions** (`except Exception: pass`, `jobs.py:445-446`, `routes.py:301-302`) and `_auto_render_sheets` writes `render/*.png` that `import_scan` then re-globs — rendering failures are invisible and unlogged (contrast with `_execute`, which records `job.error`). **Low.**
- **`main.py:69` instantiates `app = create_app()` at import time**, starting a background worker thread and creating directories as an import side effect — fragile for tests and `--reload`. **Low.**
- **`compare_report` (`routes.py:265`) calls `out_dir.relative_to(output_root)` without the try/except used in `model_detail`** — raises `ValueError` for out_dirs outside the root (imported scans). **Low.**
- **`@app.on_event("shutdown")` is deprecated** (observed in test warnings) and leaks a lingering worker thread reference. **Low.**
- **`_compute_scaling_metadata` hardcodes `params: {lower: 0.01, upper: 0.99}`** regardless of the actual spec values it computed from. **Low.**
- **`compare` out-dir collision** — `routes.py:224` names compare output `compare_{dir_a.name}_vs_{dir_b.name}`, so comparing two same-named models in different roots overwrites earlier results. **Low.**

---

## What is well done (noted to avoid rework)

- The artefact-serving routes (`routes.py:423-507`) correctly implement traversal + symlink-escape protection via full `.resolve()` + `relative_to` containment plus an extension allowlist — this is sound and the redundant symlink check is harmless.
- `safetensors_loader.py` uses default-argument binding in its closures (correct late-binding capture) and reads BF16/MXFP4 at the byte level, avoiding dtype-registry fragility.
- NaN-safety in `scaling.py`/`smoothing.py` (mask → transform → re-mask) is consistent and correct.
- Determinism is genuinely engineered (chunked float64 Frobenius accumulation, spec-seeded RNGs, fixed PNG metadata) rather than asserted.

## Summary of recommended priorities

1. Reconcile the CLI vs. web spec (`atlas_spec.v2.1.json` vs `v2.3.json`) — single canonical spec + version assertion. (Critical, small change.)
2. Memoize tensor loads (or load-once/reuse across stats), and dequantize GGUF 3D expert tensors once. (High, large perf win.)
3. Fix GGUF MoE expert extraction to use `ts.expert_id`; add a full GGUF-MoE scan regression test. (High, silent data loss.)
4. Move rescan/render/import heavy work off the event loop; confine filesystem paths; clarify local-only deployment. (High.)
5. Recover `queued` jobs on `JobQueue.start()`. (Medium.)
6. Fix/remove the dead CLI coverage warning and its redundant second scan. (Medium.)
7. Tighten `gguf_dequant` exception handling and fallback reality. (Medium.)
