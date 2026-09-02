# Phase 0: Scan Sharing — Provenance + Package Format (Proposal)

> Status: Proposal | Date: 2026-09-02
> Scope: local-only. No public website, no new server surface beyond the
> existing `/api/import`. Everything here must be useful without any
> second party ever existing.

---

## 1. Motivation

Users scan models locally. Scans are expensive (minutes to hours) and the
results — fingerprint, fields, renders — are small compared to the models
themselves (a 94 GB GGUF compresses to a ~16 MB fingerprint + ~61 MB
fields). Sharing a scan should not require re-scanning.

Two properties make weight-atlas scans unusually shareable:

1. **Determinism**: identical model files + same tool version + same seeds
   → byte-identical `fingerprint.json`. Two strangers scanning the same
   GGUF independently produce bit-identical data. This enables free
   deduplication, cross-verification, and trust-by-recomputation on any
   future public registry.
2. **Self-contained fingerprints**: every statistic is in one JSON file;
   fields/renders are derived artefacts, regenerable from the fingerprint
   and the spec.

What is missing for sharing today:

- Nothing anchors a fingerprint to the actual **model files** (provenance).
  A fingerprint is currently an unverifiable claim: nothing prevents a
  hand-edited `tensors` block.
- No defined interchange format: a scan directory is an ad-hoc pile of
   files; the server's `import_scan` reads whatever is on disk.
- No license/attribution metadata: scans are derived data of model
  weights; the license of the source model governs redistribution of the
  scan.

Phase 0 closes these three gaps locally. A public website (Phase 1) is
explicitly out of scope — see §8.

## 2. Provenance anchoring (build now, retroactively valuable)

### 2.1 Source hashes at scan time

`scan()` already mmaps every model file. During the (already serial)
loading phase, compute per-file SHA-256 as the mmap is consumed and store
it in the fingerprint:

```json
"model": {
  "n_tensors": 74824, "n_layers": 48, "moe": {...},
  "sources": [
    {"file": "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf",
     "bytes": 118110592, "sha256": "…"}
  ]
}
```

- File names are basenames only (no local paths leak into shareable data).
- Streaming readers hash the same bytes they already read — cost is one
  pass of SHA-256 over data already being read (I/O-bound phases become
  hashing-bound at most ~2×; the giant-table walk hashes its mmap while
  dequantizing blocks).
- Non-streaming loaders hash via the existing `_sha256` helper
  (`scan.py:63`), streaming paths accumulate the digest per block
  (`hashlib.update` in the block walk — deterministic block order is
  already a pipeline invariant).
- GGUF multi-shard: one entry per shard, ordered by filename (the same
  order `_discover_gguf_files` already produces; deterministic).
- Optional per-tensor anchors (e.g. first 64 bytes of each tensor's raw
  payload) are considered and **rejected for Phase 0**: the file-level
  hash subsumes them at negligible cost, and per-tensor digests would
  bloat the fingerprint ~5 KB/tensor with no additional trust
  (any file edit breaks the file hash anyway).

### 2.2 `source_digest` (short identity for dedup)

`sha256(concat of (basename, bytes, sha256) tuples)` — one stable string
identifying the *model files*, independent of scan settings. Displayed in
the UI, used as the dedup key by any future registry, and the natural key
for "scan of the same model by different people / tool versions".

### 2.3 Trust model, stated honestly

A file-hash anchor makes a fingerprint *checkable by anyone holding the
model*: re-scan → same hashes → same fingerprint (given tool+spec+seeds).
It does NOT make it checkable by someone holding only the fingerprint —
that requires either a trusted third party (Phase 1's website, out of
scope) or recomputation (always available; determinism guarantees
success). Phase 0 documents this asymmetry in the package manifest
rather than solving it.

## 3. Package format (`.wasc` — weight-atlas scan bundle)

### 3.1 Layout

Deterministic zip (fixed timestamps `1980-01-01 00:00:00`, sorted entry
names, compression deflate-6 — same discipline as OBJ/TIFF determinism):

```
package.json                # manifest: format, versions, hashes, license
fingerprint.json            # exact scan output (hash-anchored after §2)
manifest.json               # artefact sha256s (already produced by scan)
fields/field_*.tif          # rasterized fields (optional, see 'profiles')
renders/*.png               # sheet renders (optional)
```

### 3.2 `package.json`

```json
{
  "format": "wasc", "format_version": 1,
  "created_by": {"tool": "weight-atlas", "tool_version": "0.2.0",
                  "spec_version": 4},
  "model": {"name": "Qwen3.8-Flash-Next-UD-IQ4_XS",
            "sources": [...], "source_digest": "…"},
  "scan": {"arch": "gguf-dense", "n_tensors": 74824, "quantization": "ggml_20",
           "jobs": null, "quant_probe": false, "seeds": {"svd": 0, "distribution": 0}},
  "license": {"model_license": "apache-2.0 (declared)",
               "scan_license": "CC-BY-4.0",
               "declared_by": "scanner", "verified": false},
  "contents": {"fingerprint.json": "sha256…", "fields/…": "sha256…"},
  "provenance_note": "hashes anchor the scan to model files; verification requires the model (determinism) or a trusted registry"
}
```

Fields mirror what the scan already knows (`spec.seeds`, `jobs` value,
probe flags from the job). `license.declared_by: "scanner"` is an
unverified self-declaration — Phase 1 could cross-check against HF
metadata; Phase 0 records the declaration only. `verified: false` is
written by the exporter and never set by local code.

### 3.3 Profiles

Not every use needs 60 MB of fields. Two export profiles:

- **full** — fingerprint + fields + renders (~80 MB for the largest scan)
- **stats** — fingerprint only (~16 MB gzipped) — the analytics payload

`stats` is the interesting one for sharing: everything the records /
scatter / compare-by-stats surfaces consume is in the fingerprint.
Fields are re-rasterizable from a fingerprint + spec by anyone (the
rasterizer is deterministic and pure — `rasterize(stats, spec)` needs
nothing else).

### 3.4 Versioning policy

`format_version: 1` pinned by tests; additive keys (following the
`specs/AGENTS.md` additive-extension precedent) never bump it. The
fingerprint's `spec_version` + `tool_version` govern numeric
compatibility (compare already hard-rejects spec mismatches; the same
policy applies to imports — a foreign `spec_version` is a hard reject,
unknown *package* keys are tolerated).

## 4. CLI + API surface

```
weight-atlas export <scan_dir> [--out pkg.wasc] [--profile full|stats] [--license-model …] [--license-scan …]
weight-atlas import <pkg.wasc> --out <scan_dir>   # or into a running server via /api/import
```

- `export`: validates the scan dir (fingerprint + manifest hashes
  match), refuses to package a scan whose fingerprint lacks source
  hashes (pre-fix scans), writes the deterministic zip.
- `import`: extracts with zip-slip guards into the given directory,
  verifies every inner sha256, verifies `format_version` and
  `spec_version`, then registers via the existing `JobQueue.import_scan`
  (which already handles renders/artefacts). The server route
  (`POST /api/import`) grows an optional package path — extraction stays
  server-local, so the public-hardening (§8) is deliberately NOT needed
  for Phase 0's local use.
- Round-trip is pinned by a test: `export → import → byte-identical
  fingerprint.json` (trivially true given content hashes, but the test
  also pins zip determinism: two exports of the same scan are
  byte-identical).

## 5. UI surface (minimal)

- Model detail page: show `source_digest` + source hashes (collapsible),
  a "Download scan package" button (`?profile=stats|full`), and an
  "Import package" upload form (local server use).
- Models list: source-digest badge so re-scans of the same model are
  recognizable (dedup display, not enforcement).

## 6. Implementation order

1. **Hash anchoring in `scan()`** (§2) — independent of everything else;
   makes every future scan trustworthy. Fingerprint key `model.sources`
   is additive (older readers ignore it; JSON-parsing tests updated).
2. **`export`/`import` CLI** (§4) with round-trip + determinism tests.
3. **`/api/import` package support + UI bits** (§5).
4. Docs: `user_manual.md` sharing section; ARCHITECTURE note on
   provenance.

Estimated: hash anchoring is small (the loading phase is serial and
already reads every byte); export/import is a focused unit of work
(zip determinism helpers exist in spirit via the manifest `_sha256`
discipline).

## 7. What Phase 0 deliberately does not do

- **No public website, no public API.** A site needs auth, quotas,
  archive-bomb hardening at scale, Postgres, moderation, and an ops
  commitment. Nothing in Phase 0 blocks it; the package format is the
  contract Phase 1 would consume.
- **No anonymity/privacy tooling.** Fingerprints contain no weights,
  but `package.json` carries model names and hashes by design; users
  scanning private models must simply not share (documented in the
  manual, not enforced).
- **No license verification** beyond recording the declaration
  (Phase 1 can cross-check HF metadata).
- **No compare-across-registry features.** Compare works on local scan
  dirs exactly as today; the package import lands them in the same
  local DB.

## 8. Phase 1 sketch (for context only — not committed)

If sharing demand materializes: the site is a thin, hardened wrapper
over the package format — upload → extract-with-guards → verify hashes →
register → serve. Determinism gives it dedup + cross-verification for
free. Agent uploads are authenticated API keys on the same endpoint.
Every piece of Phase 1 consumes Phase 0 contracts; none of it influences
Phase 0's design beyond what is written above.

## 9. Verification plan

- Unit: hash accumulation matches `_sha256` on real files (GGUF
  multi-shard + safetensors + PyTorch/BDH).
- Determinism: two exports byte-identical; export→import round-trip
  byte-identical fingerprint.
- Rejection paths: missing source hashes (pre-fix scan) → export
  refuses; wrong inner hash → import refuses; foreign `spec_version` →
  import refuses; zip-slip attempt → import refuses (synthetic evil
  zip).
- Full suite stays green; no public surface changes.
