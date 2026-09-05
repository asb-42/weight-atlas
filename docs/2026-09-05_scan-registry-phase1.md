# Phase 1: Public Scan Registry Website (Proposal)

> Status: Proposal | Date: 2026-09-05
> Scope: community registry with public uploads — scope option (2).
> Depends on: Phase 0 (provenance anchoring + `.wasc` format, both shipped:
> `f77027b`, `8bb224a`, `bcca4ff`).
> Non-goals restated up front: the site never hosts weights (derived stats
> only), never re-scans uploads, never verifies licenses beyond recording
> declarations.

---

## 1. Goal

A public website where anyone can browse shared weight-atlas scans and
contribute their own — humans via the browser, agents via API keys. The
site is a thin, hardened wrapper over the Phase 0 contracts: the `.wasc`
package format is the ingestion contract, fingerprint determinism is the
dedup and cross-verification engine, and the existing query/render
pipeline is the presentation layer.

Success looks like: a stranger scans a model locally, uploads the
package, and within minutes their scan is browsable, comparable against
every compatible scan on the site, and queryable by agents — with its
trust tier stated honestly.

## 2. What we reuse (Phase 0 inventory — no rewrites)

| Existing piece | Role on the site |
|---|---|
| `.wasc` format (deterministic zip, `package.json`, content hashes) | Upload/ingest contract, unchanged |
| `import_package` verification (hash-before-extract, zip-slip/symlink guards, spec-version reject) | Ingestion core; needs *scale* hardening (§4), not redesign |
| Fingerprint determinism | Free dedup (same `source_digest` → same bytes) + free cross-verification (§3) |
| Provenance block (`model.sources`, `source_digest`) | Trust tiers (§3), dedup keys |
| Query engine (`query.py`) + query API (spec v0.2, scatter/records) | Agent + UI data layer, behind auth/quotas |
| Sheet/fractal renderers + matplotlib PNGs | Gallery visuals (the site's hero imagery already exists) |
| htmx server-rendered UI | Keep the stack; add a real design system (§6) |

## 3. Trust model (the one hard product decision)

Fingerprints are verifiable **by re-scan** (anyone holding the model),
never from the fingerprint alone. The site states trust explicitly via
three tiers, computed — never asserted:

- **`unverified`** (default): uploaded package passed format/hash checks.
  Displayed with a neutral badge. Most uploads live here; that is fine.
- **`hash-verified`**: the package's source hashes match a known
  distribution record (e.g. HuggingFace file checksums for that repo +
  revision, submitted by anyone and itself checksummed). Means "these
  numbers claim to be *that* published model, and the claim checks out."
- **`reproduced`**: two independent uploads (different accounts, different
  `created_by` tool runs) are byte-identical. Determinism makes this free
  and it is the strongest claim the site can offer without re-scanning.

Rules: tiers are computed by the site on ingest and on every matching
upload (a second identical upload *promotes* the first). Tiers are shown
everywhere the data is shown (gallery cards, model pages, API responses).
Disputes (wrong model attached to a hash, poisoned stats) go through the
takedown/report flow (§7), never silent deletion — tombstone + reason,
same discipline as our own review logs.

License handling: the package's self-declared `model_license` /
`scan_license` are displayed verbatim with `verified: false` unless
cross-checked against registry metadata (HF API where available). The
site records declarations; it does not adjudicate them. Takedown on
rights-holder request (§7).

## 4. Security & hardening (threat model)

The ingest path processes hostile input by design. Concrete measures:

| Threat | Mitigation |
|---|---|
| Zip bombs / decompression bombs | Hard caps: package ≤ 500 MB on the wire; decompressed-size and file-count ceilings enforced *during* streaming extract (abort mid-stream, not after); compression-ratio tripwire |
| Zip-slip / symlink / absolute paths | Already in `import_package`; add fuzz tests with synthetic evil zips in CI |
| Malformed 85 MB fingerprints (CPU/memory DoS) | Streaming JSON parse with size cap; reject before full materialization; parse in the worker, never the request thread |
| Malicious tensor names → stored XSS | All UI rendering already escapes (HTML-escaping work landed pre-launch); add a template-output fuzz pass over real hostile names |
| CSRF on browser forms | Token-based CSRF on all mutating browser routes (htmx sends the token header) |
| Credential theft / abuse | API keys are random 256-bit, stored as SHA-256 hashes only; per-key quotas (uploads/day, bytes stored); key rotation + revocation UI |
| Enumeration / scraping | Query API rate limits per key + IP; gallery pagination caps (already stride-capped deterministically) |
| Dependency / supply chain | Pinned lockfile (already `uv.lock`), `pip-audit` in CI, minimal container image |

Runtime posture: read-only filesystem except the artefact store; no model
code execution server-side (the scan pipeline never runs on uploads —
ingest only verifies + registers); separate worker processes for
extract/verify/render with memory caps; security headers
(CSP, nosniff, frame options).

## 5. Data & storage architecture

- **Database**: migrate the job store from SQLite to a server database.
  The host already operates MySQL/MariaDB (see §10.1) — use MariaDB, not
  Postgres, to reuse existing backups/monitoring/ops knowledge. The
  `JobQueue` abstraction gets a backend swap; SQLite stays for local dev
  and tests. New tables: `accounts`, `api_keys` (hash only), `packages`
  (manifest + trust tier + uploader + license declarations),
  `models` (dedup by `source_digest`), `reports` (takedown/flags).
- **Artefact store**: content-addressed filesystem layout (`sha256[0:2]/
  sha256[2:4]/sha256`) to start; S3-compatible backend later without
  changing URLs (hash-addressed names are backend-agnostic). Dedup is
  automatic: identical uploads share blobs.
- **Compatibility policy**: accept current `spec_version` fully;
  accept N−1 read-only (browse/query, no compare); reject older with a
  clear message (same hard-reject discipline as compare). Fingerprints
  are immutable once registered — re-uploads create new revisions.
- **Query performance**: the current per-request fingerprint parse is
  fine for a gallery start; add a materialized per-model stats cache
  (records/scatter payloads) refreshed on ingest when p95 latency says so
  — measure first, per our own conventions.

## 6. UI/UX — beautiful is a requirement, not polish

The current UI is functional; a public site must also be *legible and
inviting*. Concrete deliverables, not vibes:

- **Design system first**: a small token set (dark scientific-atlas
  aesthetic to match the terrain renders: deep background, one accent
  family, tabular numerals, generous whitespace), real typography
  (no system-font stack), consistent spacing scale. One CSS file, no
  framework, still zero client JS beyond htmx.
- **Landing page**: what a weight atlas *is* in one screen — hero sheet
  render, three-sentence explainer, search box, trending/recent scans.
- **Gallery**: cards built from existing sheet PNGs (they are genuinely
  beautiful and unique) + model family badges + trust-tier badge + key
  stats (tensors, spectral-norm range). Filter by family, quant, tier.
- **Model pages**: keep the tab structure (it works), but with the
  design system, proper empty states, and the trust tier + provenance
  block (source hashes, tool/spec versions) prominently shown.
- **Upload flow**: drag-and-drop `.wasc` → progress → verification
  report (tier assigned, dedup hits named) → published page. Every
  failure mode gets a human sentence, never a traceback.
- **Responsive**: gallery and model pages must read on a phone; sheets
  are wide by nature — horizontal scroll regions, not broken layouts.
- **Accessibility floor**: alt text on renders, keyboard-navigable tabs,
  contrast-checked palette (the degenerations banner pattern already
  exists — extend it).

## 7. Policy, legal, ops (the non-code half — do not skip)

- **Terms + contribution policy**: scans are derived data; uploader
  warrants redistribution rights; scan payload defaults CC-BY-4.0 (as
  in Phase 0 packages); model-license field is self-declared.
- **Takedown/report flow**: per-model report button → moderation queue →
  tombstone-with-reason (never silent delete). Rights-holder contact
  published.
- **Privacy**: stored per upload — account id, package bytes, manifest
  metadata. No weights, no model files, no telemetry beyond standard
  server logs (documented, short retention).
- **Hosting**: separate host/container from dev machines (the dev box
  already OOMed once under scan load — the site must never share fate
  with scans). The production host is rented hardware with full root,
  Debian, running an Apache2/MySQL/MariaDB/Varnish/PHP (LAMP) stack
  (see §10.1): terminate TLS at Apache, reverse-proxy to uvicorn, and
  put Varnish in front of gallery/static renders (sheet PNGs are
  immutable content-addressed bytes — cache aggressively). HTTPS via
  Let's Encrypt on the registered domain from day one. Daily artefact+DB
  backups, restore tested once before launch.
- **Monitoring**: health endpoint (exists: `/api` discovery), disk/error
  alerting, ingest-queue depth. On-call is whoever holds the deploy keys
  — name them before launch, not after the first incident.

## 8. API for agents (the "agents do this on their own" surface)

- Auth: per-account API keys (`Authorization: Bearer`), created/rotated/
  revoked in settings; scopes from day one (`upload`, `read`) even if
  only two scopes exist. Browser accounts via GitHub OAuth for v1
  (see §10.2); email-based accounts are a later, separate project.
- Endpoints: `POST /api/v1/packages` (upload, returns verification
  report + trust tier), `GET /api/v1/packages/{id}/status` (async
  ingest pipeline), plus the existing read API (spec v0.2 + scatter/
  records) under the same auth with rate limits.
- Machine docs: an agent-oriented quickstart (key → upload → poll
  status → query), curled end-to-end in CI so it never rots.
- The existing `/api` discovery body advertises every endpoint —
  agents find new surface without human docs.

## 9. Milestones (small, shippable, in order)

- **M1 — private beta on real infra**: MariaDB swap, auth (GitHub
  OAuth + API keys), HTTPS on the domain, upload pipeline with all §4
  guards, seed content = our own scans (Flash-Next, ladder series, BDH).
  Invite-only keys. Exit: three humans + one agent complete upload→
  browse→query without operator help.
- **M2 — public read**: landing + gallery + search + model pages with
  the §6 design system. No public upload yet. Exit: a stranger
  understands the site in 60 seconds (hallway test, actually run it).
- **M3 — public upload**: quotas, moderation queue, trust tiers live,
  takedown flow exercised once in staging. Exit: full threat-model
  table (§4) signed off row by row.
- **M4 — hardening + handover**: backups restored once for real,
  on-call named, runbook written, dependency audit clean. Exit: the
  operator can go on holiday.

Beauty is M2's exit criterion, not a later phase — an ugly public
launch teaches visitors the data is not worth looking at.

## 10. Open questions — RESOLVED 2026-09-05 (operator decisions)

1. **Hosting**: rented hardware, full root, Debian GNU/Linux, with an
   operating Apache2/MySQL/MariaDB/Varnish/PHP (LAMP) stack already on
   the box. Consequences recorded above: MariaDB instead of Postgres
   (§5), TLS termination + reverse proxy at Apache, Varnish in front of
   gallery/statics (§7). The app runs as its own uvicorn service behind
   this stack; the PHP side is untouched.
2. **Identity**: email preferred long-term, but **GitHub OAuth suffices
   for v1**. No email system in v1 scope; record it as a named later
   project, not scope creep.
3. **Seed content license check**: **confirmed done by operator** — seed
   scans may be published.
4. **Domain**: registered domain is **saga-ai.org**; weight-atlas lives
   at **`atlas.saga-ai.org`** (subdomain, per operator leaning —
   isolates cookies/security policy from the apex and future services).
   OAuth callbacks, cookie domain, and staging URLs all derive from it;
   wire staging to the subdomain early.
5. **Moderation staffing**: **confirmed staffed** — queue + runbook
   (M3/M4) still required, but triage ownership is settled.

## 11. Verification plan

- Threat-model table (§4) as a checklist, each row with a test or a
  documented control; evil-zip + hostile-name fuzz suites in CI.
- Upload→verify→browse→query exercised end-to-end in staging per
  milestone exit criteria (scripted, not click-tested).
- Full existing suite stays green throughout (794 tests + Phase 0
  package tests); new suites: auth, quotas, trust-tier transitions,
  MariaDB backend parity with SQLite.
- Load test the ingest path with the largest real package (Flash-Next
  full profile) before M3; quota behaviour verified by actually
  hitting quotas.
