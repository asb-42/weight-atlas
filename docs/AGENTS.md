# AGENTS.md — docs

## Purpose

Project documentation: architecture, user manual, roadmap/backlog, model
families, dated proposals, and progress/code-review reports.

## Ownership

- `ARCHITECTURE.md` (durable design contracts), `user_manual.md` (usage +
  spec reference), `ROADMAP.md`, `BACKLOG.md`, `MODEL_FAMILIES.md`,
  `DEPLOYMENT.md` (living Phase 1 server runbook — update in place as the
  deployment evolves, unlike dated proposals),
  `activity_protocol.v1.md`, dated `*.md` proposals (incl.
  `2026-08-16_weight-atlas-api-spec-v0.2.md` — LLM query API design),
  `reports/` (dated progress + review reports).

## Local Contracts

- **ARCHITECTURE.md is the design rail**: it records durable pipeline
  contracts (spec extensions, alignment semantics, render determinism,
  compare modes). Any change that alters a contract must update it — it is the
  human-readable counterpart of `specs/`.
- **Dated documents are immutable**: proposals and reports are named
  `YYYY-MM-DD_<topic>.md` and describe a point in time; do not rewrite them to
  reflect later changes. New updates get a new dated file or a BACKLOG entry.
- **Keep in sync**: `user_manual.md` embeds spec examples — when `specs/*.json`
  changes, update the matching snippets here (same single-line style).
- **No diary entries**: ROADMAP/BACKLOG hold forward-looking items; ARCHITECTURE
  holds stable contracts. Delete stale text, don't annotate it.

## Work Guidance

- Write proposals before large changes (`docs/YYYY-MM-DD_topic.md`), then
  fold the accepted design into ARCHITECTURE.md + specs.
- Reports belong in `reports/` with dated filenames.

## Verification

- Manual review: docs are prose, not code. Cross-check spec snippets against
  `specs/*.json` and pipeline behavior against tests.