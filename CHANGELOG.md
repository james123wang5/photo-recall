# Changelog

Dates from the project's docs / 日期来自项目文档。

## v0.6.0 — 2026-10-27 · Full search pipeline
- New `app/flow/`: file scan → layered index → multi-channel recall → keep candidates → on-demand rerank / original-image verification → bounded expansion → user feedback.
- Independent `flow_*` tables; old index and photos untouched. Collection tasks with coverage ledger.
- **Developed and statically checked only** — not run on real photos, no tests or benchmarks executed.

## v0.5.0 — 2026-10-26 · Local hybrid retrieval
- Hybrid retrieval, evaluation datasets, source/paper audits (SCM, Lap, unified retrieval framework).

## v0.4.0 — 2026-10-16 → 10-21 · WeMM + CORE
- WeMM-Embedding-2B main index, CORE-Reranker-2B rerank, SigLIP 2 local recall, YOLO count lower-bound, original-image detail checks; M1 Pro / 16 GB measurements.

## v0.1 – v0.3 — 2026-10-13 → 10-15
- First web UI and SigLIP index, per-image verification flow (v0.3), accuracy/speed research, Immich & market study.
