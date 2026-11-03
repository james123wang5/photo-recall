# Changelog

Dates from the project's docs / 日期来自项目文档。

## Unreleased — 2026-11-03 · First benchmark runs and pipeline fixes
- Public-benchmark harness `scripts/eval_retrieval.py` (PhotoBench, Flickr1k, NegBench, ChatIR, CountBench, DOCCI, CIRCO, DISBench adapters; per-channel ranks, existence signal / AUROC, per-stage index timing) and offline fusion replay `scripts/replay_fusion.py`.
- Configurable per-family fusion weights; default SigLIP weight 0.1 (WeMM 1.0).
- Offline GPS → administrative-region channel and Chinese date/holiday/season parsing (`app/metadata.py`); boundary data is prepared locally by `scripts/prepare_geography.py` and is not committed.
- Optional resident models (`PHOTORECALL_KEEP_MODELS=1`), order-independent face grouping, ORACLE people upper-bound switch (answer leakage, off by default).
- Recall stage, visual index only, PhotoBench validation (100 queries per album): hit@1 0.755 / 0.833 / 0.674 on albums 1–3 (album 1 was 0.447 before), about 1 s per query. Full pipeline (verify / rerank / OCR / multi-turn) not yet benchmarked; people and landmark search remain weak and are paused pending research.

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
