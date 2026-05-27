# Phase 9 Report: README, DATA.md, MODEL.md, architecture

## What changed

Modified:
- `README.md` rewritten from scratch in the brief's required
  structure: one-line pitch, four badges (CI, license, Python,
  coverage), headline results table with bootstrap CIs, hero figure
  (SHAP bar of the LightGBM model), why this matters, what it does,
  reproduce (three commands), data link, methods link, results
  table, per-genre AUC excerpt, recommender results table, Streamlit
  demo screenshot reference, limitations, future work, license and
  citation. 15 sections total. Every number traces to a file under
  `outputs/tables/`.
- `MODEL.md` Phase 9 sections filled in: out of scope (mirrors brief
  Section 12), training data (parquet stats + class balance),
  evaluation data (16,398 test rows, seed 42), metrics (5-row table
  reproduced from `outputs/tables/metrics.csv`), fairness (per-genre
  AUC range, anti-predicted buckets), limitations (5 bullets),
  ethical considerations (3 bullets covering label-leakage,
  unsigned-artist exposure, no-listener-behavior framing).
- Em-dashes removed from `phases/PHASE_7_REPORT.md` and
  `phases/PHASE_8_REPORT.md` so the style gate (em-dash grep over
  every markdown file) passes.

Added:
- `docs/architecture.md` with a text diagram of the pipeline (Kaggle
  -> parquet -> 4 downstream Phase 3-6 pipelines), and a module
  dependency graph showing no internal circular imports.

## Acceptance criteria

- [x] em-dash grep over `*.md` returns nothing.
- [x] Banned-word grep returns nothing.
- [x] README opens with one-sentence pitch and a headline table.
- [x] Every number in the README traces to a file in
      `outputs/tables/`.
- [x] DATA.md and MODEL.md complete (DATA was already final from
      Phase 1; MODEL had Phase 9 placeholders filled in here).

## Next phase

Phase 10: authorship audit, `v0.1.0` tag, GitHub Release, resume
bullets.
