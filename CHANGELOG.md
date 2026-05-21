# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/). Versioning follows
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-05-19

First public release.

### Headline

Across 1545 tracks from the 2020-2021 Spotify Top 200, the best classifier
is a random forest predicting top-quintile popularity at ROC-AUC 0.594
(95% CI 0.508 to 0.674). Switching the target to top-quintile `chart_weeks`
does not lift AUC. Audio features alone are a thin signal.

### Added

- `pyproject.toml` with strict ruff, black, and mypy configurations.
- Pre-commit hooks: ruff, black, mypy, nbstripout, plus standard file checks.
- Pytest suite (30 tests) covering schema, cleaning, modeling seeding,
  evaluation, splitting, the long-stayer target, and end-to-end pipeline.
- Multi-stage `Dockerfile` and `docker-compose.yml` for headless pipeline
  runs; `scripts/run_pipeline.sh` orchestrates the three notebooks.
- CI workflow with separate `lint`, `test`, and `docker` jobs.
- Real-data path via `scripts/fetch_chart_data.py` (Kaggle CLI); synthetic
  fallback via `scripts/make_demo_data.py`.
- Bootstrap 95% confidence intervals on ROC-AUC, calibration curves, SHAP
  summary, permutation importance, and per-genre AUC tables.
- `create_long_stayer_label` derived from `chart_weeks`, alongside the
  existing `create_sticky_label` from `popularity`.
- Defensive cleaning: `clean_dataframe` drops columns that are entirely NaN.
- MIT LICENSE.

### Changed

- README rewritten as a 69-line numbers-first artifact: headline result,
  results table with bootstrap CIs, how it works, run instructions,
  limitations, references.
- CI replaces the broken `python-package-conda.yml` workflow that
  referenced a nonexistent `environment.yml` and ran pytest against zero
  discoverable tests.

### Removed

- Tracked copies of `data/processed/*.csv`; the pipeline regenerates them
  from `data/raw/` on every run.

[0.1.0]: https://github.com/dhruvsood12/SongAddiction/releases/tag/v0.1.0
