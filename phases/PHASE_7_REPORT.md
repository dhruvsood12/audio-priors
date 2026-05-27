# Phase 7 Report: test coverage sweep

## What changed

Added:
- `src/audio_priors/features.py` - `AUDIO_FEATURES` constant plus
  `select_audio_features`, `feature_columns_present`, and
  `has_full_feature_set` helpers.
- `tests/test_features.py` - 5 tests covering the new module.
- `tests/test_pipeline.py` - 3 end-to-end tests on synthetic data
  (harmonize / dedup / validate round-trip, full label-to-evaluate
  pipeline, FAISS corpus build).
- `tests/test_labels_property.py` - Hypothesis property tests on
  `popularity_z`, `sticky_top_q`, and `sticky_top_q_by_genre`.
- `tests/test_recommend_more.py` - 7 tests pushing recommender
  coverage from 47% to 80%.

Modified:
- `pyproject.toml` - pytest `addopts` adds
  `--cov=src/audio_priors --cov-fail-under=70 --cov-report=term-missing`.
- `CLAUDE.md` - Phase 7 box checked.

## Tests

```
$ pytest tests/ -q
............................................................................   [100%]
76 passed
```

Coverage from `pytest --cov` against `src/audio_priors`:

| Module | Coverage |
|---|---|
| `__init__.py` | 100% |
| `data.py` | 78% |
| `evaluation.py` | 98% |
| `features.py` | 100% |
| `interpret.py` | 70% |
| `labels.py` | 94% |
| `models.py` | 51% |
| `recommend.py` | 80% |
| `schemas.py` | 100% |
| **TOTAL** | **78%** |

The `--cov-fail-under=70` gate now blocks merges below the
threshold. `models.py` sits at 51% because the Optuna search paths
are inherently expensive to test; they get exercised end-to-end
via `scripts/train.py`, which Phase 8 CI will cover.

## Acceptance criteria

- [x] `pytest --cov=src/audio_priors --cov-fail-under=70` passes
      (78% total).
- [x] Required test files per brief Section 6 Phase 7:
      `test_data.py` (11 + 3 new pipeline), `test_schemas.py` (4),
      `test_features.py` (5), `test_labels.py` (7 + 4 property tests),
      `test_models.py` (6), `test_evaluation.py` (6),
      `test_recommend.py` (6 + 7 new), `test_pipeline.py` (3).
- [x] All tests run in under 90 seconds locally
      (`pytest -q` finishes in ~30s on this machine).

## Next phase

Phase 8: CI matrix (3.10/3.11/3.12), Dockerfile, GHCR release
workflow.
