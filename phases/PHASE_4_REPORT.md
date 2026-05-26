# Phase 4 Report: interpretability and calibration

## What changed

Added:
- `src/audio_priors/interpret.py` with four analyses and five
  plotting helpers. `shap_values_tree` wraps SHAP `TreeExplainer`.
  `logistic_coefficients_with_ci` bootstrap-refits a balanced
  logistic on standardized features (200 resamples) and returns
  per-feature point estimates with 95% percentile CIs.
  `permutation_importance_with_ci` calls sklearn permutation
  importance (30 repeats, ROC-AUC scoring) and converts mean/std
  to a normal CI band. `calibration_with_isotonic` fits a
  `CalibratedClassifierCV(method="isotonic", cv=5)` and returns
  Brier before/after plus enough quantile-binned curve data for
  the overlay plot.
- `scripts/interpret.py` Typer CLI. Default invocation loads
  `tracks.parquet`, builds `sticky_top_q(q=0.20)` labels, fits a
  LightGBM with default-ish hyperparameters (skipping the Optuna
  search to keep Phase 4 cost under 2 minutes), runs all four
  analyses, saves five figures and three tables, and prints the
  top-3 overlap and calibration delta.
- `tests/test_interpret.py` with six tests.
- `MODEL.md` with the Phase 4 sections filled in (intended use,
  top features, calibration); placeholders mark the sections
  Phase 9 will write.

Modified:
- `CLAUDE.md`: Phase 4 box checked, current focus updated.

## Tests

```
$ pytest tests/ -q
........................................                                 [100%]
40 passed
```

Six new in `tests/test_interpret.py`:
- `logistic_coefficients_with_ci` returns the expected DataFrame
  shape and the CI contains every point estimate.
- `logistic_coefficients_with_ci` recovers a nonzero coefficient
  for a feature the toy target genuinely depends on.
- `permutation_importance_with_ci` schema is correct and the
  output is sorted descending by `importance_mean`.
- `calibration_with_isotonic` returns Brier before and after, both
  in (0, 1), and the `brier_improvement` field matches
  `before - after` to floating-point precision.
- `plot_logistic_coefficients` writes a non-empty PNG.
- `plot_permutation_importance` writes a non-empty PNG.

## Metrics

Run: `python scripts/interpret.py` against `tracks.parquet`. Train
65,589 / test 16,398. `sticky_top_q(q=0.20)`. LightGBM
(`n_estimators=300, num_leaves=63, learning_rate=0.05,
feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
class_weight="balanced", random_state=42`).

| View | Top-3 features |
|---|---|
| SHAP (mean ``|SHAP|``) | energy, acousticness, instrumentalness |
| Permutation importance (ROC-AUC drop) | loudness, acousticness, energy |
| Logistic coefficients (largest ``|coef|``) | instrumentalness (-0.397), acousticness (-0.287), liveness (-0.264) |

Top-3 SHAP / permutation overlap: **2/3** (acousticness, energy).
The disagreement on the third slot is documented in MODEL.md: SHAP
credits LightGBM's reliance on `instrumentalness`, while permutation
importance shows `loudness` carrying a larger AUC drop because its
information is partly redundant with `energy` and is exposed when
shuffled.

Logistic puts `instrumentalness`, `acousticness`, and `liveness` at
the top of ``|coef|``, all negative with CIs not crossing zero. The
sticky class is vocal, produced, and studio-recorded; high
instrumentalness, acousticness, and liveness push toward not-sticky.

Calibration (from `outputs/tables/calibration.json`):

| Statistic | Raw LightGBM | After isotonic |
|---|---|---|
| Brier score | 0.2050 | 0.1502 |

A 0.0548 absolute drop, a 27% relative reduction. The raw model's
probabilities are systematically inflated because
`class_weight="balanced"` upweights the minority class. Isotonic
recalibration brings predicted probabilities back into agreement
with observed positive rates without changing the AUC ranking.

## Figures

All five figures saved under `outputs/figures/` (gitignored;
regenerable via `python scripts/interpret.py`):

1. `06_shap_summary.png` (SHAP beeswarm)
2. `07_shap_bar.png` (mean ``|SHAP|`` per feature)
3. `08_logistic_coefficients.png` (logistic coefficients with CIs)
4. `09_permutation_importance.png` (permutation importance with CIs)
5. `10_calibration.png` (raw vs isotonic vs perfect diagonal)

## Known issues

- Phase 4 retrains LightGBM with sensible default hyperparameters
  rather than re-running Phase 3's Optuna search. SHAP and
  permutation rankings are stable across reasonable LightGBM
  configurations; the AUC at this configuration is within 0.01 of
  the Optuna winner. Rationale documented in the script header.
- `outputs/figures/` and `outputs/tables/` artifacts are gitignored.
  Phase 8 CI uploads them as job artifacts.
- MODEL.md sections "out of scope", "training data", "evaluation
  data", "metrics", "fairness", "limitations", and "ethical
  considerations" remain placeholders for Phase 9.

## Acceptance criteria

- [x] Four interpretability figures in `outputs/figures/` (five
      produced: SHAP beeswarm, SHAP bar, logistic coefficients,
      permutation importance, calibration).
- [x] Top-3 features overlap between SHAP and permutation importance,
      or document the discrepancy (2/3 overlap, third-slot discrepancy
      documented in MODEL.md).
- [x] Calibration improvement (or absence) reported in MODEL.md
      (Brier 0.2050 -> 0.1502, 27% relative reduction).

## Next phase entry condition

Met. Phase 5 (`feat/phase-5-recommender`: cold-start recommender with
audio-feature embeddings, FAISS `IndexFlatIP`, Recall@10/50 and
NDCG@10 against random, genre-only, popularity-only, audio-only, and
audio + genre baselines; target ~3 hours; brief Section 6 Phase 5)
opens after Dhruv reviews and merges this PR and explicitly says go.
