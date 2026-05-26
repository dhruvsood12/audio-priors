# Phase 2 Report: EDA and label design

## What changed

Added:
- `src/audio_priors/labels.py` with four functions: `popularity_z` (global
  z-score), `popularity_z_by_genre` (per-genre z-score, NaN-genre rows go
  to an `__unknown__` bucket), `sticky_top_q` (binary label, `1` for the
  top `q` fraction by popularity), and `sticky_top_q_by_genre` (top `q`
  within each genre). `q` is validated to be in `(0, 1)`. NaN popularity
  yields a `0` label rather than NaN, on the read that a row with no
  signal is not a stickiness positive.
- `notebooks/01_eda.ipynb` with six analysis sections: missingness,
  popularity distribution, audio features by sticky label (q = 0.20),
  correlation heatmap, per-genre popularity, and threshold sensitivity.
  Five figures saved to `outputs/figures/`, one CSV saved to
  `outputs/tables/label_sensitivity.csv`.
- `tests/test_labels.py` with 7 unit tests.

Modified:
- `CLAUDE.md`: Phase 2 box checked, current focus updated.

## Tests

```
$ pytest tests/ -q
......................                                                   [100%]
22 passed
```

`tests/test_labels.py` contributes 7 of the 22:
- `popularity_z` global mean-0, unit-std on a 2,000-row random frame.
- `popularity_z` constant-input degenerate case returns zeros (not NaN
  or inf).
- `popularity_z_by_genre` per-group means are 0 within tolerance.
- `sticky_top_q` labels exactly `q * n` rows as 1 on a strictly
  increasing input.
- `sticky_top_q` raises `ValueError` for `q` outside `(0, 1)`,
  including the boundary values `0.0` and `1.0`.
- `sticky_top_q_by_genre` keeps the top `q` fraction within each
  genre, not globally.
- `sticky_top_q` returns `0` for NaN popularity rows.

## Metrics

Threshold sensitivity, logistic regression on standardized audio
features, 80/20 stratified split with `random_state=42`,
`class_weight="balanced"`. Numbers reproduce from
`outputs/tables/label_sensitivity.csv`.

| q | popularity threshold | class balance | test ROC-AUC |
|---|---|---|---|
| 0.05 | 67.0 | 0.056 | **0.718** |
| 0.10 | 61.0 | 0.107 | 0.672 |
| 0.20 | 53.0 | 0.211 | 0.629 |
| 0.30 | 46.0 | 0.316 | 0.623 |

n_train = 65,589 and n_test = 16,398 for every q. The labeled subset is
the popularity-bearing rows after dropping NaN popularity and NaN audio
features.

Audio features predict stickiness more strongly at the head of the
popularity distribution than across the bulk. At q = 0.05 the logistic
baseline is at 0.72 AUC; at q = 0.30 it sinks to 0.62. Phase 3 will
make this choice explicit and report bootstrap CIs at the chosen q.

## Figures

All saved to `outputs/figures/` and referenced from the notebook:

1. `01_missingness.png`: bar of fraction missing per column.
2. `02_popularity_distribution.png`: histogram and KDE.
3. `03_features_by_sticky.png`: 7 audio-feature boxplots split by
   `sticky_top_q(df, q=0.20)`.
4. `04_correlation_heatmap.png`: 10x10 correlation matrix over the
   feature set plus popularity.
5. `05_popularity_by_genre.png`: top 20 genres by row count, sorted
   by mean popularity.

## Known issues

- Per-genre AUC is deferred to Phase 3. The corpus has many small
  genres (1-50 rows each) where group-level AUCs are too noisy to be
  useful at the EDA stage.
- `outputs/figures/*.png` are gitignored per `.gitignore` policy.
  Anyone reviewing the PR sees the figures by running the notebook
  themselves (`MPLBACKEND=Agg jupyter nbconvert --execute
  notebooks/01_eda.ipynb`) against the parquet built in Phase 1. Phase 8
  will add CI that produces them as job artifacts.

## Acceptance criteria

- [x] `notebooks/01_eda.ipynb` executes headlessly via
      `jupyter nbconvert --execute` (verified: 16,580 bytes after
      run with `MPLBACKEND=Agg`).
- [x] `outputs/tables/label_sensitivity.csv` produced (four rows,
      one per q value, with threshold, class balance, n_train,
      n_test, and roc_auc).
- [x] At least four figures saved to `outputs/figures/` and
      referenced from the notebook (five produced).
- [x] `tests/test_labels.py` passes 4+ tests (7 passed).

## Next phase entry condition

Met. Phase 3 (`feat/phase-3-modeling`: genre-prior baseline, logistic,
random forest, LightGBM with Optuna, XGBoost with Optuna, bootstrap
CIs, per-genre AUC; target ~4 hours; brief Section 6 Phase 3) opens
after Dhruv reviews and merges this PR and explicitly says go.
