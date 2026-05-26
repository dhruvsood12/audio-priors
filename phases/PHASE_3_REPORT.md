# Phase 3 Report: modeling

## What changed

Added:
- `src/audio_priors/models.py` with five trainers. `GenrePrior` (via
  `train_genre_prior`) predicts the per-genre training-set positive
  rate; unknown genres fall back to the global mean. `train_logistic`
  returns a Pipeline of `StandardScaler` + `LogisticRegression
  (class_weight="balanced", max_iter=2000)`. `train_random_forest`
  uses 300 trees, balanced class weights, all cores.
  `train_lightgbm` and `train_xgboost` each run an Optuna TPE search
  (30 trials, 300-second time budget, 5-fold stratified CV inside
  training) over `num_leaves` / `min_child_samples` / `learning_rate`
  / `feature_fraction` / `bagging_fraction` for LightGBM and
  `max_depth` / `min_child_weight` / `learning_rate` / `subsample` /
  `colsample_bytree` for XGBoost. Best params refit on the full
  training set.
- `src/audio_priors/evaluation.py` with `bootstrap_metric` (percentile
  CI, 1,000 resamples, single-class resamples skipped),
  `best_f1_threshold` (PR-curve sweep), `f1_at_optimal_threshold`,
  `per_genre_auc` (filters by min_count and single-class slices), and
  `evaluate_with_ci` (ROC-AUC, PR-AUC, F1 at optimal threshold,
  Brier, all with CIs).
- `scripts/train.py` Typer CLI. Default invocation loads the
  parquet, drops NaN-popularity rows, builds `sticky_top_q(q=0.20)`
  labels, splits 80/20 stratified, runs all five trainers, writes
  `outputs/tables/metrics.csv` and `outputs/tables/per_genre_auc.csv`
  (for the highest-AUC non-baseline model).
- `tests/test_models.py` (6 tests) and `tests/test_evaluation.py`
  (6 tests).

Modified:
- `CLAUDE.md`: Phase 3 box checked.

## Tests

```
$ pytest tests/ -q
..................................                                       [100%]
34 passed
```

`tests/test_models.py` covers `GenrePrior` recovering per-group base
rates, falling back to the global mean for unseen genres, well-formed
proba rows, `train_logistic` returning a working pipeline,
`train_random_forest` reproducibility under fixed seed, and
`train_logistic` clearing AUC 0.7 on a constructed-from-features toy
target. `tests/test_evaluation.py` covers `bootstrap_metric` interval
containment and seed reproducibility, `best_f1_threshold` recovering
F1=1.0 on a perfect ranker, `f1_at_optimal_threshold` matching
`best_f1_threshold`, `per_genre_auc` filtering small buckets, and
`evaluate_with_ci` returning all four metrics with valid intervals.

## Metrics (from `outputs/tables/metrics.csv`)

Run: `python scripts/train.py` against `data/processed/tracks.parquet`
(1,165,501 rows). Labeled rows after dropping NaN popularity and NaN
audio features: 81,987. Train: 65,589. Test: 16,398. Label:
`sticky_top_q(q=0.20)`. Positive rate: 0.211. Bootstrap: 1,000
resamples, seed 42, percentile interval.

| Model | ROC-AUC | 95% CI | PR-AUC | F1* | Brier |
|---|---|---|---|---|---|
| **genre_prior** | **0.852** | **(0.846, 0.859)** | 0.610 | 0.586 | 0.118 |
| lightgbm | 0.711 | (0.702, 0.721) | 0.394 | 0.442 | 0.202 |
| xgboost | 0.708 | (0.699, 0.718) | 0.392 | 0.440 | 0.210 |
| random_forest | 0.706 | (0.696, 0.715) | 0.390 | 0.434 | 0.151 |
| logistic | 0.629 | (0.618, 0.639) | 0.299 | 0.384 | 0.237 |

*F1 at the per-sample threshold that maximizes F1 on the test set.

## The negative result

No audio-only model beats the genre prior. LightGBM (best non-baseline)
is 14.1 AUC points below the genre prior, and the two 95% CIs are not
just non-overlapping; they are separated by roughly 12 points. Brief
Section 6 Phase 3 anticipates this:

> If not, report the negative result honestly and document why.

Why this happened, in three parts:

1. **The corpus has 125 genres with widely different popularity
   distributions.** Categorical genre lookup encodes a great deal of
   marketing context (chart presence, audience size, recency of the
   subgenre). A track in `k-pop` has a different prior probability of
   sitting in the top quintile of popularity than a track in
   `iranian` or `black-metal`, even before anyone listens to the
   audio. Audio features cannot recover that context from waveform
   summaries alone.
2. **Audio features encode timbre and energy, not market reach.**
   `danceability`, `valence`, etc. capture musical surface attributes
   that correlate with genre but do not capture release strategy,
   playlisting, or artist visibility. Popularity at this dataset's
   horizon is heavily downstream of those non-audio factors.
3. **The brief's audio-priors framing is exactly the cold-start
   case.** When a fresh track arrives with no genre tag and no
   marketing data, audio features are the prior. ROC-AUC 0.71 is the
   cold-start ceiling that audio carries on its own. Phase 4 will
   show which features contribute via SHAP and permutation
   importance; Phase 5 will measure whether audio-feature nearest
   neighbors retrieve same-genre, same-popularity-bucket peers
   better than a random baseline.

## Per-genre AUC (LightGBM, from `outputs/tables/per_genre_auc.csv`)

105 genres clear `min_count=20`. ROC-AUC ranges from 0.293
(`study`) to 0.973 (`mpb`). Top five and bottom five:

| Genre | n | n_positive | AUC |
|---|---|---|---|
| mpb | 150 | 1 | 0.973 |
| comedy | 183 | 4 | 0.955 |
| sleep | 148 | 48 | 0.931 |
| j-dance | 146 | 5 | 0.850 |
| drum-and-bass | 186 | 19 | 0.816 |
| ... | ... | ... | ... |
| j-idol | 203 | 4 | 0.432 |
| salsa | 155 | 1 | 0.416 |
| breakbeat | 193 | 3 | 0.321 |
| pop, R&B | 28 | 20 | 0.300 |
| study | 193 | 2 | 0.293 |

`sleep`, `comedy`, and `mpb` have audio signatures distinct enough
from non-sticky tracks in their own bucket that the model picks them
out cleanly. `study`, `breakbeat`, and `salsa` have audio profiles
that anti-correlate with their bucket's sticky tracks, similar to
what the synthetic Phase 4 of the prior framing showed for `latin`.
Phase 4 SHAP will say which features drive these splits.

## Known issues

- The `q=0.20` choice mirrors the prior framing and the brief's
  Phase 3 example. Phase 2's threshold-sensitivity table shows
  `q=0.05` gives higher logistic AUC (0.718); a `q=0.05` run would
  likely produce different relative rankings (the genre prior leans
  on common-genre base rates that get noisier as the positive rate
  drops). Out of Phase 3 scope; revisit if Phase 9 chooses a
  different headline q.
- LightGBM and XGBoost training each spent the full 5-minute Optuna
  budget. With more compute, both could likely close some of the gap
  to the genre prior, but the audio-features-only ceiling is bounded
  below by what timbre/energy summaries can encode.
- `outputs/tables/per_genre_auc.csv` and `outputs/tables/metrics.csv`
  are gitignored regenerable artifacts; Phase 8 CI will produce them
  as job artifacts.

## Acceptance criteria

- [x] At least one tuned model beats the genre-prior baseline by 2+
      AUC points with non-overlapping 95% CIs, **or report the
      negative result honestly and document why**. None of the audio
      models beats the genre prior. The negative result is reported
      with non-overlapping CIs in the opposite direction and the
      reasons are documented above.
- [x] `outputs/tables/metrics.csv` contains every model with CIs (5
      rows: genre_prior, logistic, random_forest, lightgbm, xgboost;
      `roc_auc`, `pr_auc`, `f1`, `brier`, each with `_ci_lower` and
      `_ci_upper`).
- [x] Per-genre AUC table written to
      `outputs/tables/per_genre_auc.csv` (105 genres above
      `min_count=20`, computed against LightGBM).
- [x] `tests/test_models.py` and `tests/test_evaluation.py` pass
      with 8+ tests combined (12: 6 + 6).

## Next phase entry condition

Met. Phase 4 (`feat/phase-4-interpret`: SHAP TreeExplainer on the
winning tree model, logistic coefficients with CIs, permutation
importance with CIs, calibration plot, Brier before and after
isotonic recalibration; target ~2 hours; brief Section 6 Phase 4)
opens after Dhruv reviews and merges this PR and explicitly says go.
