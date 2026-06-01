# Model card

A living model card for the audio-priors LightGBM classifier. Phase 4
contributes the calibration and interpretability sections; Phase 9
fills in the remaining model-card prose.

## Intended use

Predict whether a Spotify track is in the top quintile of popularity
(`sticky_top_q(q=0.20)`) from ten audio features alone. The intended
use is as a prior in cold-start music recommendation, where the
listener-side signal (skips, replays, saves) is missing and the
service falls back to audio-derived heuristics.

## Out of scope

- Real-time Spotify Web API integration (the live endpoint was
  deprecated for new applications in November 2024).
- Deep-learning audio embeddings (MERT, CLAP, JukeBox); we use the
  pre-computed 10-feature Spotify summary.
- Online A/B testing infrastructure.
- Multi-user personalization or session modeling.
- Live deployment beyond local Streamlit and the GHCR Docker image.

## Top features

Three views of feature importance, computed against a LightGBM
trained on 65,589 rows (Phase 3's labeled subset, `q=0.20`) and
evaluated on the 16,398-row hold-out set.

SHAP top 3 (mean ``|SHAP|`` over the test set, beeswarm in
`outputs/figures/06_shap_summary.png`, bar in
`outputs/figures/07_shap_bar.png`):

1. `energy`
2. `acousticness`
3. `instrumentalness`

Permutation top 3 (ROC-AUC drop when shuffled, 30 repeats, bars in
`outputs/figures/09_permutation_importance.png`,
`outputs/tables/permutation_importance.csv`):

1. `loudness`
2. `acousticness`
3. `energy`

Logistic coefficients (200 bootstrap resamples on standardized
features, full table in `outputs/tables/logistic_coefficients.csv`,
bars in `outputs/figures/08_logistic_coefficients.png`):

| Feature | Coef | 95% CI |
|---|---|---|
| instrumentalness | -0.397 | (-0.420, -0.373) |
| acousticness | -0.287 | (-0.320, -0.256) |
| liveness | -0.264 | (-0.287, -0.243) |

### Overlap and discrepancy

SHAP and permutation importance agree on two of three features:
`acousticness` and `energy` appear in both top-3 lists. They disagree
on the third: SHAP picks `instrumentalness`, permutation picks
`loudness`. The discrepancy is consistent with how each method
attributes signal:

- SHAP credits a feature for the marginal change it makes to every
  prediction, including correlated structure. LightGBM uses
  `instrumentalness` heavily because the tracks at the extreme of
  that feature (near 0 or near 1) carry a strong prior on the label.
- Permutation importance credits a feature for the loss when its
  signal is destroyed, which downweights features whose information
  is also available through correlated channels. `loudness`
  correlates with `energy`, so shuffling `loudness` reveals
  information the model would otherwise have leaned on.

The logistic coefficient ordering names a third concern. Standardized
logistic puts `instrumentalness`, `acousticness`, and `liveness` at
the top of `|coef|`, all with negative sign and CIs that do not
contain zero. The sticky class is **vocal, produced, studio-recorded**;
high `instrumentalness` and `liveness` push toward not-sticky. This is
consistent with the SHAP top three but adds direction.

The Phase 4 acceptance "top-3 SHAP and permutation overlap" sits at
2/3; the disagreement is documented above.

## Calibration

`CalibratedClassifierCV(method="isotonic", cv=5)` fit on the training
set, applied to the test set. Numbers from
`outputs/tables/calibration.json`; figure in
`outputs/figures/10_calibration.png`.

| Statistic | Raw LightGBM | After isotonic |
|---|---|---|
| Brier score | 0.2050 | 0.1502 |

Improvement: **0.0548** (a 27% relative reduction). The raw model's
probabilities are systematically inflated because
`class_weight="balanced"` upweights the minority class during
training. Isotonic recalibration shrinks the predicted probabilities
back toward observed positive rates without changing the AUC ranking.
The reliability curve in the figure shows the raw curve bowing well
above the diagonal at low predicted probabilities, and the isotonic
curve tracking the diagonal closely.

## Training data

`data/processed/tracks.parquet` after deduplication and range
validation: 1,165,501 rows total, of which 81,987 carry a non-null
`popularity` value and all ten audio features. The 80/20 stratified
split (`random_state=42`) yields 65,589 training rows and 16,398
test rows. Class balance at `q=0.20`: 21.09% positive. See
`DATA.md` for the full per-source breakdown, the Spotify Web API
deprecation note, and the four preprocessing decisions.

## Evaluation data

The 16,398-row hold-out from the 80/20 stratified split,
`random_state=42`, positive rate 0.211. Every metric in this card
plus `outputs/tables/metrics.csv` is computed against this set.

## Metrics

| Model | ROC-AUC | 95% CI | PR-AUC | F1 (best thr) | Brier |
|---|---|---|---|---|---|
| **genre_prior** | **0.852** | **(0.846, 0.859)** | 0.610 | 0.586 | 0.118 |
| lightgbm | 0.711 | (0.702, 0.721) | 0.394 | 0.442 | 0.202 |
| xgboost | 0.708 | (0.699, 0.718) | 0.392 | 0.440 | 0.210 |
| random_forest | 0.706 | (0.696, 0.715) | 0.390 | 0.434 | 0.151 |
| logistic | 0.629 | (0.618, 0.639) | 0.299 | 0.384 | 0.237 |

No audio-only model beats the genre prior. The audio-priors
framing treats this as the cold-start ceiling: when no genre is
known, the LightGBM at 0.71 AUC is the prior we ship.

## Fairness

Per-genre AUC ranges from 0.29 (`study`) to 0.97 (`mpb`) across
the 105 genres with at least 20 test rows. The bottom five
(`study`, `breakbeat`, `salsa`, `pop, R&B`, `j-idol`) are
anti-predicted: the same audio signals that flag sticky tracks in
the global sample flag the opposite within those buckets. The
finding is real but rests on small group sizes (20 to 50 test rows
each); larger samples would say whether the direction holds.

Full table in `outputs/tables/per_genre_auc.csv`. Recommend
treating predictions in the anti-predicted genres as low
confidence regardless of the model's reported probability.

## Limitations

- Popularity is a market signal, not a quality signal. It mixes
  artist reach, marketing spend, and timing. The model predicts
  market success conditional on the existing 2020-2022 streaming
  landscape; it does not predict whether a track is good.
- Audio features predate the November 2024 Spotify Web API
  deprecation. We do not call the live API. Newer tracks released
  after the snapshot are out of distribution.
- The 125-genre slice is uneven. Per-genre AUCs at small n are
  noisy and the anti-predicted buckets in the fairness section
  rest on 20-50 test rows.
- Calibration is fit on a CV split of the training set; per-genre
  recalibration would likely tighten reliability in specific
  buckets but is out of scope here.
- The cold-start framing assumes genre is unknown at inference
  time. When genre is reliably known, the genre prior is a much
  stronger baseline than the audio model.

## Ethical considerations

- **Label leakage via popularity-by-genre.** Popularity is
  correlated with genre, so a model that "predicts popularity from
  audio" partly predicts genre from audio. The cold-start framing
  is more defensible than a "what makes a song addictive" framing
  because it explicitly positions the audio score as one input to a
  larger recommender, not a quality verdict.
- **Implications for unsigned artists.** A system that scores
  tracks on a popularity proxy can reinforce existing distribution
  advantages. Down-stream uses should pair the audio score with a
  recency boost or an exposure-fairness layer.
- **No listener behavior used.** The cold-start framing deliberately
  excludes skip and replay data. This is honest about what audio
  alone can do, but it also means the model cannot capture
  listener-level preferences. Personalization is out of scope here
  (see "Out of scope").

## Intended use (retrieval)

Beyond the classifier head, the same 10-feature audio embedding
supports cold-start nearest-neighbor retrieval. The intended use is
to seed playlist generation or recommendation candidate lists for
newly-uploaded tracks whose listener history is empty: given a query
track, return K similar tracks by audio cosine to bootstrap further
recommender stages.

## Retrieval metrics

90/10 stratified hold-out (`random_state=42`), 7,645 evaluable
queries (those with at least one relevant training row). Relevance
per query Q: `(R.genre == Q.genre OR R.artist == Q.artist) AND
R.sticky`. Bootstrap 1,000 percentile resamples over queries.

| Baseline | Recall@10 | 95% CI | NDCG@10 |
|---|---|---|---|
| **audio_genre** (audio KNN within genre) | **0.0186** | (0.0176, 0.0197) | **0.2312** |
| audio_genre_weighted | 0.0182 | (0.0171, 0.0194) | 0.2300 |
| genre_only | 0.0180 | (0.0169, 0.0190) | 0.2249 |
| audio_only | **0.00147** | (0.00129, 0.00166) | 0.0208 |
| audio_only_weighted | 0.00113 | (0.00099, 0.00129) | 0.0173 |
| popularity_only | 0.000394 | (0.000315, 0.000484) | 0.00626 |
| random | 0.000127 | (0.000096, 0.000158) | 0.00187 |

Audio-only Recall@10 beats random by **11.6x** with non-overlapping
independent CIs. Audio + genre vs genre-only: the independent CIs
overlap by about 0.005 of width, but a **paired-bootstrap** on the
per-query NDCG@10 difference gives **+0.00632 (95% CI +0.00203,
+0.01057)**, strictly positive. The within-genre lift is real but
small enough that the independent-CI test is underpowered to detect
it.

Full numbers in `outputs/tables/recommender_metrics.csv`.

## Training procedure

| Component | Setting |
|---|---|
| Train/test split | 80/20 stratified on `sticky`, `random_state=42`. The test rows are the same set across all five classifiers. |
| Hyperparameter search | Optuna TPE with 30 trials, 300-second time budget, 5-fold stratified CV inside the training fence. LightGBM searches `num_leaves`, `min_child_samples`, `learning_rate`, `feature_fraction`, `bagging_fraction`. XGBoost searches `max_depth`, `min_child_weight`, `learning_rate`, `subsample`, `colsample_bytree`. |
| Class imbalance | Logistic / LightGBM / Random Forest use `class_weight='balanced'`. XGBoost uses `scale_pos_weight = n_neg / n_pos`. Note: this is one source of the Brier-comparison caveat in the Limitations section. |
| Calibration | `CalibratedClassifierCV(method='isotonic', cv=5)` fit on the training set; evaluated on the test set. |
| Bootstrap | 1,000 percentile resamples per metric (ROC-AUC, PR-AUC, F1, Brier). Resamples with a single class are skipped so AUC stays defined. |
| Phase 4 retrain | The interpretability pipeline retrains LightGBM with default hyperparameters (`n_estimators=300, num_leaves=63, learning_rate=0.05`) rather than the Optuna winner. SHAP and permutation rankings are stable across reasonable LightGBM configurations; documented in `phases/PHASE_4_REPORT.md`. |

## Known fix-in-progress items

These are tracked as GitHub issues and disclosed up front in
[docs/RESUME_BULLETS.md](docs/RESUME_BULLETS.md) for interview
prep:

- Artist-grouped split (`StratifiedGroupKFold` on `artist_name`)
  to make the cold-start framing honest; current random split lets
  the same artist sit in both folds.
- Train-only popularity quantile (`sticky_top_q` currently fits on
  the full corpus before the split).
- F1 reported at a fixed validation-set threshold, with F1@0.5 as
  the deployable fixed-policy number alongside.
- Per-genre AUC: `min_positive>=10` filter plus Wilson/DeLong per-genre
  CIs and Benjamini-Hochberg correction over the 105 genre tests.
- Like-for-like Brier comparison: recalibrate every model with the
  same isotonic CV before reporting raw Brier.

## Contact

Dhruv Sood, `d2sood@ucsd.edu`. Issues:
<https://github.com/dhruvsood12/audio-priors/issues>.
