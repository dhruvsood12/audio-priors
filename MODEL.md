# Model card

A model card for the audio-priors LightGBM classifier, reported under
the v2 evaluation protocol: an artist-grouped split, a popularity
threshold fit on the train fence only, an F1 threshold frozen on a
validation slice, hyperparameters tuned once and frozen across both
split arms, and artist-cluster bootstrap CIs. The protocol is specified
once below and enforced by `tests/test_invariants.py`.

## Intended use

Predict whether a Spotify track is in the top quintile of popularity
(top 20% by `popularity`, cutoff fit on the train fence) from ten audio
features alone. The intended use is as a prior in cold-start music
recommendation, where the listener-side signal (skips, replays, saves)
is missing and the service falls back to audio-derived heuristics.
Cold start means unseen artists as well as missing behavior, so the
headline figure comes from the artist-grouped split.

## Out of scope

- Real-time Spotify Web API integration (the live endpoint was
  deprecated for new applications in November 2024).
- Deep-learning audio embeddings (MERT, CLAP, JukeBox); we use the
  pre-computed 10-feature Spotify summary.
- Online A/B testing infrastructure.
- Multi-user personalization or session modeling.
- Live deployment beyond local Streamlit and the GHCR Docker image.

## Top features

Three views of feature importance, computed against a LightGBM trained
on the grouped arm's train fence (65,182 rows) with the frozen
hyperparameters and evaluated on the 16,805-row grouped hold-out.

SHAP top 3 (mean ``|SHAP|`` over the test set, beeswarm in
`outputs/figures/06_shap_summary.png`, bar in
`outputs/figures/07_shap_bar.png`):

1. `instrumentalness`
2. `energy`
3. `acousticness`

Permutation top 3 (ROC-AUC drop when shuffled, 30 repeats, bars in
`outputs/figures/09_permutation_importance.png`,
`outputs/tables/permutation_importance.csv`):

1. `instrumentalness`
2. `loudness`
3. `acousticness`

Logistic coefficients (200 bootstrap resamples on standardized
features, full table in `outputs/tables/logistic_coefficients.csv`,
bars in `outputs/figures/08_logistic_coefficients.png`):

| Feature | Coef | 95% CI |
|---|---|---|
| instrumentalness | -0.380 | (-0.403, -0.357) |
| acousticness | -0.282 | (-0.312, -0.252) |
| liveness | -0.245 | (-0.270, -0.223) |

### Overlap and discrepancy

SHAP and permutation importance agree on two of three features:
`instrumentalness` and `acousticness` appear in both top-3 lists. They
disagree on the third: SHAP picks `energy`, permutation picks
`loudness`. The discrepancy is consistent with how each method
attributes signal:

- SHAP credits a feature for the marginal change it makes to every
  prediction, including correlated structure. LightGBM uses
  `instrumentalness` heavily because the tracks at the extreme of
  that feature (near 0 or near 1) carry a strong prior on the label.
- Permutation importance credits a feature for the loss when its
  signal is destroyed, which downweights features whose information
  is also available through correlated channels. `loudness` and
  `energy` correlate, so shuffling either reveals information the
  model would otherwise have leaned on through the other.

The logistic coefficient ordering names a third concern. Standardized
logistic puts `instrumentalness`, `acousticness`, and `liveness` at
the top of `|coef|`, all with negative sign and CIs that do not
contain zero. The sticky class is **vocal, produced, studio-recorded**;
high `instrumentalness` and `liveness` push toward not-sticky. This is
consistent with the SHAP top three but adds direction. The
loudness-vs-energy split, and single-feature credit assignment within
the energy/loudness/acousticness cluster generally, is the subject of
the collinearity work tracked for v0.2.1.

## Calibration

`CalibratedClassifierCV(method="isotonic", cv=5)` fit on the grouped
train fence, applied to the grouped test set. Numbers from
`outputs/tables/calibration.json`; figure in
`outputs/figures/10_calibration.png`.

| Statistic | Raw LightGBM | After isotonic |
|---|---|---|
| Brier score | 0.2118 | 0.1614 |

Improvement: **0.0504** (a 24% relative reduction). The raw model's
probabilities are systematically inflated because
`class_weight="balanced"` upweights the minority class during
training. Isotonic recalibration shrinks the predicted probabilities
back toward observed positive rates without changing the AUC ranking.
The reliability curve in the figure shows the raw curve bowing well
above the diagonal at low predicted probabilities, and the isotonic
curve tracking the diagonal closely. This per-model calibration is not
yet a like-for-like Brier comparison across the panel; that recalibration
is tracked for v0.2.1 (issue #24), so the raw Brier column in the
metrics table below is not comparable across models.

## Training data

`data/processed/tracks.parquet` after deduplication and range
validation: 1,165,501 rows total, of which 81,987 carry a non-null
`popularity` value and all ten audio features. Each protocol arm splits
these 81,987 rows 64/16/20 into train_fit / validation / test with no
stratification (the label does not exist at split time under the
train-only-threshold protocol). The grouped arm groups on `artist_name`
so no artist crosses a fence; the random arm does not. The popularity
cutoff is the top-quintile boundary of the train fence (fit+val),
popularity 53 in both arms. See `DATA.md` for the full per-source
breakdown, the Spotify Web API deprecation note, and the four
preprocessing decisions.

## Evaluation data

The held-out test slice of each arm: 16,805 rows (6,349 unseen artists)
in the grouped arm, 16,398 in the random arm. Realized positive rates
are reported per slice in `metrics.csv` rather than forced to `q`: train
0.207 / test 0.227 in the grouped arm (artist popularity clusters, so
grouped test prevalence drifts up about 2 points), train 0.211 / test
0.212 in the random arm. Unseen-genre test rows (a genre whose artists
all landed in test, so the genre prior falls back to the global rate)
number 3 in the grouped arm and 6 in the random arm: negligible.

## Metrics

Artist-grouped arm (the cold-start figure), with artist-cluster
bootstrap CIs:

| Model | ROC-AUC | 95% CI | PR-AUC | F1 (frozen thr) | F1@0.5 | Brier |
|---|---|---|---|---|---|---|
| **genre_prior** | **0.836** | **(0.818, 0.853)** | 0.601 | 0.588 | 0.450 | 0.128 |
| lightgbm | 0.692 | (0.675, 0.708) | 0.387 | 0.438 | 0.437 | 0.209 |
| xgboost | 0.691 | (0.674, 0.707) | 0.384 | 0.441 | 0.440 | 0.211 |
| random_forest | 0.687 | (0.671, 0.703) | 0.382 | 0.435 | 0.329 | 0.172 |
| logistic | 0.633 | (0.615, 0.649) | 0.313 | 0.407 | 0.405 | 0.236 |

Random arm (same models, random split), with row bootstrap CIs:

| Model | ROC-AUC | 95% CI | PR-AUC | F1 (frozen thr) | F1@0.5 | Brier |
|---|---|---|---|---|---|---|
| **genre_prior** | **0.855** | **(0.849, 0.861)** | 0.613 | 0.587 | 0.476 | 0.118 |
| lightgbm | 0.721 | (0.712, 0.730) | 0.395 | 0.450 | 0.451 | 0.203 |
| xgboost | 0.718 | (0.709, 0.727) | 0.391 | 0.447 | 0.446 | 0.206 |
| random_forest | 0.716 | (0.707, 0.725) | 0.396 | 0.444 | 0.357 | 0.165 |
| logistic | 0.647 | (0.637, 0.656) | 0.308 | 0.395 | 0.398 | 0.234 |

No audio-only model beats the genre prior under either split. The
audio-priors framing treats the grouped LightGBM at 0.692 AUC as the
cold-start figure we ship: it is what audio recovers about an artist
the model has never seen. The F1 column is at a threshold frozen on the
validation slice; F1@0.5 is the parameter-free policy. `metrics.csv`
also carries the in-sample oracle F1 as a single labeled upper bound
(LightGBM grouped: 0.445), retained one release to quantify the v0.1
column it replaces; freezing the threshold costs under one F1 point.

### Split comparison and the artist-memorization gap

| Protocol | LightGBM ROC-AUC | 95% CI |
|---|---|---|
| v0.1 published (random stratified, full-corpus cutoff, oracle F1) | 0.711 | (0.702, 0.721) |
| v2 random (train-only cutoff, frozen threshold) | 0.721 | (0.712, 0.730) |
| **v2 grouped (cold-start figure)** | **0.692** | **(0.675, 0.708)** |

Scoring the same grouped test rows with both arms' models (a paired
artist-cluster bootstrap, `outputs/tables/split_delta.csv`) isolates
what training on an artist's other tracks buys: random forest +0.257
(0.244, 0.270), LightGBM +0.090 (0.084, 0.096), XGBoost +0.070 (0.064,
0.075), genre prior +0.010 (0.005, 0.017), logistic +0.001 (-0.001,
0.003, CI spans zero). The tree models memorize artist timbre; logistic
cannot, so it neither gains from the random split nor loses under
grouping. Every model-vs-model and arm-vs-arm claim in this card uses a
paired cluster bootstrap, not independent-CI overlap.

## Fairness

Per-genre AUC on the grouped arm (LightGBM) ranges from 0.18
(`pop-film`) to 0.94 (`comedy`) across the 103 genres with at least 20
test rows. The extremes are not yet trustworthy: the top entries rest
on near-degenerate class balance (`comedy` has 4 positives in 224 rows,
`classical` has 3) and the bottom entries on the mirror image
(`pop-film` has 124 positives in 127 rows, so 3 negatives). At those
counts AUC is a near-single-rank statistic and the apparent
anti-prediction may be sampling noise.

The honest reading waits on per-genre CIs with a `min_positive` and
`min_negative` floor and a Benjamini-Hochberg correction over the
roughly 80 surviving genres, tracked for v0.2.1 (issue #23). Until then,
read the full table in `outputs/tables/per_genre_auc.csv` (both arms) as
descriptive, not as a list of confirmed anti-predicted genres. The one
defensible mid-table observation is that genres with balanced classes
and adequate n (for example `sleep`, 50 positives in 124 rows, AUC 0.93)
are well predicted from audio.

## Limitations

- Popularity is a market signal, not a quality signal. It mixes
  artist reach, marketing spend, and timing. The model predicts
  market success conditional on the existing 2020-2022 streaming
  landscape; it does not predict whether a track is good.
- Audio features predate the November 2024 Spotify Web API
  deprecation. We do not call the live API. Newer tracks released
  after the snapshot are out of distribution.
- The genre slice is uneven. Per-genre AUCs at small n are noisy,
  and the extremes in the fairness section rest on near-degenerate
  class balance; treat them as descriptive until the CI work in
  issue #23 lands.
- Calibration is fit on a CV split of the train fence; per-genre
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

Three configurations isolate how much of the v0.1 audio retrieval lift
was the query artist matching itself. All write to
`outputs/tables/recommender_metrics.csv`.

- **legacy**: the v0.1 protocol unchanged (full-corpus label, stratified
  random 90/10 split, `(same_genre OR same_artist) AND sticky`
  relevance, no exclusion). Reproduces the v0.1 table.
- **random + excluded**: random split, but the query artist's tracks
  leave both the candidate set and the relevance denominator, and
  relevance is genre-only. Holds the split fixed so the drop isolates
  artist self-matching.
- **grouped + excluded**: the honest cold-start figure. Artist-grouped
  query split, genre-only relevance, query-artist exclusion, and an
  artist-cluster bootstrap over query artists.

Headline (grouped + excluded), 7,809 evaluable queries:

| Baseline | Recall@10 | 95% CI | NDCG@10 |
|---|---|---|---|
| **genre_only** | **0.0182** | (0.0173, 0.0194) | 0.2350 |
| audio_genre (audio KNN within genre) | 0.0180 | (0.0170, 0.0192) | **0.2396** |
| audio_only | 0.00126 | (0.00109, 0.00143) | 0.0165 |
| popularity_only | 0.000618 | (0.00041, 0.00090) | 0.00852 |
| random | 0.000150 | (0.00011, 0.00019) | 0.00217 |

audio_only Recall@10 across the three configurations: 0.00147 (legacy,
11.6x random) to 0.00140 (random + excluded, 9.5x) to 0.00126 (grouped
+ excluded, 8.4x). The artist effect on retrieval is real but modest,
in contrast to the classifier where grouping cost 0.090 AUC: same-genre
matches dominate the relevance set, so removing artist self-matches
trims the audio lift by about 15% rather than collapsing it. The
query-side bootstrap clusters by artist in the grouped arm, since one
artist contributes several correlated queries there.

A note on the within-genre comparison: genre_only and audio_genre sit
within each other's CIs at Recall@10. audio_genre leads on NDCG@10
(0.2396 vs 0.2350), meaning audio ordering helps rank the genre-matched
sticky tracks, but the Recall@10 gap is not separable at this sample.
The honest reading is that audio adds ranking quality within a genre,
not retrieval coverage beyond it.

## Training procedure

| Component | Setting |
|---|---|
| Split | `audio_priors.splits.protocol_split`: 64/16/20 train_fit / validation / test, no stratification. Grouped arm groups on `artist_name` (`GroupShuffleSplit`, plus a grouped fit/val carve); random arm mirrors the structure without groups. Reported under both arms; grouped is the headline. |
| Label threshold | Top-quintile popularity cutoff fit on the train fence (fit+val) only via `sticky_top_q_train_threshold`, then frozen and applied to every slice. Cutoff is popularity 53 in both arms. |
| Hyperparameter search | Optuna TPE, 30 trials, run ONCE by `scripts/tune.py` on the grouped train_fit with `GroupKFold(5)` on `artist_name` as the inner CV, never touching the validation carve. Winners frozen to `configs/hparams.json` (sha recorded per metrics row); both arms refit from that file. LightGBM searches `num_leaves`, `min_child_samples`, `learning_rate`, `feature_fraction`, `bagging_fraction`; XGBoost searches `max_depth`, `min_child_weight`, `learning_rate`, `subsample`, `colsample_bytree`. |
| F1 threshold | Chosen on the validation slice (grouped in the grouped arm) via `pick_f1_threshold_on_train`, frozen, then F1 evaluated at that fixed value on test. F1@0.5 reported beside it; the in-sample oracle F1 is a single labeled upper bound, never bootstrapped. |
| Class imbalance | Logistic / LightGBM / Random Forest use `class_weight='balanced'`. XGBoost uses `scale_pos_weight = n_neg / n_pos`. This distorts the probability scales differently per model, so the raw Brier column is not comparable across models (issue #24). |
| Calibration | `CalibratedClassifierCV(method='isotonic', cv=5)` fit on the train fence; evaluated on the test set. Per-model only; like-for-like cross-panel recalibration is issue #24. |
| Bootstrap | 1,000 resamples per metric, seed 42. Grouped-arm CIs resample artists with replacement (cluster bootstrap); the random arm reports a row bootstrap as primary plus an artist-cluster CI beside it. Single-class resamples are skipped and counted per row. Ranking and arm-delta claims use a paired cluster bootstrap on identical rows. |
| Interpretability retrain | `scripts/interpret.py` refits LightGBM from the same frozen hyperparameters on the grouped train fence, so the calibration and attribution figures share the metrics protocol. |

## Known fix-in-progress items

Resolved in v0.2.0 (this card): artist-grouped split (#20), train-only
popularity cutoff (#21), frozen F1 threshold (#22), artist-disjoint
retrieval evaluation (#25). Remaining, tracked as GitHub issues and
disclosed in
[docs/RESUME_BULLETS.md](docs/RESUME_BULLETS.md):

- Per-genre AUC: `min_positive>=10` and `min_negative>=10` floors plus
  per-genre bootstrap CIs and a Benjamini-Hochberg correction over the
  surviving genre tests (issue #23).
- Like-for-like Brier: recalibrate every model with the same OOF
  isotonic procedure (grouped folds in the grouped arm) before any
  cross-model Brier comparison (issue #24).

## Contact

Dhruv Sood, `d2sood@ucsd.edu`. Issues:
<https://github.com/dhruvsood12/audio-priors/issues>.
