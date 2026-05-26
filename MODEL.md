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

[Phase 9 will fill in. Holding for the brief's Section 12 list:
real-time Spotify API integration, deep-learning audio embeddings,
online A/B testing, multi-user personalization, live deployment
beyond local Streamlit and the Docker image.]

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

[Phase 9 will summarize the corpus, schema, dedup, and range gates.
Reference: `DATA.md`. 1,165,501 rows in the parquet, 81,987 with
non-null `popularity` and the ten audio features.]

## Evaluation data

[Phase 9 will summarize the 80/20 stratified hold-out. 16,398 test
rows, `random_state=42`, positive rate 0.211.]

## Metrics

[Phase 9 will reference `outputs/tables/metrics.csv` and reproduce
the headline numbers with CIs. Phase 3 produced them; phase 9 chooses
the headline format.]

## Fairness

[Phase 9 will surface the per-genre AUC table at
`outputs/tables/per_genre_auc.csv`, with the ranges and the
documented anti-predicted genres.]

## Limitations

[Phase 9 will write the narrative version. Headline known
limitations: popularity is a market signal not a quality signal, the
105-genre slice is uneven, audio features cannot recover marketing
context, calibration is sample-driven, the dataset's audio features
predate the November 2024 Spotify Web API deprecation.]

## Ethical considerations

[Phase 9. Label-leakage risks via popularity-by-genre, the fairness
implications of audio-only popularity prediction for unsigned
artists, and the cold-start framing that explicitly does not rely on
listener behavior.]

## Contact

Dhruv Sood, `d2sood@ucsd.edu`. Issues:
<https://github.com/dhruvsood12/audio-priors/issues>.
