# Resume bullets

Three bullets and supporting material for ML / DS recruiter conversations.
Every number traces to a file under `outputs/tables/` in this repo;
caveats in [MODEL.md](../MODEL.md) and the open issue tracker name what
the bullets glide over.

## Headline (resume top line)

ML / Data Science | Cold-start music classifier under a leakage-controlled
evaluation protocol with artist-cluster bootstrap CIs

## Three bullets

- Built a cold-start popularity classifier on a 1.16M-track Spotify
  corpus and measured it under an artist-grouped split (no artist on
  both sides of the fence), the popularity cutoff fit on the train
  fence only, and the decision threshold frozen on validation. LightGBM
  reached ROC-AUC 0.692 (95% CI 0.675, 0.708) on 16,805 held-out tracks
  across 6,349 unseen artists (`outputs/tables/metrics.csv`).
- Quantified label and split leakage directly rather than asserting it:
  a paired artist-cluster bootstrap on identical test rows put the
  random-split model's advantage from having seen an artist's other
  tracks at +0.090 AUC for LightGBM and +0.257 for random forest, with
  logistic at +0.001 (CI spans zero) because it cannot memorize timbre
  (`outputs/tables/split_delta.csv`). Reported the negative result that
  a one-hot genre prior at 0.836 still beats every audio-only model.
- Hardened the protocol with an invariant test suite that runs as a
  blocking CI job: artist-disjoint splits, a train-only label cutoff,
  a frozen F1 threshold, and artist-cluster resampling are each pinned
  by a test that is red against the old code and green against the new
  (`tests/test_invariants.py`). Hyperparameters are tuned once with a
  grouped inner CV and frozen to `configs/hparams.json` so the
  split-arm comparison isolates the split alone.

## LinkedIn blurb (about 60-90 words)

audio-priors asks how far Spotify's 10 audio features carry cold-start
popularity prediction, and answers it under a protocol built to resist
leakage: an artist-grouped split, a train-only label cutoff, a frozen
decision threshold, and artist-cluster bootstrap CIs. LightGBM reaches
ROC-AUC 0.692 (95% CI 0.675, 0.708) on unseen artists; a paired
bootstrap shows a random split would have inflated that by 0.090 AUC
through artist memorization. The honest finding stands: audio alone
does not beat a one-hot genre prior at 0.836. An invariant suite gates
every claim in CI.

## Interview talking points

1. **The artist-memorization gap is measured, not assumed.** Scoring
   the same grouped-test rows with both the grouped and random models
   gives a paired artist-cluster bootstrap delta of +0.090 AUC for
   LightGBM, +0.257 for random forest, +0.001 for logistic
   (`outputs/tables/split_delta.csv`). The tree models were partly
   fingerprinting artists; the grouped 0.692 is the figure that
   survives.
2. **Three leaks fixed in one protocol.** The label cutoff now fits on
   the train fence only (was full-corpus), the split groups on artist
   (was random), and the F1 threshold is frozen on validation (was the
   test-set optimum). Each is pinned by an invariant test that is red
   against the old code, so the fix cannot silently regress.
3. **Why cluster bootstrap.** Tracks cluster by artist, so a row
   bootstrap treats correlated rows as independent and reports a CI
   that is too tight. The grouped arm resamples whole artists; the
   random arm shows both so the difference is visible in
   `metrics.csv` (artist-cluster CIs run a touch wider, as expected).
4. **Hyperparameters frozen across arms.** If Optuna ran per arm, the
   grouped-vs-random delta would absorb two hyperparameter sets. One
   grouped search with a `GroupKFold` inner CV, frozen to
   `configs/hparams.json` and refit read-only, keeps the delta about
   the split.
5. **Isotonic recalibration cut Brier from 0.2118 to 0.1614** on the
   grouped LightGBM, a 24% reduction, with the reliability curve
   tracking the diagonal (`outputs/tables/calibration.json`). Raw Brier
   is not yet comparable across models with different imbalance
   handling; that recalibration is a tracked follow-up.
6. **Standardized logistic coefficients with 200-resample bootstrap
   CIs**: instrumentalness -0.380 (-0.403, -0.357), acousticness
   -0.282 (-0.312, -0.252), liveness -0.245 (-0.270, -0.223). All three
   negative with CIs excluding zero, so the sticky class is vocal,
   produced, and studio-recorded
   (`outputs/tables/logistic_coefficients.csv`).
7. **Ranking claims use a paired bootstrap, not CI overlap.** Two
   independent CIs overlapping does not settle a paired comparison;
   `paired_cluster_bootstrap_delta` evaluates both models on identical
   resamples so between-sample variance cancels. The same machinery
   backs the arm delta and any model-vs-model claim.

## Risks the recruiter will probe (and how to answer)

- **"Is your train/test split leaking artist-level information?"**
  It was, in v0.1. It now reports both an artist-grouped split (the
  headline) and a random split, with `GroupShuffleSplit` on
  `artist_name` keeping every artist on one side of every fence,
  including the validation carve. The gap is measured: +0.090 AUC for
  LightGBM on a paired artist-cluster bootstrap. The shipped figure is
  the grouped 0.692, and a test pins the disjointness. (Fixed, #20.)
- **"Did you fit the popularity quantile on the full corpus before
  splitting?"** It did in v0.1. Now `sticky_top_q_train_threshold`
  fits the cutoff on the train fence only and applies the frozen
  scalar to every slice, so test popularity never moves its own label.
  A test constructs a corpus where the train-only and full-corpus
  cutoffs disagree and asserts the pipeline uses the train-only one.
  (Fixed, #21.)
- **"Your F1 is at the F1-optimal threshold on the test set. Is that
  an in-sample upper bound?"** It was. The deployable column is now F1
  at a threshold frozen on the validation slice, with F1@0.5 beside it;
  the oracle F1 stays one release as a single labeled upper bound to
  quantify the old column. Freezing cost under one F1 point. A test
  makes a re-optimizing implementation fail on separable data.
  (Fixed, #22.)
- **"You compare Brier across models with different class-imbalance
  handling. Is that apples-to-apples?"** Not yet. Logistic / LGBM / RF
  use `class_weight='balanced'` and XGB uses `scale_pos_weight`, which
  distort the probability scales differently, so the raw Brier column
  is not comparable across models. The fix is a uniform OOF isotonic
  recalibration (grouped folds in the grouped arm) before any Brier
  comparison; the per-model calibrated LightGBM Brier of 0.161 is the
  defensible number today. (Tracked, #24.)
- **"Per-genre extremes: how many positives back those?"** A handful,
  and the card says so. The top and bottom genres rest on near-
  degenerate class balance (comedy 4 positives, pop-film 3 negatives),
  so they are descriptive only. The fix is `min_positive>=10` and
  `min_negative>=10` floors plus per-genre bootstrap CIs and a
  Benjamini-Hochberg correction, flagging only genres whose corrected
  interval excludes 0.5. (Tracked, #23.)
- **"What is the cold-start signal actually doing if a one-hot genre
  baseline beats it by 14 AUC points?"** That is the contribution.
  The genre prior at 0.836 is the strong baseline you reach for when
  genre is known; the grouped audio model at 0.692 is the prior you
  ship when it is not, and it is honest about an artist the model has
  never seen. The gap is the measured value of categorical genre
  information beyond what audio summaries recover.
