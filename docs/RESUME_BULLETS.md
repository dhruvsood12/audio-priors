# Resume bullets

Three bullets and supporting material for ML / DS recruiter conversations.
Every number traces to a file under `outputs/tables/` in this repo;
caveats in [MODEL.md](../MODEL.md) and the open issue tracker name what
the bullets glide over.

## Headline (resume top line)

ML / Data Science | Cold-start music recommender, calibrated audio classifier with bootstrap CIs

## Three bullets

- Built a cold-start music recommender on a 1.16M-track Spotify corpus.
  FAISS `IndexFlatIP` retrieval over L2-normalized audio embeddings hit
  Recall@10 of 0.0186 against random's 0.000127 across 7,645 evaluable
  queries, an 11.6x lift with non-overlapping 95% bootstrap CIs
  (`outputs/tables/recommender_metrics.csv`).
- Trained and tuned a five-model panel (logistic, random forest,
  LightGBM, XGBoost, genre-prior baseline) on 81,987 popularity-labeled
  tracks with 30-trial Optuna search and 5-fold CV. LightGBM reached
  ROC-AUC 0.711 (95% CI 0.702, 0.721); reported the negative result
  that a one-hot genre prior at 0.852 beats every audio-only model by
  about 14 AUC points (`outputs/tables/metrics.csv`).
- Calibrated LightGBM probabilities with 5-fold isotonic CV, cutting
  Brier from 0.2050 to 0.1502, a 27% relative reduction with the
  reliability curve tracking the diagonal
  (`outputs/tables/calibration.json`). Ranked features across SHAP,
  30-repeat permutation, and bootstrapped logistic coefficients;
  documented the 2-of-3 overlap and the loudness vs instrumentalness
  split in MODEL.md.

## LinkedIn blurb (about 60-90 words)

audio-priors (v0.1.0) asks how far Spotify's 10 audio features carry
cold-start music recommendation. I trained five classifiers on 81,987
labeled tracks (LightGBM at ROC-AUC 0.711, 95% CI 0.702 to 0.721),
calibrated probabilities with isotonic CV (Brier 0.205 to 0.150, a 27%
relative cut), and built a FAISS retriever that beats random Recall@10
by 11.6x across 7,645 queries with non-overlapping bootstrap CIs. The
honest finding: audio alone cannot beat a one-hot genre prior at
0.852, which reframes the work as a cold-start ceiling study.

## Interview talking points

1. **LightGBM ROC-AUC 0.711, 95% CI (0.702, 0.721)** from 1,000
   bootstrap resamples on the 16,398-row hold-out, and why the
   non-overlapping CI against the 0.852 genre-prior baseline is the
   load-bearing comparison
   (`outputs/tables/metrics.csv`).
2. **Isotonic recalibration cut Brier from 0.2050 to 0.1502** on the
   LightGBM head. The raw probabilities sat well above the diagonal
   because `class_weight='balanced'` upweights minorities during
   training; isotonic shrinks them back toward observed positive rates
   without changing AUC ranking (`outputs/tables/calibration.json`).
3. **FAISS IndexFlatIP over L2-normalized features beat random by
   11.6x on Recall@10** (0.0186 vs 0.000127) across 7,645 queries.
   Paired-bootstrap NDCG@10 of audio + genre over genre-only is
   +0.00632 (95% CI 0.00203, 0.01057), strictly positive even where
   the independent CIs overlap
   (`outputs/tables/recommender_metrics.csv`).
4. **Feature attribution sits at 2 of 3 overlap** across methods: SHAP
   picks energy / acousticness / instrumentalness, permutation picks
   loudness / acousticness / energy. The loudness vs instrumentalness
   split is consistent with loudness and energy being correlated, so
   permutation rewards the redundant channel
   (MODEL.md, `outputs/tables/permutation_importance.csv`).
5. **Standardized logistic coefficients with 200-resample bootstrap
   CIs**: instrumentalness -0.397 (-0.420, -0.373), acousticness
   -0.287 (-0.320, -0.256), liveness -0.264 (-0.287, -0.243). All
   three negative and CIs exclude zero, so the sticky class is vocal,
   produced, and studio-recorded
   (`outputs/tables/logistic_coefficients.csv`).
6. **Per-genre AUC ranges from 0.29 (`study`) to 0.97 (`mpb`)** across
   105 genres with n>=20. The extremes have very small n_positive and
   are flagged in MODEL.md as needing Wilson/DeLong CIs and a
   Benjamini-Hochberg correction (tracked as a follow-up issue).
7. **Released v0.1.0 across ten phase-scoped PRs** with conventional
   commits and a `git filter-repo` authorship pass. Demo boots in
   under five seconds with warm caches and serves per-query inference
   in under one second.

## Risks the recruiter will probe (and how to answer)

- **"Is your train/test split leaking artist-level information?"**
  Yes. The current split is stratified random with no group key, so
  the same artist's tracks can sit in both folds and the audio model
  partly learns artist fingerprints. The 0.711 number is the
  random-split figure, not a true cold-start figure. Fix on deck:
  `StratifiedGroupKFold` on `artist_name`, reported under both random
  and grouped splits, with the grouped number as the cold-start
  headline. (Open issue.)
- **"Did you fit the popularity quantile on the full corpus before
  splitting?"** Yes. `sticky_top_q` fits on the full labeled corpus,
  which leaks test-set popularity into the label threshold. The right
  move is to fit the threshold on train only and apply it to test.
  Rerunning metrics under that protocol is the next change. (Open
  issue.)
- **"Your F1 is at the F1-optimal threshold on the test set. Is that
  an in-sample upper bound?"** Yes. The F1 column is an oracle
  threshold and the bootstrap CI estimates variance of that oracle,
  not of a deployable policy. The honest fix is to pick the threshold
  on a validation slice of train, freeze it, then bootstrap F1 at the
  fixed threshold. F1 at threshold 0.5 will sit alongside as the
  fixed-policy number. (Open issue.)
- **"You compare Brier across models with different class-imbalance
  handling. Is that apples-to-apples?"** No. Logistic / LGBM / RF use
  `class_weight='balanced'` and XGB uses `scale_pos_weight`, which
  distort the probability scales differently. Either recalibrate
  everything with the same isotonic CV before reporting Brier, or
  drop the raw Brier column. The **0.150 post-calibration LightGBM
  Brier is the comparable number**.
- **"Per-genre extremes like mpb 0.97 and study 0.29: how many
  positives back those?"** A handful. The current filter is only
  n>=20 with no `min_positive` constraint, so the extremes are
  essentially single-rank statistics. Fix: `min_positive>=10` plus
  Wilson/DeLong per-genre CIs and Benjamini-Hochberg over 105 tests,
  and only flag genres whose corrected CI excludes 0.5. (Open issue.)
- **"What is the cold-start signal actually doing if a one-hot genre
  baseline beats it by 14 AUC points?"** That is the contribution.
  The genre prior is the strong baseline you reach for when genre is
  known; the audio model is the prior you ship when it is not. The
  0.711 number is the ceiling of pre-behavior audio features, and the
  11.6x retrieval lift over random is the deployable signal in the
  cold-start setting.
