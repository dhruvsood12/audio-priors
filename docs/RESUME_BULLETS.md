# Resume bullets

Three bullets derived from the v0.1.0 metrics, each defensible
against a 30-minute technical interview. Numbers come from
`outputs/tables/metrics.csv` and `outputs/tables/recommender_metrics.csv`
in this repo.

---

- **Shipped `audio-priors v0.1.0`**: an audio-feature classification
  and cold-start retrieval study over a 1.16M-track corpus combined
  from three Kaggle Spotify datasets. Five-model panel (genre prior,
  scaled logistic, balanced random forest, LightGBM and XGBoost with
  30-trial Optuna search and 5-fold CV) with 1,000-resample bootstrap
  CIs on ROC-AUC, PR-AUC, F1 at the F1-optimal threshold, and Brier.
  LightGBM reaches ROC-AUC **0.711 (95% CI 0.702, 0.721)** on a 16,398-row
  test set; the genre-prior baseline hits **0.852 (95% CI 0.846, 0.859)**.
  Reported the honest negative result that audio features alone do
  not beat a categorical genre prior; documented the three reasons in
  `MODEL.md`.

- **Built a FAISS `IndexFlatIP` cold-start retriever** over
  standardized, L2-normalized 10-feature audio embeddings against a
  90/10 split. Compared five baselines (random, genre-only,
  popularity-only, audio-only, audio + genre) on Recall@10/50 and
  NDCG@10 with bootstrap CIs over 7,645 evaluable queries.
  Audio-only Recall@10 beats random by **11.6x with non-overlapping
  CIs**. Paired-bootstrap NDCG@10 of audio + genre over genre-only
  is **+0.00632 (95% CI +0.00203, +0.01057)**, strictly positive
  even though independent CIs overlap. Surfaced and reported the
  standardize-before-L2-normalize footgun that initially flattened
  the audio signal.

- **Engineering surface**: 76-test pytest suite at 78% coverage with
  `--cov-fail-under=70` pinned in pyproject; 4 Hypothesis property
  tests on the label functions; pre-commit hooks (`ruff`,
  `ruff-format`, `nbstripout`, plus standard file checks); CI matrix
  on Python 3.10 / 3.11 / 3.12 with `pip-audit`, Docker build, and
  em-dash / banned-word style gates; multi-stage Dockerfile published
  to GHCR on `v*` tag push; cached Streamlit demo (cold boot 4.1s,
  per-query under 1s) wiring LightGBM predictions, SHAP per-feature
  contributions, and FAISS nearest-neighbor retrieval.
