# What Makes a Song Sticky? — Project Summary

## Goal

Identify which **Spotify audio features** co-occur with **high popularity** in a track-level dataset, using popularity as a **public proxy** for broad replayability (“stickiness”). This is **not** a measure of addiction, individual replays, skips, or saves.

## Dataset

- **File:** `data/raw/spotify_tracks.csv` (track/artist/genre metadata where available, `popularity` 0–100, standard Spotify audio features, `duration_ms`).
- **Cleaning:** Column harmonization via `COLUMN_MAP` in `src/data_prep.py`, duplicate removal, missing-value drops on required numeric fields, range checks, optional duration unit fix (seconds → ms).

## Stickiness Definition

- **`popularity_z`:** z-score of popularity within the sample.
- **`sticky`:** 1 if popularity ≥ **80th percentile** (top ~20%), else 0.

## Analysis Approach

1. Cleaning & target creation (`01_data_cleaning.ipynb`).
2. EDA: distributions, sticky vs non-sticky, correlations, genre slices if present (`02_eda.ipynb`).
3. Models: **logistic regression** (standardized features, `class_weight='balanced'`) and **random forest** (`class_weight='balanced'`), 80/20 stratified split, `random_state=42` (`03_modeling.ipynb`).

## Key Findings

_Results from the latest executed pipeline (see `outputs/tables/model_metrics.csv`; re-run notebooks after changing the raw CSV)._

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic regression | ~0.55 | ~0.24 | ~0.54 | ~0.33 | **~0.59** |
| Random forest | ~0.79 | ~1.0 | ~0.01 | ~0.02 | ~0.55 |

- **Stronger on ranking / discrimination:** **Logistic regression** shows higher **ROC-AUC** and **F1** here; the random forest achieves high accuracy mostly by predicting the majority class (sticky is ~20% — check confusion matrices in `outputs/figures/`).
- **EDA (correlation with popularity):** **Danceability** and **energy** show the clearest positive linear associations in this run; **valence** is near zero; **duration** is negligible at the linear level — patterns are **weak**, which is common for real-world music data.
- **Interpretability (logistic, positive coefficients toward sticky):** **Danceability** and **energy** rank among the top positive drivers; **loudness** and **instrumentalness** lean negative in this sample (see `09_logistic_coefficients.png`).

## Limitations

- Popularity ≠ individual replay behavior; marketing and artist reach confound the label.
- Audio features alone explain only a **small fraction** of variance; expect modest ROC-AUC.
- **Correlation is not causation**; coefficients are associative.

## Product Relevance (Suggestive)

- **Cold-start priors** when behavioral logs are missing.
- **Playlist / session heuristics** as soft signals next to collaborative filtering.
- **Skip-risk exploration** — audio-only features are incomplete; treat as supplementary.

## Next Steps

- External validation on another snapshot or market.
- Add release timing / playlist reach if available.
- Calibrate decision thresholds for precision–recall tradeoffs in a product setting.
