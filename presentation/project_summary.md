# What Makes a Song Sticky? — Project Summary

## Goal

Understand which **Spotify audio features** are associated with songs that are especially **sticky** in aggregate, where stickiness is approximated with a **public proxy** (Spotify **popularity**), not individual replay, skip, or save behavior.

## Dataset

- **Source (expected):** `data/raw/spotify_tracks.csv` (you provide).
- **Key fields:** track/artist metadata (optional `genre`), `popularity` (0–100), and standard Spotify audio features (e.g. danceability, energy, valence, tempo, loudness, speechiness, acousticness, instrumentalness, liveness, `duration_ms`).

## Stickiness Definition

- **Continuous:** `popularity_z` — z-score of `popularity` within the dataset.
- **Binary (`sticky`):** 1 if popularity is at or above the **80th percentile** (top ~20%), else 0.

This is a **pragmatic proxy**: popularity reflects broad reach and staying power in the market, but it is **not** a direct measure of repeated listening or “addiction.”

## Analysis Approach

1. **Cleaning & validation** — column harmonization, duplicates, missingness, range checks (`01_data_cleaning.ipynb`).
2. **EDA** — distributions, sticky vs non-sticky comparisons, correlations, optional genre slices (`02_eda.ipynb`).
3. **Modeling** — logistic regression (interpretable linear effects on standardized features) and random forest (nonlinear baseline), plus optional linear views on `popularity_z` (`03_modeling.ipynb`).

## Key Findings

_Update after running the notebooks on your CSV._

- **Classification (sticky):** [e.g. ROC-AUC / F1 for logistic vs RF — fill from `outputs/tables/model_metrics.csv`.]
- **Directional patterns:** [e.g. which features align positively/negatively with the proxy in coefficients / importance plots.]
- **Surprises / null results:** [Weak effects are still informative — note them honestly.]

## Limitations

- **Popularity ≠ replay rate** — marketing, artist fanbase, playlist placement, and release timing confound the outcome.
- **No individual-level engagement** — skips, saves, and session-level repeats are unavailable in typical public extracts.
- **Genre labels** can be coarse or inconsistent.
- **Correlation is not causation** — models describe association in this sample, not mechanisms.

## Product Relevance (Suggestive, Not Definitive)

- **Cold-start ranking** — weak priors on new tracks when behavioral data is sparse.
- **Playlist generation / sequencing** — features may act as soft constraints alongside collaborative filters.
- **Skip-risk estimation** — audio-derived signals as a fallback when logs are missing (never a substitute for real feedback).
- **Session-aware recommendation** — combine with context; audio features alone are incomplete.

## Next Steps

- Incorporate **temporal** features (release date, chart momentum) if available.
- Add **hierarchical or mixed models** by genre/artist with partial pooling.
- Validate proxy outcomes against **proprietary engagement** data in a product setting (gold standard).
