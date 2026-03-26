# What Makes a Song Sticky? — Project Summary

One-page narrative for interviews and slides. **“Sticky”** here means **high popularity in the sample** (top ~20%), used as a **public proxy** for broad replayability — not clinical addiction, individual replay counts, or skip/save behavior.

---

## Goal

Identify which **Spotify audio features** co-occur with **high popularity** at the track level, and whether simple models can separate **sticky vs not** using only audio + metadata. Findings are **associative** and **sample-specific**.

---

## Dataset

- **Input:** `data/raw/spotify_tracks.csv` — track/artist/genre (optional), `popularity` (0–100), Spotify audio features, `duration_ms`.
- **Cleaning:** `COLUMN_MAP` renaming, duplicate removal, missingness on required numerics, range validation, optional **seconds → milliseconds** fix (`src/data_prep.py`).

---

## Stickiness Definition

| Field | Definition |
|-------|------------|
| `popularity_z` | Z-score of popularity within the dataset |
| `sticky` | 1 if popularity ≥ 80th percentile (~top 20%), else 0 |

---

## Analysis Approach

1. **Cleaning** — `01_data_cleaning.ipynb` + `clean_dataframe()`.
2. **EDA** — Distributions, **KDE by sticky**, boxplots, scatters, correlation heatmap, genre panels if `genre` exists (`02_eda.ipynb`).
3. **Modeling** — Logistic regression + random forest (`class_weight='balanced'`), 80/20 stratified split, metrics + **ROC** + **precision–recall** curves (`03_modeling.ipynb`).

---

## Key Findings

| Model | ROC-AUC | F1 | Role |
|-------|---------|-----|------|
| Logistic regression | **~0.59** | **~0.33** | Better **ranking** / minority-class signal in this run |
| Random forest | ~0.55 | ~0.02 | Higher **accuracy** but weak sticky recall (majority bias) |

- **EDA:** **Danceability** and **energy** show the clearest positive correlations with popularity; **valence** ~0; **duration** ~0 — effects are **weak**, which is typical.
- **Interpretability:** Logistic coefficients point to **danceability** and **energy** as top positive drivers toward the sticky label; **loudness** / **instrumentalness** lean negative in this sample (see `outputs/figures/09_logistic_coefficients.png`).
- **Curves:** ROC and PR plots (`13_roc_curves.png`, `14_pr_curves.png`) make tradeoffs explicit under ~20% positive rate.

---

## Visual Highlights (committed after `02` / `03`)

- `01_popularity_histogram.png` — marginal popularity distribution  
- `06b_kde_popularity_by_sticky.png` — overlap of sticky vs not on the proxy  
- `04_correlation_heatmap.png` — linear associations  
- `09_logistic_coefficients.png` / `10_rf_feature_importance.png` — interpretability  
- `13_roc_curves.png` / `14_pr_curves.png` — discrimination vs chance  

---

## Limitations

- Popularity mixes quality, marketing, artist fanbase, and timing.  
- No individual-level replay/skip/save data.  
- Modest ROC-AUC: audio alone does not “explain” the market.  
- **Correlation ≠ causation.**

---

## Product Relevance (Suggestive)

Use audio features as **weak priors** for cold-start or playlist heuristics when engagement logs are missing — always subordinate to real user feedback.

---

## Next Steps

- Re-run the pipeline on your own CSV; refresh `outputs/tables/model_metrics.csv` and figures.  
- Optional: tune thresholds on PR curves for product-specific precision/recall goals.
