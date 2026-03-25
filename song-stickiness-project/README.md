# What Makes a Song Sticky?

End-to-end data science project: **which Spotify audio features are associated with highly “sticky” songs** when stickiness is approximated with **public popularity** (not individual replay or skip data).

---

## Project Overview

Streaming products care whether a song **pulls listeners back**—but public datasets rarely expose repeat listens, skips, or saves. This project studies **audio features** alongside **Spotify popularity** as an **observable proxy** for broad replayability and staying power. The emphasis is on **clear measurement**, **interpretable models**, and **honest limits**—not on claiming literal “addiction” or access to proprietary engagement logs.

---

## Why This Project

Direct **replay, skip, and save** signals are the gold standard for stickiness, yet they are typically **private** and context-specific. **Popularity** is imperfect, but it is **public**, comparable across tracks, and reflects aggregate market attention—making it a defensible **proxy** for an interview-ready portfolio analysis when behavioral logs are unavailable.

---

## Research Question

**Which Spotify audio features are most associated with highly sticky songs, using popularity as a public proxy for replayability?**

---

## Dataset

Place your CSV at `data/raw/spotify_tracks.csv`. The pipeline expects (or maps to) columns such as:

| Area | Columns |
|------|---------|
| Identity | `track_name`, `artist_name`, `genre` (optional) |
| Target proxy | `popularity` (0–100) |
| Audio features | `danceability`, `energy`, `valence`, `tempo`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `duration_ms` |

Column names are normalized in code; minor naming variants are handled in `src/data_prep.py`.

---

## Methodology

1. **Data cleaning** — Harmonize names, remove duplicates, handle missing values, validate ranges (`01_data_cleaning.ipynb` + `src/data_prep.py`).
2. **Stickiness targets** — `popularity_z` (z-score); binary `sticky` = 1 for tracks at or above the **80th percentile** of popularity (top ~20%).
3. **Exploratory analysis** — Distributions, sticky vs non-sticky comparisons, correlations, optional genre breakdowns (`02_eda.ipynb` + `src/visuals.py`).
4. **Classification** — Logistic regression (standardized features) and random forest; metrics include accuracy, precision, recall, F1, ROC-AUC (`03_modeling.ipynb` + `src/modeling.py`).
5. **Interpretability** — Logistic coefficients and random forest feature importance; optional linear regression / OLS on `popularity_z` for a continuous view.

---

## Repository Structure

```
song-stickiness-project/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   └── spotify_tracks.csv          # you add this
│   └── processed/
│       ├── spotify_cleaned.csv         # produced by 01
│       └── spotify_model_data.csv      # produced by 01
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data_prep.py
│   ├── features.py
│   ├── modeling.py
│   └── visuals.py
├── outputs/
│   ├── figures/
│   └── tables/
└── presentation/
    └── project_summary.md
```

---

## Key Visualizations

Produced when you run the notebooks (saved under `outputs/figures/`):

- Popularity histogram  
- Boxplots of audio features by `sticky`  
- Scatter plots (e.g. danceability, energy, duration vs popularity)  
- Correlation heatmap  
- Top 10% vs bottom 10% mean feature comparison  
- Logistic regression coefficient plot  
- Random forest feature importance  
- Confusion matrices  

Optional (if `genre` is present): average popularity by genre, sticky rate by genre, violin plot of a key feature by genre.

---

## Main Findings

**Run the notebooks on your dataset** to generate metrics and figures. After execution, summarize ROC-AUC, F1, and the main coefficient/importance directions here. Until then: *results populate from `outputs/tables/model_metrics.csv` and the modeling notebook after a full run.*

---

## Product Implications (Suggestive)

Audio-only signals are **incomplete**, but they can still support product thinking when logs are scarce:

- **Cold-start recommendation** — weak priors for new tracks before behavioral data exists.  
- **Playlist generation / sequencing** — soft constraints alongside collaborative filtering.  
- **Skip-risk estimation** — fallback features when session data is missing (never a replacement for real feedback).  
- **Session-aware recommendation** — combine with context; audio is one slice of the full picture.

---

## Limitations

- **Popularity is not the same as replay rate** — it mixes quality, marketing, artist reach, and timing.  
- **No direct skip/save/replay data** in this public framing.  
- **Correlation is not causation** — associations in one sample do not imply universal rules.  
- **Confounds** — label noise, regional effects, and **genre** metadata quality can distort patterns.

Stating these limits is part of the analysis, not an apology.

---

## How to Run

**Prerequisites:** Python 3.10+ recommended.

```bash
cd song-stickiness-project
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. Copy your CSV to `data/raw/spotify_tracks.csv`.  
2. Launch Jupyter from the **project root** (`song-stickiness-project`):  
   `jupyter lab` or `jupyter notebook`  
3. Run notebooks **in order:** `01_data_cleaning.ipynb` → `02_eda.ipynb` → `03_modeling.ipynb`.

Figures write to `outputs/figures/`; the model comparison table writes to `outputs/tables/model_metrics.csv`.

---

## Interview Summary

This project asks a **product-relevant question**—what makes tracks **stick** in the wild—using only **public audio + popularity** data. I define **stickiness** transparently as a **top-popularity proxy**, clean and validate the data, explore differences between **sticky vs non-sticky** groups, then fit **interpretable** (logistic) and **nonlinear** (random forest) models. I emphasize **limitations** (popularity confounds, no replay logs) and connect results to **recommendation systems** as **priors and heuristics**, not ground truth. The story is: **rigorous proxy measurement, clear methods, honest interpretation.**
