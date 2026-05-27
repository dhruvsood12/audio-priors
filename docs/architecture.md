# Architecture

Text diagram of the audio-priors pipeline.

```
+---------------------+
| Kaggle datasets     |  maharshipandya/-spotify-tracks-dataset
| (3 sources, ~1.3M   |  rodolfofigueroa/spotify-12m-songs
|  rows pre-dedup)    |  paradisejoy/top-hits-spotify-from-20002019
+---------+-----------+
          |  scripts/download_data.py
          v
+---------------------+
| data/raw/<slug>/    |  one CSV per source
| MANIFEST.json       |  per-source sha256 + row count
+---------+-----------+
          |  src/audio_priors/data.py
          |    load_*  -> per-source harmonization
          |    deduplicate (case-insensitive track+artist)
          |    validate_ranges (popularity, tempo, duration_ms, [0,1] features)
          v
+---------------------+
| data/processed/     |  pandera-validated canonical schema
| tracks.parquet      |  1,165,501 rows, 15 columns + source
+---------+-----------+
          |
          |  src/audio_priors/labels.py
          |    sticky_top_q(q=0.20), popularity_z, popularity_z_by_genre
          |
          +--------------------+--------------------+----------------------+
          |                    |                    |                      |
          v                    v                    v                      v
+-----------------+  +-----------------+  +-----------------+  +---------------------+
| Phase 3:        |  | Phase 4:        |  | Phase 5:        |  | Phase 6:            |
| scripts/        |  | scripts/        |  | scripts/        |  | scripts/            |
| train.py        |  | interpret.py    |  | recommend_eval  |  | prepare_app.py      |
|                 |  |                 |  | .py             |  |                     |
| 5 models +      |  | SHAP, perm,     |  | FAISS over      |  | bake artifacts ->   |
| bootstrap CIs   |  | logistic CIs,   |  | L2-normalized   |  | outputs/models/     |
| -> outputs/     |  | calibration     |  | features        |  | for the demo        |
| tables/metrics. |  | -> outputs/     |  | -> outputs/     |  |                     |
| csv             |  | figures + json  |  | tables/recom-   |  | app/streamlit_app.py|
|                 |  |                 |  | mender_metrics  |  | reads them on boot  |
+-----------------+  +-----------------+  +-----------------+  +---------------------+
                                                                       |
                                                                       v
                                                              +---------------------+
                                                              | Streamlit demo on   |
                                                              | http://localhost:   |
                                                              | 8501                |
                                                              |                     |
                                                              | A. track search +   |
                                                              |    SHAP bars        |
                                                              | B. sliders + top-10 |
                                                              |    nearest tracks   |
                                                              +---------------------+
```

Every Phase 3-6 box reads `data/processed/tracks.parquet` directly. Phase 6 also reads the LightGBM and FAISS artifacts produced by `prepare_app.py`. CI (Phase 8) runs Phase 7 tests on every push and builds the Dockerfile that wraps `src/audio_priors/cli.py`.

## Module dependency graph

```
audio_priors/
  __init__.py
  schemas.py       (pandera models, no internal deps)
  features.py      (constants + selectors, no internal deps)
  labels.py        (no internal deps)
  data.py          -> schemas
  models.py        (no internal deps; uses sklearn / lightgbm / xgboost)
  evaluation.py    (no internal deps; uses sklearn metrics)
  interpret.py     (no internal deps; uses sklearn calibration + shap)
  recommend.py     (no internal deps; uses faiss + sklearn)
  cli.py           -> dispatches to scripts/*.py
```

Each module imports only from the standard library and pinned third-party dependencies. No circular imports.
