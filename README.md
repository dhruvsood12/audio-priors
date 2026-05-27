# audio-priors

Audio-feature priors for cold-start music recommendation. A study of how far Spotify audio features alone carry popularity discrimination and cold-start track retrieval, reported with bootstrap confidence intervals and calibration.

## Status

Rebuild in progress through Phase 6. The data pipeline (Phase 1), label design (Phase 2), modeling (Phase 3), interpretability (Phase 4), cold-start recommender (Phase 5), and Streamlit demo (Phase 6) are all merged. CI, Docker, hooks (Phase 8) and the full README rewrite (Phase 9) are still ahead. See `CLAUDE.md` for live phase status and `phases/PHASE_*_REPORT.md` for per-phase change logs.

## Quickstart

```
make install-dev    # editable install of the package and dev extras
make lint           # ruff check + ruff format --check
make test           # pytest (40+ tests)
```

## Pipelines

```
python scripts/download_data.py     # Kaggle download + processed parquet
python scripts/train.py             # five models, bootstrap CIs, metrics.csv
python scripts/interpret.py         # SHAP, permutation, calibration figures
python scripts/recommend_eval.py    # FAISS retrieval baselines
python scripts/prepare_app.py       # build pickled artifacts for the demo
```

## Streamlit demo

```
make app    # streamlit run app/streamlit_app.py
```

![audio-priors demo](outputs/figures/demo.png)

Two input modes. **Track search** picks a track from the corpus and shows the predicted sticky probability with per-feature SHAP contributions. **Manual sliders** sets the ten audio features by hand and returns the predicted probability plus the top-10 nearest tracks by audio cosine similarity. Boot is under 5 seconds after the first run; pre-built artifacts under `outputs/models/` come from `scripts/prepare_app.py`.

## License

MIT. See `LICENSE`.
