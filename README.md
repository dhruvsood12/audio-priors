# audio-priors

Audio-feature priors for cold-start music recommendation. A study of how far Spotify audio features alone carry popularity discrimination and cold-start track retrieval, reported with bootstrap confidence intervals and calibration.

## Status

Rebuild in progress. Phase 0 (hygiene and identity) is complete. Subsequent phases add the data pipeline, modeling, recommender, Streamlit demo, and full CI. See `CLAUDE.md` for phase status and `phases/PHASE_*_REPORT.md` for per-phase change logs.

## Quickstart (Phase 0)

```
make install-dev
make lint
```

Pipelines, tests, and a runnable demo arrive in later phases.

## License

MIT. See `LICENSE`.
