# audio-priors

Cold-start music recommendation from 10 Spotify audio features. When a fresh track lands with no genre tag and no listener history, how far does audio alone carry you?

**Headline:** under an artist-grouped split (no artist on both sides of the fence), a tuned LightGBM hits **ROC-AUC 0.692 (95% CI 0.675, 0.708)** on a 16,805-row hold-out. The same model under a random split reads 0.721; scoring identical test rows with both models puts the artist-memorization gap at **+0.090 AUC (95% CI 0.084, 0.096)**, so the grouped number is the honest cold-start figure and the one this project ships. Isotonic recalibration cuts Brier from 0.212 to 0.161 (a 24% relative reduction).

[![CI](https://github.com/dhruvsood12/audio-priors/actions/workflows/ci.yml/badge.svg)](https://github.com/dhruvsood12/audio-priors/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

## Results

The audio-priors framing is the cold-start case: genre is unknown at inference time, audio is what you have. Cold-start also means unseen *artists*, so the headline split groups by `artist_name`: no artist sits on both sides of any fence. The label cutoff (popularity 53, the top-quintile boundary) is fit on the train fence only and frozen; the F1 threshold is frozen on a grouped validation slice before touching test rows; grouped-arm CIs resample artists, not rows (1,000 resamples, seed 42). Every row below traces to [`outputs/tables/metrics.csv`](outputs/tables/metrics.csv), which records the full protocol per row.

Artist-grouped arm, 16,805 test rows across 6,349 unseen artists:

| Model | ROC-AUC | 95% CI (artist cluster) | PR-AUC | F1 (frozen thr) | F1@0.5 | Brier |
|---|---|---|---|---|---|---|
| **lightgbm** (audio only) | **0.692** | **(0.675, 0.708)** | 0.387 | 0.438 | 0.437 | 0.209 |
| xgboost (audio only) | 0.691 | (0.674, 0.707) | 0.384 | 0.441 | 0.440 | 0.211 |
| random_forest (audio only) | 0.687 | (0.671, 0.703) | 0.382 | 0.435 | 0.329 | 0.172 |
| logistic (audio only) | 0.633 | (0.615, 0.649) | 0.313 | 0.407 | 0.405 | 0.236 |
| genre_prior (context, not audio) | 0.836 | (0.818, 0.853) | 0.601 | 0.588 | 0.450 | 0.128 |

How much of the old headline was artist memorization? The split comparison for LightGBM:

| Protocol | ROC-AUC | 95% CI |
|---|---|---|
| v0.1 published (random stratified split, full-corpus label cutoff, oracle F1) | 0.711 | (0.702, 0.721) |
| v2 random split (train-only cutoff, frozen threshold) | 0.721 | (0.712, 0.730) |
| **v2 artist-grouped split (the cold-start figure)** | **0.692** | **(0.675, 0.708)** |

Scoring the SAME grouped test rows with both arms' models isolates the effect: the random-arm model, which saw these artists' other tracks in training, gains **+0.090 AUC (95% CI 0.084, 0.096)** over the grouped-arm model. Random forest is the heaviest memorizer at +0.257; logistic, which cannot memorize timbre fingerprints, gains +0.001 (CI spans zero). Full table in [`outputs/tables/split_delta.csv`](outputs/tables/split_delta.csv).

The F1 column is at a threshold frozen on the validation slice (never test); F1@0.5 sits beside it as the parameter-free policy. The in-sample oracle F1 the v0.1 table reported is retained in `metrics.csv` as a single labeled upper bound for one release; freezing the threshold cost under one F1 point, so the old column's problem was methodological rather than numerical.

A track is **sticky** if its Spotify popularity sits at or above the train-fence top-quintile cutoff (53). Realized positive rates are reported per slice in `metrics.csv` (train 0.207, test 0.227 in the grouped arm; artist popularity clusters, so grouped test prevalence drifts and the table says so rather than forcing it). The genre_prior row is the strong baseline you reach for when genre is *known*; the audio-only rows are what you ship when it is not. The 14-point gap between them survives the protocol change.

![SHAP summary for the winning tree model](outputs/figures/07_shap_bar.png)

## Why this matters

Streaming services need a recommendation signal for tracks that have no behavior history yet. Behavioral signals (skips, replays, saves) take days to weeks to accumulate; audio features are available from the first upload. This study quantifies how far they go on their own and where they break.

## What it does

Three pipelines build on a single 1.16-million-track corpus combining maharshipandya, rodolfofigueroa, and paradisejoy Kaggle datasets. The modeling pipeline trains a five-model panel (genre prior, logistic, random forest, LightGBM, XGBoost) under both protocol arms with hyperparameters tuned once on grouped data and frozen to [`configs/hparams.json`](configs/hparams.json). The interpretability pipeline runs SHAP, permutation importance, logistic CIs, and isotonic recalibration under the same grouped protocol; Brier improves from 0.212 to 0.161 after calibration, a 24% relative reduction. The recommender pipeline builds a FAISS `IndexFlatIP` over L2-normalized audio features; its evaluation still runs under the v0.1 protocol and is being reworked ([#25](https://github.com/dhruvsood12/audio-priors/issues/25)).

## Reproduce

```bash
make install-dev      # editable install with dev, notebooks, app extras
make data             # Kaggle download + processed parquet (needs ~/.kaggle/kaggle.json)
make train            # train all five models, write outputs/tables/metrics.csv
make prepare-app      # bake model + FAISS artifacts to outputs/models/
make app              # launch the Streamlit demo on http://localhost:8501
```

No Kaggle account? `make data-demo` generates a 2K-row synthetic CSV in the same schema so the pipeline runs end-to-end without credentials.

Or with Docker:

```bash
docker compose up app    # builds the image, prepares artifacts, serves Streamlit
```

## Data

See [DATA.md](DATA.md) for the per-source attribution, license, fields used and dropped, and the Spotify Web API deprecation note. Three Kaggle datasets feed the corpus; raw CSVs are never committed.

## Methods

See [MODEL.md](MODEL.md) for the model card, including the intended use, top features across SHAP and permutation importance with the 2/3 overlap statement, calibration before and after isotonic, and the per-genre AUC.

## Per-genre AUC

LightGBM on the grouped-arm test set, top and bottom genres by audio predictability. Full table (both arms) in [`outputs/tables/per_genre_auc.csv`](outputs/tables/per_genre_auc.csv). Caution: several extremes rest on near-degenerate class balance (comedy has 4 positives; pop-film has 3 negatives); per-genre CIs with a multiple-comparison correction are tracked in [#23](https://github.com/dhruvsood12/audio-priors/issues/23).

| Genre | n | n positive | AUC |
|---|---|---|---|
| comedy | 224 | 4 | 0.94 |
| sleep | 124 | 50 | 0.93 |
| classical | 87 | 3 | 0.92 |
| ... | ... | ... | ... |
| samba | 125 | 1 | 0.27 |
| pop, R&B | 27 | 23 | 0.26 |
| pop-film | 127 | 124 | 0.18 |

## Recommender results

**Protocol note:** the retrieval tables below still run under the v0.1 protocol (full-corpus label cutoff, random query split, `(same_genre OR same_artist) AND sticky` relevance, no exclusion of the query artist's tracks from the candidates). The classifier results above show artist effects are large, so treat the audio rows here as upper bounds until the artist-disjoint rework lands ([#25](https://github.com/dhruvsood12/audio-priors/issues/25)).

7,645 evaluable queries from the 90/10 split. Bootstrap 1,000 resamples.

| Baseline | Recall@10 | NDCG@10 |
|---|---|---|
| audio_genre | 0.0186 | **0.2312** |
| genre_only | 0.0180 | 0.2249 |
| audio_only | 0.00147 | 0.0208 |
| popularity_only | 0.000394 | 0.00626 |
| random | 0.000127 | 0.00187 |

Under that legacy protocol, audio-only KNN reads 11.6x random on Recall@10, and audio + genre beats genre-only on a paired-bootstrap NDCG@10 difference of +0.00632 (95% CI 0.00203, 0.01057).

## Streamlit demo

![audio-priors demo](outputs/figures/demo.png)

Track-search mode picks a track and shows the sticky probability with per-feature SHAP contributions. Slider mode sets the ten audio features by hand and returns the top-10 nearest tracks by audio cosine. Boot is under 5 seconds with warm caches; each query returns in under one second.

## Limitations

- Popularity is a market signal, not a quality signal. It conflates artist reach, marketing spend, and timing.
- Audio features are Spotify-API heuristics computed before the November 2024 endpoint deprecation; the live API is not called.
- The 125-genre slice is uneven; per-genre AUCs at small n (under 50 test rows) are noisy.
- The cold-start ceiling at 0.71 AUC is what audio alone can do; behavioral signals (skips, replays) would dominate when available.

## Future work

- Add a second-stage learned projection that combines the genre prior with audio embeddings.
- Replace the 10-feature embedding with a deep audio model (MERT, CLAP) and see if the gap to the genre prior closes.
- Per-genre calibration: the global isotonic fit may understate or overstate confidence within specific buckets.
- A/B style offline eval against Last.fm play-count data once available.

## License and citation

MIT. See [LICENSE](LICENSE). The MIT license applies to code only; the three underlying Kaggle datasets retain their original licenses (ODbL-1.0, unknown, other; see [DATA.md](DATA.md)). The processed corpus is not redistributed by this project.

Cite as: Dhruv Sood, "audio-priors: cold-start audio-feature priors for music recommendation," v0.1.0, 2026.
