"""More coverage for the recommender baselines and summary helper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from audio_priors.recommend import (
    BASELINES,
    audio_genre_baseline,
    build_corpus,
    evaluate_baselines,
    genre_baseline,
    l2_normalize,
    popularity_baseline,
    random_baseline,
    summarize_with_ci,
)


def _toy_corpus(n: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    feats = pd.DataFrame(rng.normal(0, 1, size=(n, 4)), columns=["a", "b", "c", "d"])
    genres = pd.Series(rng.choice(["pop", "rock", "jazz"], size=n))
    artists = pd.Series([f"artist_{i % 10}" for i in range(n)])
    sticky = pd.Series(rng.integers(0, 2, size=n))
    popularity = pd.Series(rng.integers(0, 101, size=n))
    return feats, genres, artists, sticky, popularity


def test_random_baseline_returns_k_distinct_indices() -> None:
    feats, genres, artists, sticky, popularity = _toy_corpus()
    corpus = build_corpus(feats, genres, artists, sticky, popularity)
    rng = np.random.default_rng(0)
    top = random_baseline(np.zeros(4), "pop", "artist_0", corpus, k=10, rng=rng)
    assert len(top) == 10
    assert len(set(top)) == 10


def test_popularity_baseline_returns_topk_popular() -> None:
    feats, genres, artists, sticky, popularity = _toy_corpus()
    corpus = build_corpus(feats, genres, artists, sticky, popularity)
    rng = np.random.default_rng(0)
    top = popularity_baseline(np.zeros(4), "pop", "artist_0", corpus, k=5, rng=rng)
    # The first index in popularity_rank should be the most popular.
    assert top[0] == int(np.argmax(popularity.to_numpy()))


def test_genre_baseline_keeps_results_in_same_genre_when_pool_large() -> None:
    feats, genres, artists, sticky, popularity = _toy_corpus(n=200)
    corpus = build_corpus(feats, genres, artists, sticky, popularity)
    rng = np.random.default_rng(0)
    target = "pop"
    top = genre_baseline(np.zeros(4), target, "artist_0", corpus, k=5, rng=rng)
    # With 200 rows the pop pool is large; every retrieved index should be pop.
    for j in top:
        assert corpus.genre[j] == target


def test_genre_baseline_falls_back_to_random_for_unknown_genre() -> None:
    feats, genres, artists, sticky, popularity = _toy_corpus()
    corpus = build_corpus(feats, genres, artists, sticky, popularity)
    rng = np.random.default_rng(0)
    top = genre_baseline(np.zeros(4), "nonexistent", "artist_0", corpus, k=5, rng=rng)
    assert len(top) == 5
    assert top.min() >= 0 and top.max() < len(corpus.genre)


def test_audio_genre_baseline_returns_topk_via_subindex() -> None:
    feats, genres, artists, sticky, popularity = _toy_corpus(n=120)
    corpus = build_corpus(feats, genres, artists, sticky, popularity)
    rng = np.random.default_rng(0)
    q = l2_normalize(np.array([feats.iloc[0].to_numpy()]))[0]
    top = audio_genre_baseline(q, corpus.genre[0], corpus.artist[0], corpus, k=5, rng=rng)
    assert len(top) == 5


def test_evaluate_baselines_then_summarize_with_ci_round_trip() -> None:
    feats, genres, artists, sticky, popularity = _toy_corpus(n=120)
    # Ensure at least some rows are sticky so relevance is non-empty for queries.
    sticky = pd.Series((feats["a"] > 0).astype(int))
    corpus = build_corpus(feats, genres, artists, sticky, popularity)

    # Use the first 20 rows as queries.
    Xq = feats.iloc[:20].to_numpy()
    qg = genres.iloc[:20]
    qa = artists.iloc[:20]

    per_query = evaluate_baselines(corpus, Xq, qg, qa, k_list=(5, 10), ndcg_k=5)
    assert {"baseline", "query_idx", "recall_at_5", "recall_at_10", "ndcg_at_5"}.issubset(
        per_query.columns
    )

    summary = summarize_with_ci(per_query, metric_cols=("recall_at_5", "ndcg_at_5"), n_resamples=50)
    assert "baseline" in summary.columns
    # Each baseline appears once in the summary.
    assert len(summary) == len(set(per_query["baseline"]))


def test_baselines_registry_lists_five() -> None:
    assert set(BASELINES.keys()) == {
        "random",
        "genre_only",
        "popularity_only",
        "audio_only",
        "audio_genre",
    }
