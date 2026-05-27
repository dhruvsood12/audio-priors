"""Tests for the Phase 5 cold-start recommender."""

from __future__ import annotations

import numpy as np
import pandas as pd

from audio_priors.recommend import (
    build_corpus,
    compute_relevance,
    l2_normalize,
    ndcg_at_k,
    recall_at_k,
)


def _toy_corpus(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    feats = pd.DataFrame(rng.normal(0, 1, size=(n, 4)), columns=["a", "b", "c", "d"])
    genres = pd.Series(rng.choice(["pop", "rock", "jazz"], size=n))
    artists = pd.Series([f"artist_{i % 12}" for i in range(n)])
    sticky = pd.Series(rng.integers(0, 2, size=n))
    popularity = pd.Series(rng.integers(0, 101, size=n))
    return feats, genres, artists, sticky, popularity


def test_l2_normalize_returns_unit_norm_rows() -> None:
    X = np.array([[3.0, 4.0], [1.0, 1.0], [0.0, 5.0]])
    Xn = l2_normalize(X)
    norms = np.linalg.norm(Xn, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_ndcg_at_k_is_one_for_perfect_ranking() -> None:
    """Brief acceptance: a perfect ranking has NDCG = 1.0."""
    # All 5 relevant items appear in the top 5 of the retrieved list.
    retrieved = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    relevance = np.array([True, True, True, True, True, False, False, False, False, False])
    assert ndcg_at_k(retrieved, relevance, k=10) == 1.0
    assert ndcg_at_k(retrieved, relevance, k=5) == 1.0


def test_recall_at_k_counts_hits_correctly() -> None:
    retrieved = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    relevance = np.array([True] * 4 + [False] * 6)
    # Top 5 retrieves 4/4 relevant.
    assert recall_at_k(retrieved, relevance, k=5) == 1.0
    # Top 3 retrieves 3/4 relevant.
    assert recall_at_k(retrieved, relevance, k=3) == 3 / 4


def test_compute_relevance_or_of_genre_and_artist_and_sticky() -> None:
    feats, genres, artists, _, popularity = _toy_corpus(n=12, seed=1)
    # Hand-craft sticky so the OR-and-sticky logic is testable.
    sticky = pd.Series([1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0])
    corpus = build_corpus(feats, genres, artists, sticky, popularity)
    # Query: same genre as row 0, same artist as row 6 (artist_6 % 12 = 6).
    g0 = corpus.genre[0]
    a6 = corpus.artist[6]
    rel = compute_relevance(g0, a6, corpus)
    # Every relevant row must be sticky and share genre or artist with the query.
    for i in np.where(rel)[0]:
        assert corpus.sticky[i]
        assert corpus.genre[i] == g0 or corpus.artist[i] == a6
    # Non-relevant rows are either non-sticky or in a different genre / artist.
    for i in np.where(~rel)[0]:
        assert not corpus.sticky[i] or (corpus.genre[i] != g0 and corpus.artist[i] != a6)


def test_faiss_audio_baseline_returns_topk_within_corpus() -> None:
    from audio_priors.recommend import audio_baseline

    feats, genres, artists, sticky, popularity = _toy_corpus(n=40, seed=2)
    corpus = build_corpus(feats, genres, artists, sticky, popularity)
    q = feats.iloc[0].to_numpy()
    rng = np.random.default_rng(0)
    top = audio_baseline(q, corpus.genre[0], corpus.artist[0], corpus, k=5, rng=rng)
    assert len(top) == 5
    # Indices must be valid.
    assert top.min() >= 0 and top.max() < len(corpus.genre)
    # The closest neighbor of a corpus point queried against itself is itself.
    assert top[0] == 0


def test_ndcg_returns_nan_when_no_relevant_items() -> None:
    retrieved = np.array([0, 1, 2])
    relevance = np.array([False, False, False])
    assert np.isnan(ndcg_at_k(retrieved, relevance, k=3))
    assert np.isnan(recall_at_k(retrieved, relevance, k=3))
