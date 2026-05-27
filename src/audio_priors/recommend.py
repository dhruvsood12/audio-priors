"""Cold-start recommender via FAISS over L2-normalized audio features.

Five-baseline retrieval evaluation. Each baseline is a callable that
returns the top-K training indices for a query row:

- ``random_baseline``: sample K training indices uniformly.
- ``genre_baseline``: sample K training indices whose genre matches
  the query.
- ``popularity_baseline``: return the top-K training indices ranked
  by popularity, ignoring the query.
- ``audio_baseline``: FAISS top-K over L2-normalized audio features.
- ``audio_genre_baseline``: FAISS top-K within the query's genre.

Relevance for a query ``Q`` and a training row ``R`` is

    same_genre(R, Q) OR same_artist(R, Q)) AND sticky(R)

The "(same genre or same artist)" component anchors retrieval to the
musical context of the query, and the sticky filter restricts hits to
top-quintile-popularity targets. Random retrieval at K=10 hits this
relevance set at a rate near ``K / N``; the brief asks the audio
baseline to clear ``5x`` random Recall@10.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization. Zero-norm rows return zero rows."""

    return normalize(X.astype(np.float32), norm="l2", axis=1)


def build_faiss_index(X: np.ndarray) -> faiss.Index:
    """L2-normalize ``X`` and build an ``IndexFlatIP`` (cosine similarity)."""

    Xn = l2_normalize(X)
    index = faiss.IndexFlatIP(Xn.shape[1])
    index.add(Xn)
    return index


def embed_features(
    X_train: np.ndarray,
    X_query: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize on train statistics, optionally weight, then L2-normalize rows.

    Returns ``(X_train_embedded, X_query_embedded)`` aligned with the input
    row order. ``weights`` is a per-feature vector; if provided, every
    column is scaled by its weight after standardization. The brief's
    "learned projection from logistic coefficients" passes the per-feature
    absolute coefficients here so the audio space stretches along the
    directions the model finds informative.
    """

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X_train)
    Xq = scaler.transform(X_query)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float32).reshape(1, -1)
        Xt = Xt * w
        Xq = Xq * w
    return l2_normalize(Xt), l2_normalize(Xq)


@dataclass
class TrainingCorpus:
    """Pre-built lookup tables used by every baseline.

    ``X`` is the feature matrix (already L2-normalized). ``genre`` and
    ``artist`` are aligned arrays. ``popularity_rank`` is descending so
    ``popularity_rank[:k]`` gives the K most popular tracks.
    """

    X: np.ndarray
    genre: np.ndarray
    artist: np.ndarray
    sticky: np.ndarray
    popularity_rank: np.ndarray
    index: faiss.Index
    genre_to_indices: dict[str, np.ndarray]
    genre_to_index: dict[str, faiss.Index]


def build_corpus(
    X_train: pd.DataFrame,
    genres_train: pd.Series,
    artists_train: pd.Series,
    sticky_train: pd.Series,
    popularity_train: pd.Series,
    X_train_embedded: np.ndarray | None = None,
    rng_seed: int = 42,
) -> TrainingCorpus:
    """Pre-compute FAISS indices and per-genre lookups for fast eval.

    If ``X_train_embedded`` is passed, it must already be the
    standardized + L2-normalized embedding for the training set; the
    function uses it directly. Otherwise a plain L2 normalize is
    applied (kept for the unit tests on toy data).
    """

    if X_train_embedded is not None:
        X = X_train_embedded.astype(np.float32)
    else:
        X = l2_normalize(X_train.to_numpy())
    genre = genres_train.fillna("__unknown__").astype(str).to_numpy()
    artist = artists_train.fillna("__unknown__").astype(str).to_numpy()
    sticky = sticky_train.astype(bool).to_numpy()
    popularity = popularity_train.astype(float).to_numpy()
    pop_rank = np.argsort(-popularity, kind="stable")  # descending

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    # Per-genre indices for the audio+genre baseline.
    genre_to_indices: dict[str, np.ndarray] = {}
    genre_to_index: dict[str, faiss.Index] = {}
    unique_genres = np.unique(genre)
    for g in unique_genres:
        idx = np.where(genre == g)[0]
        if len(idx) == 0:
            continue
        genre_to_indices[g] = idx
        sub_X = X[idx]
        sub_index = faiss.IndexFlatIP(sub_X.shape[1])
        sub_index.add(sub_X)
        genre_to_index[g] = sub_index

    return TrainingCorpus(
        X=X,
        genre=genre,
        artist=artist,
        sticky=sticky,
        popularity_rank=pop_rank,
        index=index,
        genre_to_indices=genre_to_indices,
        genre_to_index=genre_to_index,
    )


def compute_relevance(
    query_genre: str,
    query_artist: str,
    corpus: TrainingCorpus,
) -> np.ndarray:
    """Boolean mask over the training corpus for one query."""

    return ((corpus.genre == query_genre) | (corpus.artist == query_artist)) & corpus.sticky


def recall_at_k(retrieved_indices: np.ndarray, relevance_mask: np.ndarray, k: int) -> float:
    """Standard recall at K. NaN when there are no relevant items."""

    n_total = int(relevance_mask.sum())
    if n_total == 0:
        return float("nan")
    top_k = retrieved_indices[:k]
    n_hit = int(relevance_mask[top_k].sum())
    return n_hit / n_total


def ndcg_at_k(retrieved_indices: np.ndarray, relevance_mask: np.ndarray, k: int) -> float:
    """Binary NDCG at K. NaN when there are no relevant items."""

    n_total = int(relevance_mask.sum())
    if n_total == 0:
        return float("nan")
    top_k = retrieved_indices[:k]
    gains = relevance_mask[top_k].astype(float)
    discounts = np.log2(np.arange(2, k + 2))
    dcg = float((gains / discounts).sum())
    ideal_k = min(k, n_total)
    idcg = float((1.0 / discounts[:ideal_k]).sum())
    if idcg == 0:
        return 0.0
    return dcg / idcg


def random_baseline(
    query_vec: np.ndarray,
    query_genre: str,
    query_artist: str,
    corpus: TrainingCorpus,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample K indices uniformly from the training corpus."""

    n = len(corpus.genre)
    return rng.choice(n, size=min(k, n), replace=False)


def genre_baseline(
    query_vec: np.ndarray,
    query_genre: str,
    query_artist: str,
    corpus: TrainingCorpus,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample K indices from the same-genre subset; fall back to random fill."""

    pool = corpus.genre_to_indices.get(query_genre)
    if pool is None or len(pool) == 0:
        return random_baseline(query_vec, query_genre, query_artist, corpus, k, rng)
    if len(pool) >= k:
        return rng.choice(pool, size=k, replace=False)
    # Pool too small; take everyone and fill the rest from outside.
    inside = pool.copy()
    rest = np.setdiff1d(np.arange(len(corpus.genre)), inside, assume_unique=False)
    fill = rng.choice(rest, size=k - len(inside), replace=False)
    return np.concatenate([inside, fill])


def popularity_baseline(
    query_vec: np.ndarray,
    query_genre: str,
    query_artist: str,
    corpus: TrainingCorpus,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return the K most popular training tracks. Query is ignored."""

    return corpus.popularity_rank[:k]


def audio_baseline(
    query_vec: np.ndarray,
    query_genre: str,
    query_artist: str,
    corpus: TrainingCorpus,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """FAISS top-K. Expects ``query_vec`` already at the same embedding scale
    as the corpus (standardized + L2-normalized)."""

    q = query_vec.reshape(1, -1).astype(np.float32)
    _, idx = corpus.index.search(q, k)
    return idx[0]


def audio_genre_baseline(
    query_vec: np.ndarray,
    query_genre: str,
    query_artist: str,
    corpus: TrainingCorpus,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """FAISS top-K within the query's genre. Fills from global FAISS if short."""

    sub_index = corpus.genre_to_index.get(query_genre)
    sub_idx = corpus.genre_to_indices.get(query_genre)
    if sub_index is None or sub_idx is None or len(sub_idx) == 0:
        return audio_baseline(query_vec, query_genre, query_artist, corpus, k, rng)
    q = query_vec.reshape(1, -1).astype(np.float32)
    k_sub = min(k, len(sub_idx))
    _, sub_top = sub_index.search(q, k_sub)
    out = sub_idx[sub_top[0]]
    if len(out) < k:
        # Pad from global retrieval, skipping ones already in out.
        _, global_top = corpus.index.search(q, k * 5)
        seen = set(out.tolist())
        for j in global_top[0]:
            if int(j) in seen:
                continue
            out = np.concatenate([out, [int(j)]])
            if len(out) >= k:
                break
    return out


BASELINES: dict[str, Callable[..., np.ndarray]] = {
    "random": random_baseline,
    "genre_only": genre_baseline,
    "popularity_only": popularity_baseline,
    "audio_only": audio_baseline,
    "audio_genre": audio_genre_baseline,
}


def evaluate_baselines(
    corpus: TrainingCorpus,
    X_query: np.ndarray,
    query_genres: pd.Series,
    query_artists: pd.Series,
    k_list: tuple[int, ...] = (10, 50),
    ndcg_k: int = 10,
    baselines: dict[str, Callable[..., np.ndarray]] | None = None,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Per-query metrics for every baseline. One row per (baseline, query)."""

    if baselines is None:
        baselines = BASELINES
    X_query_n = l2_normalize(X_query)
    qg = query_genres.fillna("__unknown__").astype(str).to_numpy()
    qa = query_artists.fillna("__unknown__").astype(str).to_numpy()
    rng = np.random.default_rng(rng_seed)

    rows: list[dict[str, float | str]] = []
    max_k = max(*k_list, ndcg_k)
    for i in range(len(X_query_n)):
        rel = compute_relevance(qg[i], qa[i], corpus)
        if rel.sum() == 0:
            continue
        for name, fn in baselines.items():
            top = fn(X_query_n[i], qg[i], qa[i], corpus, max_k, rng)
            row: dict[str, Any] = {"baseline": name, "query_idx": int(i)}
            for k in k_list:
                row[f"recall_at_{k}"] = recall_at_k(top, rel, k)
            row[f"ndcg_at_{ndcg_k}"] = ndcg_at_k(top, rel, ndcg_k)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_with_ci(
    per_query: pd.DataFrame,
    metric_cols: tuple[str, ...],
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-baseline point estimates with bootstrap CI over queries."""

    rng = np.random.default_rng(seed)
    out_rows: list[dict[str, Any]] = []
    for name, group in per_query.groupby("baseline"):
        row: dict[str, Any] = {"baseline": name, "n_queries": len(group)}
        for metric in metric_cols:
            vals = group[metric].dropna().to_numpy()
            if len(vals) == 0:
                row[metric] = float("nan")
                row[f"{metric}_ci_lower"] = float("nan")
                row[f"{metric}_ci_upper"] = float("nan")
                continue
            point = float(vals.mean())
            n = len(vals)
            samples = np.empty(n_resamples, dtype=float)
            for i in range(n_resamples):
                idx = rng.integers(0, n, size=n)
                samples[i] = vals[idx].mean()
            row[metric] = point
            row[f"{metric}_ci_lower"] = float(np.percentile(samples, 100.0 * alpha / 2.0))
            row[f"{metric}_ci_upper"] = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
        out_rows.append(row)
    return pd.DataFrame(out_rows)
