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
    ``max_artist_count`` is the largest number of tracks any single
    artist holds; fetching ``k + max_artist_count`` candidates and then
    dropping one artist's tracks always leaves at least ``k``.
    """

    X: np.ndarray
    genre: np.ndarray
    artist: np.ndarray
    sticky: np.ndarray
    popularity_rank: np.ndarray
    index: faiss.Index
    genre_to_indices: dict[str, np.ndarray]
    genre_to_index: dict[str, faiss.Index]
    artist_to_indices: dict[str, np.ndarray]
    max_artist_count: int


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

    # Per-artist row lookup, and the largest single-artist track count, used
    # by the artist-disjoint retrieval path to size the over-fetch.
    artist_to_indices: dict[str, np.ndarray] = {}
    uniq_artists, counts = np.unique(artist, return_counts=True)
    for a in uniq_artists:
        artist_to_indices[a] = np.where(artist == a)[0]
    max_artist_count = int(counts.max()) if len(counts) else 0

    return TrainingCorpus(
        X=X,
        genre=genre,
        artist=artist,
        sticky=sticky,
        popularity_rank=pop_rank,
        index=index,
        genre_to_indices=genre_to_indices,
        genre_to_index=genre_to_index,
        artist_to_indices=artist_to_indices,
        max_artist_count=max_artist_count,
    )


RELEVANCE_MODES = ("genre_or_artist", "genre_only")


def compute_relevance(
    query_genre: str,
    query_artist: str,
    corpus: TrainingCorpus,
    mode: str = "genre_or_artist",
    exclude_query_artist: bool = False,
) -> np.ndarray:
    """Boolean mask over the training corpus for one query.

    ``mode`` is ``"genre_or_artist"`` (the v0.1 relevance,
    ``(same_genre OR same_artist) AND sticky``) or ``"genre_only"``
    (``same_genre AND sticky``, which stops crediting audio for matching
    the query's own artist). With ``exclude_query_artist`` the query
    artist's tracks also leave the relevance denominator, keeping it
    coherent with an artist-disjoint candidate set; otherwise Recall@K
    would count relevant items the retriever is forbidden to return.
    """

    if mode == "genre_only":
        rel = (corpus.genre == query_genre) & corpus.sticky
    elif mode == "genre_or_artist":
        rel = ((corpus.genre == query_genre) | (corpus.artist == query_artist)) & corpus.sticky
    else:
        raise ValueError(f"mode must be one of {RELEVANCE_MODES}, got {mode!r}")
    if exclude_query_artist:
        rel = rel & (corpus.artist != query_artist)
    return rel


def drop_query_artist(
    retrieved: np.ndarray,
    query_artist: str,
    corpus: TrainingCorpus,
    k: int,
) -> np.ndarray:
    """Order-preserving removal of the query artist's tracks, then truncate.

    Keeps the retriever's ranking intact while guaranteeing no candidate
    shares the query's artist.
    """

    kept = retrieved[corpus.artist[retrieved] != query_artist]
    return kept[:k]


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
    # Discount by the actual retrieved length: an underflowing candidate list
    # (fewer than k items survived artist exclusion) is scored over the
    # positions it filled, while the ideal still spans min(k, n_total) slots,
    # so a short list is penalized rather than silently rescaled.
    discounts = np.log2(np.arange(2, len(top_k) + 2))
    dcg = float((gains / discounts).sum())
    ideal_k = min(k, n_total)
    idcg = float((1.0 / np.log2(np.arange(2, ideal_k + 2))).sum())
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
    k_eff = min(k, corpus.index.ntotal)
    _, idx = corpus.index.search(q, k_eff)
    out = idx[0]
    return out[out >= 0]  # strip FAISS -1 padding if k_eff somehow over-asks


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
        k_global = min(corpus.index.ntotal, k * 5)
        _, global_top = corpus.index.search(q, k_global)
        seen = set(out.tolist())
        for j in global_top[0]:
            if int(j) < 0 or int(j) in seen:
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
    relevance_mode: str = "genre_or_artist",
    exclude_query_artist: bool = False,
    fetch_buffer: int = 64,
) -> pd.DataFrame:
    """Per-query metrics for every baseline. One row per (baseline, query).

    With ``exclude_query_artist`` the query artist's tracks are dropped
    from every candidate list (order-preserving) and from the relevance
    denominator, so audio is not credited for matching the query's own
    artist. The over-fetch is ``max_k + max(fetch_buffer,
    max_artist_count)`` so dropping one artist cannot underflow on the
    full corpus; on a corpus too small for that (toy data, sparse
    genres) the result carries an ``underflow`` flag. With exclusion off
    the call path matches v0.1 exactly (same fetch size, same RNG draws)
    so the legacy table reproduces.
    """

    if baselines is None:
        baselines = BASELINES
    if relevance_mode not in RELEVANCE_MODES:
        raise ValueError(f"relevance_mode must be one of {RELEVANCE_MODES}, got {relevance_mode!r}")
    X_query_n = l2_normalize(X_query)
    qg = query_genres.fillna("__unknown__").astype(str).to_numpy()
    qa = query_artists.fillna("__unknown__").astype(str).to_numpy()
    rng = np.random.default_rng(rng_seed)
    n = len(corpus.genre)

    rows: list[dict[str, Any]] = []
    max_k = max(*k_list, ndcg_k)
    fetch_k = (
        min(n, max_k + max(fetch_buffer, corpus.max_artist_count))
        if exclude_query_artist
        else max_k
    )
    for i in range(len(X_query_n)):
        rel = compute_relevance(
            qg[i], qa[i], corpus, mode=relevance_mode, exclude_query_artist=exclude_query_artist
        )
        if rel.sum() == 0:
            continue
        for name, fn in baselines.items():
            cand = fn(X_query_n[i], qg[i], qa[i], corpus, fetch_k, rng)
            if exclude_query_artist:
                top = drop_query_artist(cand, qa[i], corpus, max_k)
            else:
                top = cand[:max_k]
            underflow = len(top) < max_k
            row: dict[str, Any] = {
                "baseline": name,
                "query_idx": int(i),
                "query_artist": qa[i],
                "underflow": bool(underflow),
            }
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
    group_col: str | None = None,
) -> pd.DataFrame:
    """Per-baseline point estimates with bootstrap CI over queries.

    With ``group_col`` the bootstrap resamples whole groups (e.g. query
    artists) rather than individual queries, the right unit when queries
    cluster: under an artist-grouped query split, one artist contributes
    several correlated queries, and a per-query bootstrap would report a
    CI that is too tight. ``bootstrap_unit`` is recorded per row.
    """

    rng = np.random.default_rng(seed)
    unit = "artist" if group_col else "query"
    out_rows: list[dict[str, Any]] = []
    for name, group in per_query.groupby("baseline"):
        row: dict[str, Any] = {
            "baseline": name,
            "n_queries": len(group),
            "bootstrap_unit": unit,
        }
        for metric in metric_cols:
            sub = group[[metric] + ([group_col] if group_col else [])].dropna(subset=[metric])
            vals = sub[metric].to_numpy()
            if len(vals) == 0:
                row[metric] = float("nan")
                row[f"{metric}_ci_lower"] = float("nan")
                row[f"{metric}_ci_upper"] = float("nan")
                continue
            point = float(vals.mean())
            samples = np.empty(n_resamples, dtype=float)
            if group_col:
                codes = sub[group_col].astype("category").cat.codes.to_numpy()
                order = np.argsort(codes, kind="stable")
                vals_sorted = vals[order]
                codes_sorted = codes[order]
                counts = np.bincount(codes_sorted)
                starts = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts)[:-1]])
                n_groups = len(counts)
                for i in range(n_resamples):
                    gidx = rng.integers(0, n_groups, size=n_groups)
                    picks = [vals_sorted[starts[g] : starts[g] + counts[g]] for g in gidx]
                    samples[i] = float(np.concatenate(picks).mean())
            else:
                n = len(vals)
                for i in range(n_resamples):
                    idx = rng.integers(0, n, size=n)
                    samples[i] = vals[idx].mean()
            row[metric] = point
            row[f"{metric}_ci_lower"] = float(np.percentile(samples, 100.0 * alpha / 2.0))
            row[f"{metric}_ci_upper"] = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
        out_rows.append(row)
    return pd.DataFrame(out_rows)
