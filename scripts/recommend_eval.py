"""Cold-start retrieval evaluation under three protocol configurations.

The audio baselines retrieve a query's nearest tracks by audio cosine,
and the nearest tracks are very often the query artist's own. The v0.1
relevance rule counted those as hits, so the audio lift over random was
partly an artist-matching score. This evaluator measures that:

- ``legacy``: the v0.1 protocol unchanged (full-corpus sticky label,
  stratified random 90/10 split, ``(same_genre OR same_artist)``
  relevance, no exclusion, raw and coefficient-weighted embeddings).
  The only configuration expected to reproduce the v0.1 table.
- ``random_excl``: a random split with the query artist's tracks
  removed from candidates and the relevance denominator, genre-only
  relevance. Holds the split fixed against ``legacy`` so the drop
  isolates artist fingerprinting.
- ``grouped_excl``: the honest cold-start figure. Artist-grouped split
  (no artist on both sides), genre-only relevance, query-artist
  exclusion, and an artist-cluster bootstrap over query artists.

Each row in the outputs carries ``split``, ``relevance``,
``artist_excluded`` and ``bootstrap_unit`` so no number is read out of
protocol.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from audio_priors import labels, recommend, splits

app = typer.Typer(add_completion=False)
ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLS = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "loudness",
    "tempo",
    "duration_ms",
]

METRIC_COLS = ("recall_at_10", "recall_at_50", "ndcg_at_10")


def _subsample(query_df: pd.DataFrame, n_queries: int) -> pd.DataFrame:
    if 0 < n_queries < len(query_df):
        return query_df.sample(n=n_queries, random_state=42)
    return query_df


def _legacy_config(df: pd.DataFrame, holdout: float, n_queries: int) -> pd.DataFrame:
    """The v0.1 path: full-corpus label, stratified random split, raw +
    weighted embeddings, seven baselines, no exclusion. Reproduces v0.1."""

    work = df.copy()
    work["sticky"] = labels.sticky_top_q(work, q=0.20)
    train_df, query_df = train_test_split(
        work, test_size=holdout, random_state=42, stratify=work["sticky"]
    )
    query_df = _subsample(query_df, n_queries)

    X_train = train_df[FEATURE_COLS].to_numpy()
    X_query = query_df[FEATURE_COLS].to_numpy()
    xt_raw, xq_raw = recommend.embed_features(X_train, X_query)

    scaler = StandardScaler().fit(X_train)
    lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    lr.fit(scaler.transform(X_train), train_df["sticky"])
    weights = np.abs(lr.coef_.ravel())
    xt_w, xq_w = recommend.embed_features(X_train, X_query, weights=weights)

    corpus_raw = recommend.build_corpus(
        train_df[FEATURE_COLS],
        train_df["genre"],
        train_df["artist_name"],
        train_df["sticky"],
        train_df["popularity"],
        X_train_embedded=xt_raw,
    )
    corpus_w = recommend.build_corpus(
        train_df[FEATURE_COLS],
        train_df["genre"],
        train_df["artist_name"],
        train_df["sticky"],
        train_df["popularity"],
        X_train_embedded=xt_w,
    )
    pq_raw = recommend.evaluate_baselines(
        corpus_raw, xq_raw, query_df["genre"], query_df["artist_name"]
    )
    pq_w = recommend.evaluate_baselines(
        corpus_w,
        xq_w,
        query_df["genre"],
        query_df["artist_name"],
        baselines={
            "audio_only_weighted": recommend.audio_baseline,
            "audio_genre_weighted": recommend.audio_genre_baseline,
        },
    )
    pq = pd.concat([pq_raw, pq_w], ignore_index=True)
    pq["split"] = "random_stratified"
    pq["relevance"] = "genre_or_artist"
    pq["artist_excluded"] = False
    return pq


def _v2_config(df: pd.DataFrame, kind: str, holdout: float, n_queries: int) -> pd.DataFrame:
    """A v2 path: train-only label, protocol split, genre-only relevance,
    query-artist exclusion. ``kind`` is ``random`` or ``grouped``."""

    work = df.reset_index(drop=True)
    sp = splits.protocol_split(work, kind, test_size=holdout, val_size=0.20)
    y_all, _ = labels.sticky_top_q_train_threshold(work, work.index[sp.train], q=0.20)
    train_df = work.iloc[sp.train].copy()
    train_df["sticky"] = y_all.iloc[sp.train].to_numpy()
    query_df = _subsample(work.iloc[sp.test].copy(), n_queries)

    xt, xq = recommend.embed_features(
        train_df[FEATURE_COLS].to_numpy(), query_df[FEATURE_COLS].to_numpy()
    )
    corpus = recommend.build_corpus(
        train_df[FEATURE_COLS],
        train_df["genre"],
        train_df["artist_name"],
        train_df["sticky"],
        train_df["popularity"],
        X_train_embedded=xt,
    )
    pq = recommend.evaluate_baselines(
        corpus,
        xq,
        query_df["genre"],
        query_df["artist_name"],
        relevance_mode="genre_only",
        exclude_query_artist=True,
    )
    pq["split"] = kind
    pq["relevance"] = "genre_only"
    pq["artist_excluded"] = True
    return pq


@app.command()
def main(
    parquet_path: Path = typer.Option(ROOT / "data" / "processed" / "tracks.parquet"),
    holdout: float = typer.Option(0.10, help="Query hold-out fraction."),
    n_queries: int = typer.Option(-1, help="Queries to sample (-1 uses all evaluable queries)."),
    n_resamples: int = typer.Option(1000, help="Bootstrap resamples for CI."),
    out_path: Path = typer.Option(ROOT / "outputs" / "tables" / "recommender_metrics.csv"),
    per_query_path: Path = typer.Option(ROOT / "outputs" / "tables" / "recommender_per_query.csv"),
) -> None:
    """Run the three retrieval configurations and write tagged metric tables."""

    typer.echo(f"loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=["popularity"]).copy()
    df = df.dropna(subset=FEATURE_COLS).copy()
    typer.echo(f"labeled rows: {len(df):,}")

    typer.echo("== config: legacy (v0.1 protocol) ==")
    pq_legacy = _legacy_config(df, holdout, n_queries)
    typer.echo("== config: random split, artist-excluded, genre-only ==")
    pq_random = _v2_config(df, "random", holdout, n_queries)
    typer.echo("== config: grouped split, artist-excluded, genre-only (headline) ==")
    pq_grouped = _v2_config(df, "grouped", holdout, n_queries)

    per_query = pd.concat([pq_legacy, pq_random, pq_grouped], ignore_index=True)
    per_query_path.parent.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(per_query_path, index=False)
    typer.echo(f"wrote {per_query_path}")

    summaries: list[pd.DataFrame] = []
    for (split, relevance, excluded), grp in per_query.groupby(
        ["split", "relevance", "artist_excluded"], sort=False
    ):
        # Cluster the bootstrap by query artist only for the grouped arm,
        # where one artist contributes several correlated queries.
        group_col = "query_artist" if split == "grouped" else None
        s = recommend.summarize_with_ci(
            grp, METRIC_COLS, n_resamples=n_resamples, group_col=group_col
        )
        s.insert(0, "split", split)
        s.insert(1, "relevance", relevance)
        s.insert(2, "artist_excluded", excluded)
        underflow_by_baseline = grp.groupby("baseline")["underflow"].sum()
        s["underflow_rows"] = s["baseline"].map(underflow_by_baseline).fillna(0).astype(int)
        summaries.append(s)
    summary = pd.concat(summaries, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    typer.echo(f"wrote {out_path}")
    typer.echo("")
    cols = ["split", "relevance", "baseline", "recall_at_10", "ndcg_at_10", "bootstrap_unit"]
    typer.echo(summary[cols].to_string(index=False))


if __name__ == "__main__":
    app()
