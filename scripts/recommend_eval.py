"""Phase 5 evaluator for cold-start retrieval.

Builds a 90/10 split of the labeled corpus (per brief). Trains nothing;
just sets up a FAISS corpus over the 90% train side and runs every
baseline on every query in the 10% hold-out. Writes per-baseline
Recall@10, Recall@50, NDCG@10 with bootstrap CIs to
``outputs/tables/recommender_metrics.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from audio_priors import labels, recommend

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


@app.command()
def main(
    parquet_path: Path = typer.Option(ROOT / "data" / "processed" / "tracks.parquet"),
    q: float = typer.Option(0.20, help="Top-q fraction for the sticky relevance filter."),
    holdout: float = typer.Option(0.10, help="Hold-out fraction for queries (brief: 10%)."),
    n_queries: int = typer.Option(
        2000,
        help="Sample size from the hold-out (set to -1 to use all). Random subsample for compute.",
    ),
    n_resamples: int = typer.Option(1000, help="Bootstrap resamples for CI over queries."),
    out_path: Path = typer.Option(ROOT / "outputs" / "tables" / "recommender_metrics.csv"),
    per_query_path: Path = typer.Option(
        ROOT / "outputs" / "tables" / "recommender_per_query.csv",
        help="Optional per-query metric dump (helpful for inspection).",
    ),
) -> None:
    """Run all five baselines on a 10% query hold-out and write metrics CSV."""

    typer.echo(f"loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=["popularity"]).copy()
    df["sticky"] = labels.sticky_top_q(df, q=q)
    df = df.dropna(subset=FEATURE_COLS).copy()
    typer.echo(f"labeled rows: {len(df):,} | positive rate: {df['sticky'].mean():.4f}")

    train_df, query_df = train_test_split(
        df, test_size=holdout, random_state=42, stratify=df["sticky"]
    )
    typer.echo(f"train: {len(train_df):,} | query hold-out: {len(query_df):,}")

    if n_queries > 0 and n_queries < len(query_df):
        query_df = query_df.sample(n=n_queries, random_state=42)
        typer.echo(f"sampled {n_queries:,} queries for evaluation")

    X_train = train_df[FEATURE_COLS].to_numpy()
    X_query = query_df[FEATURE_COLS].to_numpy()

    typer.echo("standardized + L2-normalized embedding")
    Xt_raw, Xq_raw = recommend.embed_features(X_train, X_query)

    typer.echo("coefficient-weighted embedding (logistic on standardized features)")
    scaler = StandardScaler().fit(X_train)
    X_train_std = scaler.transform(X_train)
    lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    lr.fit(X_train_std, train_df["sticky"])
    weights = np.abs(lr.coef_.ravel())
    typer.echo(f"  weights: {dict(zip(FEATURE_COLS, weights.round(3).tolist(), strict=True))}")
    Xt_w, Xq_w = recommend.embed_features(X_train, X_query, weights=weights)

    typer.echo("building FAISS corpus (raw audio embedding)")
    corpus_raw = recommend.build_corpus(
        train_df[FEATURE_COLS],
        train_df["genre"],
        train_df["artist_name"],
        train_df["sticky"],
        train_df["popularity"],
        X_train_embedded=Xt_raw,
    )
    typer.echo("building FAISS corpus (coefficient-weighted embedding)")
    corpus_w = recommend.build_corpus(
        train_df[FEATURE_COLS],
        train_df["genre"],
        train_df["artist_name"],
        train_df["sticky"],
        train_df["popularity"],
        X_train_embedded=Xt_w,
    )

    typer.echo("evaluating baselines on raw audio embedding")
    per_query_raw = recommend.evaluate_baselines(
        corpus_raw,
        Xq_raw,
        query_df["genre"],
        query_df["artist_name"],
        k_list=(10, 50),
        ndcg_k=10,
    )

    typer.echo("evaluating audio variants on coefficient-weighted embedding")
    per_query_w = recommend.evaluate_baselines(
        corpus_w,
        Xq_w,
        query_df["genre"],
        query_df["artist_name"],
        k_list=(10, 50),
        ndcg_k=10,
        baselines={
            "audio_only_weighted": recommend.audio_baseline,
            "audio_genre_weighted": recommend.audio_genre_baseline,
        },
    )

    per_query = pd.concat([per_query_raw, per_query_w], ignore_index=True)
    typer.echo(
        f"evaluable queries (>=1 relevant): {len(per_query_raw) // 5:,}  "
        f"(weighted variants reuse the same queries)"
    )

    per_query_path.parent.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(per_query_path, index=False)
    typer.echo(f"wrote {per_query_path}")

    summary = recommend.summarize_with_ci(
        per_query,
        metric_cols=("recall_at_10", "recall_at_50", "ndcg_at_10"),
        n_resamples=n_resamples,
    )
    summary = summary.sort_values("recall_at_10", ascending=False).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    typer.echo(f"wrote {out_path}")
    typer.echo("")
    typer.echo(summary.to_string(index=False))


if __name__ == "__main__":
    app()
