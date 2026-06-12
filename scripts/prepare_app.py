"""Pre-build the artifacts the Streamlit demo loads at boot.

Writes:

- ``outputs/models/lightgbm.pkl``: a fitted LightGBM classifier.
- ``outputs/models/scaler.pkl``: the ``StandardScaler`` fitted on the
  audio-feature columns of the labeled corpus.
- ``outputs/models/faiss.index``: the FAISS ``IndexFlatIP`` over
  standardized, L2-normalized features.
- ``outputs/models/embeddings.npy``: the L2-normalized feature matrix
  (kept alongside the index so the app can do per-genre subsetting).
- ``outputs/models/metadata.parquet``: ``track_name``, ``artist_name``,
  ``genre``, ``popularity``, and the ten audio features for every row
  in the labeled corpus.

The Streamlit app reads these via ``@st.cache_resource`` and never
needs to retrain on boot, so cold-start latency stays under the
brief's 5-second bar.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
import typer
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
    q: float = typer.Option(0.20, help="Top-q fraction for the sticky label."),
    models_dir: Path = typer.Option(ROOT / "outputs" / "models"),
) -> None:
    """Build and persist the app's model + FAISS artifacts."""

    typer.echo(f"loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=["popularity"]).copy()
    # Full-corpus threshold is correct HERE: this builds the deployed app
    # artifact, not an evaluation. Train/test scripts must use
    # labels.sticky_top_q_train_threshold instead.
    df["sticky"] = labels.sticky_top_q(df, q=q)
    df = df.dropna(subset=FEATURE_COLS).copy()
    typer.echo(f"labeled rows: {len(df):,} | positive rate: {df['sticky'].mean():.4f}")

    # Cast to plain float64 so LightGBM can handle the matrix; pandas nullable
    # Int64 (which duration_ms inherits) segfaults the LightGBM build on macOS.
    X = df[FEATURE_COLS].astype("float64").to_numpy()
    y = df["sticky"]

    typer.echo("fitting StandardScaler on audio features")
    scaler = StandardScaler().fit(X)

    typer.echo("training LightGBM")
    lgbm = lgb.LGBMClassifier(
        n_estimators=300,
        num_leaves=63,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        objective="binary",
        verbose=-1,
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
    )
    lgbm.fit(X, y)

    typer.echo("building FAISS index")
    embeddings = recommend.l2_normalize(scaler.transform(X))
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "lightgbm.pkl").write_bytes(pickle.dumps(lgbm))
    (models_dir / "scaler.pkl").write_bytes(pickle.dumps(scaler))
    faiss.write_index(index, str(models_dir / "faiss.index"))
    np.save(models_dir / "embeddings.npy", embeddings)

    meta_cols = ["track_name", "artist_name", "genre", "popularity", "sticky", *FEATURE_COLS]
    df[meta_cols].reset_index(drop=True).to_parquet(models_dir / "metadata.parquet", index=False)

    typer.echo("done")
    for p in (
        models_dir / "lightgbm.pkl",
        models_dir / "scaler.pkl",
        models_dir / "faiss.index",
        models_dir / "embeddings.npy",
        models_dir / "metadata.parquet",
    ):
        typer.echo(f"  {p.name}: {p.stat().st_size:,} bytes")


if __name__ == "__main__":
    app()
