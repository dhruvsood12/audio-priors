"""Generate a 2,000-row synthetic CSV in the maharshipandya schema.

Lets the pipeline run end-to-end without Kaggle credentials. The CSV is
written to the same path the real maharshipandya download lands at, so
``scripts/download_data.py --skip-download`` can build a processed
parquet straight from the synthetic source.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUT = REPO_ROOT / "data" / "raw" / "spotify-tracks-dataset" / "dataset.csv"
GENRES = ("pop", "rock", "hip hop", "electronic", "r&b", "country", "jazz", "folk")


@app.command()
def main(
    out: Path = typer.Option(DEFAULT_OUT, help="Destination CSV path."),
    n: int = typer.Option(2000, help="Number of synthetic rows."),
    seed: int = typer.Option(42, help="RNG seed for reproducibility."),
) -> None:
    """Write a synthetic dataset matching the maharshipandya raw schema."""

    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "track_id": [f"id_{i}" for i in range(n)],
            "artists": [f"artist_{i % 200}" for i in range(n)],
            "album_name": [f"album_{i % 500}" for i in range(n)],
            "track_name": [f"track_{i}" for i in range(n)],
            "popularity": rng.integers(0, 101, size=n),
            "duration_ms": rng.integers(60_000, 480_000, size=n),
            "explicit": rng.choice([True, False], size=n),
            "danceability": np.clip(rng.beta(2, 2, size=n), 0, 1),
            "energy": np.clip(rng.beta(2, 2, size=n), 0, 1),
            "key": rng.integers(0, 12, size=n),
            "loudness": rng.uniform(-30, 0, size=n),
            "mode": rng.choice([0, 1], size=n),
            "speechiness": np.clip(rng.beta(1, 5, size=n), 0, 1),
            "acousticness": np.clip(rng.beta(2, 3, size=n), 0, 1),
            "instrumentalness": np.clip(rng.beta(1, 8, size=n), 0, 1),
            "liveness": np.clip(rng.beta(1, 4, size=n), 0, 1),
            "valence": np.clip(rng.beta(2, 2, size=n), 0, 1),
            "tempo": rng.uniform(60, 200, size=n),
            "time_signature": rng.choice([3, 4], size=n),
            "track_genre": rng.choice(GENRES, size=n),
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    typer.echo(f"wrote {out} ({n:,} rows)")


if __name__ == "__main__":
    app()
