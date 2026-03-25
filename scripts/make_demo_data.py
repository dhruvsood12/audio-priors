#!/usr/bin/env python3
"""Create a small demo CSV at data/raw/spotify_tracks.csv if the file is missing (for clone-and-run)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def main() -> None:
    out = ROOT / "data" / "raw" / "spotify_tracks.csv"
    if out.is_file():
        print("Already exists:", out)
        return
    np.random.seed(42)
    n = 2500
    genres = ["pop", "rock", "hip hop", "electronic", "r&b"]
    rows = []
    for i in range(n):
        pop = int(np.clip(np.random.beta(2, 5) * 100 + np.random.randn() * 5, 0, 100))
        rows.append(
            {
                "track_name": f"track_{i}",
                "artist_name": f"artist_{i % 80}",
                "genre": np.random.choice(genres),
                "popularity": pop,
                "danceability": float(np.clip(np.random.beta(2, 2), 0, 1)),
                "energy": float(np.clip(np.random.beta(2, 2), 0, 1)),
                "valence": float(np.clip(np.random.beta(2, 2), 0, 1)),
                "tempo": float(np.random.uniform(70, 180)),
                "loudness": float(np.random.uniform(-20, -3)),
                "speechiness": float(np.clip(np.random.beta(1, 5), 0, 1)),
                "acousticness": float(np.clip(np.random.beta(2, 3), 0, 1)),
                "instrumentalness": float(np.clip(np.random.beta(1, 8), 0, 1)),
                "liveness": float(np.clip(np.random.beta(1, 4), 0, 1)),
                "duration_ms": int(np.random.uniform(120, 400) * 1000),
            }
        )
    df = pd.DataFrame(rows)
    df["popularity"] = (
        (df["popularity"] + (df["danceability"] * 15 + df["energy"] * 10)).round().clip(0, 100).astype(int)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("Wrote", out, df.shape)


if __name__ == "__main__":
    main()
