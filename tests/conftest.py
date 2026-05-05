"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_tracks() -> pd.DataFrame:
    """Spotify-like dataframe with all expected columns and 200 rows; deterministic."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "track_name": [f"track_{i}" for i in range(n)],
            "artist_name": [f"artist_{i % 30}" for i in range(n)],
            "genre": rng.choice(["pop", "rock", "jazz", "hip hop"], size=n),
            "popularity": rng.integers(0, 101, size=n),
            "chart_weeks": rng.integers(1, 50, size=n),
            "danceability": rng.uniform(0, 1, size=n),
            "energy": rng.uniform(0, 1, size=n),
            "valence": rng.uniform(0, 1, size=n),
            "tempo": rng.uniform(60, 200, size=n),
            "loudness": rng.uniform(-30, 0, size=n),
            "speechiness": rng.uniform(0, 1, size=n),
            "acousticness": rng.uniform(0, 1, size=n),
            "instrumentalness": rng.uniform(0, 1, size=n),
            "liveness": rng.uniform(0, 1, size=n),
            "duration_ms": rng.integers(120000, 400000, size=n),
        }
    )
