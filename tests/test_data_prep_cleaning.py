"""Dedup, missingness, and range-validation behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import data_prep


def test_remove_duplicates_drops_exact_duplicates() -> None:
    df = pd.DataFrame(
        {
            "track_name": ["a", "a", "b"],
            "artist_name": ["x", "x", "y"],
            "duration_ms": [200000, 200000, 240000],
        }
    )
    out, n_dropped = data_prep.remove_duplicates(df)
    assert len(out) == 2
    assert n_dropped == 1


def test_handle_missing_values_drops_rows_with_required_nan() -> None:
    df = pd.DataFrame(
        {
            "popularity": [50.0, np.nan, 80.0],
            "danceability": [0.5, 0.6, 0.7],
        }
    )
    out, missing_before = data_prep.handle_missing_values(df)
    assert len(out) == 2
    assert int(missing_before["popularity"]) == 1


def test_validate_ranges_drops_out_of_bound_popularity() -> None:
    df = pd.DataFrame(
        {
            "popularity": [50, 150, -5, 80],
            "danceability": [0.5, 0.6, 0.7, 0.8],
            "duration_ms": [200000, 240000, 200000, 200000],
        }
    )
    out, notes = data_prep.validate_ranges(df)
    assert len(out) == 2
    assert notes["popularity_out_of_range_dropped"] == 2


def test_validate_ranges_clips_invalid_audio_features_to_nan() -> None:
    df = pd.DataFrame(
        {
            "popularity": [50, 60],
            "danceability": [0.5, 1.5],
            "duration_ms": [200000, 240000],
        }
    )
    out, notes = data_prep.validate_ranges(df)
    assert pd.isna(out.loc[1, "danceability"])
    assert notes["danceability_invalid_clipped_to_nan"] == 1
