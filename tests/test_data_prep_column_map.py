"""COLUMN_MAP renaming behavior in standardize_column_names + map_expected_columns."""

from __future__ import annotations

import pandas as pd

from src import data_prep


def test_artists_renamed_to_artist_name() -> None:
    df = pd.DataFrame({"artists": ["a", "b"], "popularity": [10, 20]})
    out = data_prep.map_expected_columns(df)
    assert "artist_name" in out.columns
    assert "artists" not in out.columns


def test_track_renamed_to_track_name() -> None:
    df = pd.DataFrame({"track": ["a"], "popularity": [10]})
    out = data_prep.map_expected_columns(df)
    assert "track_name" in out.columns


def test_duration_renamed_to_duration_ms() -> None:
    df = pd.DataFrame({"duration": [200000], "popularity": [50]})
    out = data_prep.map_expected_columns(df)
    assert "duration_ms" in out.columns


def test_uppercase_column_names_are_normalized() -> None:
    df = pd.DataFrame(
        {"Artist_Name": ["a"], "Popularity": [10], "Track_Name": ["t"]}
    )
    out = data_prep.map_expected_columns(df)
    assert {"artist_name", "popularity", "track_name"}.issubset(out.columns)
