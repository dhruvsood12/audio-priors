"""Schema-level behavior of the data-prep pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from src import data_prep


def test_clean_dataframe_succeeds_with_all_expected_columns(
    synthetic_tracks: pd.DataFrame,
) -> None:
    out, stats = data_prep.clean_dataframe(synthetic_tracks)
    assert len(out) > 0
    assert "sticky" in out.columns
    assert "popularity_z" in out.columns
    assert stats["n_rows_final"] == len(out)


def test_clean_dataframe_raises_when_popularity_missing(
    synthetic_tracks: pd.DataFrame,
) -> None:
    """popularity is the load-bearing column; its absence should surface, not silently."""
    df = synthetic_tracks.drop(columns=["popularity"])
    with pytest.raises(KeyError):
        data_prep.clean_dataframe(df)
