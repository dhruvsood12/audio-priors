"""Long-stayer label derivation from chart_weeks."""

from __future__ import annotations

import pandas as pd
import pytest

from src import data_prep


def test_long_stayer_threshold_marks_top_quintile() -> None:
    df = pd.DataFrame({"chart_weeks": [1, 1, 2, 4, 5, 8, 10, 12, 17, 80]})
    out, thresh = data_prep.create_long_stayer_label(df, percentile=0.8)
    assert thresh > 0
    assert out["long_stayer"].sum() == (out["chart_weeks"] >= thresh).sum()
    assert 0 < out["long_stayer"].sum() < len(out)


def test_long_stayer_raises_on_missing_chart_weeks() -> None:
    df = pd.DataFrame({"popularity": [10, 20, 30]})
    with pytest.raises(KeyError, match="chart_weeks"):
        data_prep.create_long_stayer_label(df)


def test_clean_dataframe_produces_long_stayer_when_chart_weeks_present(
    synthetic_tracks: pd.DataFrame,
) -> None:
    out, stats = data_prep.clean_dataframe(synthetic_tracks)
    assert "long_stayer" in out.columns
    assert "long_stayer_threshold" in stats
    assert 0.0 < stats["long_stayer_balance"] < 1.0


def test_clean_dataframe_skips_long_stayer_when_chart_weeks_absent(
    synthetic_tracks: pd.DataFrame,
) -> None:
    df = synthetic_tracks.drop(columns=["chart_weeks"])
    out, stats = data_prep.clean_dataframe(df)
    assert "long_stayer" not in out.columns
    assert "long_stayer_threshold" not in stats
