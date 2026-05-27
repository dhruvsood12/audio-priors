"""Tests for the feature-selection helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from audio_priors.features import (
    AUDIO_FEATURES,
    feature_columns_present,
    has_full_feature_set,
    select_audio_features,
)


def test_audio_features_is_ten_columns() -> None:
    assert len(AUDIO_FEATURES) == 10
    # All ten Spotify-style features the brief uses for modeling.
    expected = {
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
    }
    assert set(AUDIO_FEATURES) == expected


def test_select_audio_features_returns_only_audio_columns() -> None:
    df = pd.DataFrame({col: [0.5] for col in AUDIO_FEATURES})
    df["popularity"] = 50
    df["genre"] = "pop"
    out = select_audio_features(df)
    assert list(out.columns) == list(AUDIO_FEATURES)
    assert "popularity" not in out.columns
    assert "genre" not in out.columns


def test_select_audio_features_raises_on_missing() -> None:
    df = pd.DataFrame({"danceability": [0.5], "energy": [0.5]})
    with pytest.raises(KeyError, match="audio features missing"):
        select_audio_features(df)


def test_feature_columns_present_returns_only_intersecting() -> None:
    df = pd.DataFrame({"danceability": [0.5], "tempo": [120.0], "popularity": [50]})
    present = feature_columns_present(df)
    assert set(present) == {"danceability", "tempo"}


def test_has_full_feature_set_truthy_when_complete() -> None:
    full = pd.DataFrame({col: [0.5] for col in AUDIO_FEATURES})
    partial = pd.DataFrame({"danceability": [0.5]})
    assert has_full_feature_set(full)
    assert not has_full_feature_set(partial)
