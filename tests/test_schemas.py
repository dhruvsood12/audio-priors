"""Pandera schema acceptance and rejection tests."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
import pytest

from audio_priors.schemas import ProcessedTrack


def _valid_row() -> dict[str, object]:
    return {
        "source": "demo",
        "track_name": "Comedy",
        "artist_name": "Gen Hoshino",
        "genre": "pop",
        "popularity": 50.0,
        "danceability": 0.5,
        "energy": 0.5,
        "valence": 0.5,
        "acousticness": 0.5,
        "instrumentalness": 0.5,
        "liveness": 0.5,
        "speechiness": 0.5,
        "loudness": -5.0,
        "tempo": 120.0,
        "duration_ms": 200_000,
    }


def test_processed_track_accepts_valid_frame() -> None:
    df = pd.DataFrame([_valid_row()])
    out = ProcessedTrack.validate(df)
    assert len(out) == 1


def test_processed_track_rejects_popularity_above_100() -> None:
    row = _valid_row()
    row["popularity"] = 101.0
    df = pd.DataFrame([row])
    with pytest.raises(pa.errors.SchemaError):
        ProcessedTrack.validate(df)


def test_processed_track_rejects_danceability_above_1() -> None:
    row = _valid_row()
    row["danceability"] = 1.2
    df = pd.DataFrame([row])
    with pytest.raises(pa.errors.SchemaError):
        ProcessedTrack.validate(df)


def test_processed_track_rejects_tempo_at_lower_bound() -> None:
    """Brief uses an open interval ``(40, 250)``; tempo == 40 must be rejected."""
    row = _valid_row()
    row["tempo"] = 40.0
    df = pd.DataFrame([row])
    with pytest.raises(pa.errors.SchemaError):
        ProcessedTrack.validate(df)
