"""Pandera schemas for raw source frames and the processed canonical frame."""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing.pandas import Series

# Audio-feature columns that share [0, 1] bounds across every source.
AUDIO_01_FEATURES = (
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
)

# Names that must exist in the processed frame in this order.
PROCESSED_COLUMNS = (
    "source",
    "track_name",
    "artist_name",
    "genre",
    "popularity",
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
)


class ProcessedTrack(pa.DataFrameModel):
    """The canonical schema written to data/processed/tracks.parquet.

    Range bounds match brief Section 6.1: audio features in [0, 1], popularity
    in [0, 100], tempo strictly inside (40, 250), and duration_ms strictly
    inside (30000, 600000). genre and popularity are nullable because one of
    the source datasets carries neither.
    """

    source: Series[str] = pa.Field(nullable=False)
    track_name: Series[str] = pa.Field(nullable=False)
    artist_name: Series[str] = pa.Field(nullable=False)
    genre: Series[str] = pa.Field(nullable=True)
    popularity: Series[float] = pa.Field(nullable=True, ge=0, le=100)
    danceability: Series[float] = pa.Field(nullable=True, ge=0, le=1)
    energy: Series[float] = pa.Field(nullable=True, ge=0, le=1)
    valence: Series[float] = pa.Field(nullable=True, ge=0, le=1)
    acousticness: Series[float] = pa.Field(nullable=True, ge=0, le=1)
    instrumentalness: Series[float] = pa.Field(nullable=True, ge=0, le=1)
    liveness: Series[float] = pa.Field(nullable=True, ge=0, le=1)
    speechiness: Series[float] = pa.Field(nullable=True, ge=0, le=1)
    loudness: Series[float] = pa.Field(nullable=True)
    tempo: Series[float] = pa.Field(nullable=True, gt=40, lt=250)
    duration_ms: Series[int] = pa.Field(nullable=True, gt=30000, lt=600000)

    class Config:
        strict = True
        coerce = True


class RawMaharshipandya(pa.DataFrameModel):
    """Minimal raw schema for maharshipandya/-spotify-tracks-dataset."""

    track_name: Series[str] = pa.Field(nullable=True)
    artists: Series[str] = pa.Field(nullable=True)
    track_genre: Series[str] = pa.Field(nullable=True)
    popularity: Series[int] = pa.Field(nullable=True)

    class Config:
        strict = False
        coerce = True


class RawRodolfofigueroa(pa.DataFrameModel):
    """Minimal raw schema for rodolfofigueroa/spotify-12m-songs."""

    name: Series[str] = pa.Field(nullable=True)
    artists: Series[str] = pa.Field(nullable=True)

    class Config:
        strict = False
        coerce = True


class RawParadisejoy(pa.DataFrameModel):
    """Minimal raw schema for paradisejoy/top-hits-spotify-from-20002019."""

    song: Series[str] = pa.Field(nullable=True)
    artist: Series[str] = pa.Field(nullable=True)
    genre: Series[str] = pa.Field(nullable=True)
    popularity: Series[int] = pa.Field(nullable=True)

    class Config:
        strict = False
        coerce = True
