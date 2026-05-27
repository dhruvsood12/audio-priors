"""Feature column constants and audio-feature selection helpers.

The brief Section 5 layout places a ``features.py`` next to ``data.py``;
this module owns the canonical 10-feature audio list plus a couple of
thin selectors so downstream scripts and notebooks do not redefine the
list locally.
"""

from __future__ import annotations

import pandas as pd

AUDIO_FEATURES: tuple[str, ...] = (
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


def select_audio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df[list(AUDIO_FEATURES)]`` after checking presence."""

    missing = [c for c in AUDIO_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"audio features missing from frame: {missing}")
    return df[list(AUDIO_FEATURES)].copy()


def feature_columns_present(df: pd.DataFrame) -> list[str]:
    """Subset of ``AUDIO_FEATURES`` that exists in ``df``."""

    return [c for c in AUDIO_FEATURES if c in df.columns]


def has_full_feature_set(df: pd.DataFrame) -> bool:
    """True iff every audio feature column is present."""

    return all(c in df.columns for c in AUDIO_FEATURES)
