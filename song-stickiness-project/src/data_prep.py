"""Load, clean, validate, and derive targets for Spotify track data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Numeric audio features used in modeling (mirrors src.features.DEFAULT_AUDIO_FEATURES)
MODEL_NUMERIC_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "duration_ms",
]

EXPECTED_COLUMNS = [
    "track_name",
    "artist_name",
    "genre",
    "popularity",
    "danceability",
    "energy",
    "valence",
    "tempo",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "duration_ms",
]

# Alternate names (lowercase keys) -> canonical name
COLUMN_ALIASES: dict[str, str] = {
    "track": "track_name",
    "name": "track_name",
    "song": "track_name",
    "title": "track_name",
    "trackname": "track_name",
    "artist": "artist_name",
    "artists": "artist_name",
    "artistname": "artist_name",
    "genres": "genre",
    "track_genre": "genre",
    "duration": "duration_ms",
    "durationms": "duration_ms",
    "length_ms": "duration_ms",
}


def load_data(path: str | Path) -> pd.DataFrame:
    """Load CSV from path."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Data file not found: {p.resolve()}")
    return pd.read_csv(p)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and snake_case column names."""
    out = df.copy()
    out.columns = [str(c).strip().replace(" ", "_").lower() for c in out.columns]
    return out


def map_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names using aliases and fuzzy matching."""
    out = standardize_column_names(df)
    rename_map: dict[str, str] = {}
    cols = list(out.columns)
    lower_to_actual = {c.lower(): c for c in cols}

    for col in cols:
        low = col.lower()
        if low in COLUMN_ALIASES:
            target = COLUMN_ALIASES[low]
            if target not in rename_map.values() or col == lower_to_actual.get(low, col):
                rename_map[col] = target
        elif low in {e.lower() for e in EXPECTED_COLUMNS}:
            canon = next(e for e in EXPECTED_COLUMNS if e.lower() == low)
            rename_map[col] = canon

    out = out.rename(columns=rename_map)
    return out


def remove_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    keep: str = "first",
) -> tuple[pd.DataFrame, int]:
    """Drop duplicate rows; default subset uses track_name, artist_name, duration_ms if present."""
    if subset is None:
        subset = [c for c in ("track_name", "artist_name", "duration_ms") if c in df.columns]
        if not subset:
            subset = None
    before = len(df)
    out = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    return out, before - len(out)


def handle_missing_values(
    df: pd.DataFrame,
    required_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Drop rows with missing values in required columns.
    If required_cols is None, uses core modeling columns that exist in df.
    """
    if required_cols is None:
        core = ["popularity"] + MODEL_NUMERIC_FEATURES
        required_cols = [c for c in core if c in df.columns]

    missing_before = df[required_cols].isna().sum()
    mask = df[required_cols].notna().all(axis=1)
    out = df.loc[mask].reset_index(drop=True)
    return out, missing_before


def validate_ranges(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Clip or filter invalid ranges. Popularity 0-100; Spotify features typically 0-1;
    duration_ms must be positive.
    """
    out = df.copy()
    notes: dict[str, Any] = {}

    if "popularity" in out.columns:
        bad = (out["popularity"] < 0) | (out["popularity"] > 100)
        n_bad = int(bad.sum())
        if n_bad:
            out = out.loc[~bad].reset_index(drop=True)
            notes["popularity_out_of_range_dropped"] = n_bad

    spotify_01 = [
        "danceability",
        "energy",
        "valence",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
    ]
    for col in spotify_01:
        if col not in out.columns:
            continue
        bad = (out[col] < 0) | (out[col] > 1)
        n_bad = int(bad.sum())
        if n_bad:
            out.loc[bad, col] = np.nan
            notes[f"{col}_invalid_clipped_to_nan"] = n_bad

    if "duration_ms" in out.columns:
        bad = out["duration_ms"] <= 0
        n_bad = int(bad.sum())
        if n_bad:
            out = out.loc[~bad].reset_index(drop=True)
            notes["nonpositive_duration_dropped"] = n_bad

    return out, notes


def create_standardized_popularity(df: pd.DataFrame, col: str = "popularity") -> pd.DataFrame:
    """Add z-scored popularity column `popularity_z`."""
    out = df.copy()
    s = out[col].astype(float)
    mu = s.mean()
    sigma = s.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        out["popularity_z"] = 0.0
    else:
        out["popularity_z"] = (s - mu) / sigma
    return out


def create_sticky_label(
    df: pd.DataFrame,
    popularity_col: str = "popularity",
    percentile: float = 0.8,
    label_col: str = "sticky",
) -> tuple[pd.DataFrame, float]:
    """
    Binary label: 1 if popularity >= percentile threshold (default top 20%).
    Returns (dataframe, threshold_value).
    """
    out = df.copy()
    thresh = float(out[popularity_col].quantile(percentile))
    out[label_col] = (out[popularity_col] >= thresh).astype(int)
    return out, thresh


def save_processed_data(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
