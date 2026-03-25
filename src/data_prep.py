"""Load, clean, validate, and derive targets for Spotify track data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:

    def display(obj: Any) -> None:  # noqa: A001
        print(obj)


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

# Normalized column name (after standardize_column_names) -> canonical name.
# Extend as needed for new datasets.
COLUMN_MAP: dict[str, str] = {
    "track": "track_name",
    "name": "track_name",
    "song": "track_name",
    "title": "track_name",
    "trackname": "track_name",
    "song_name": "track_name",
    "track_name": "track_name",
    "artist": "artist_name",
    "artists": "artist_name",
    "artistname": "artist_name",
    "artist_name": "artist_name",
    "genres": "genre",
    "track_genre": "genre",
    "genre_name": "genre",
    "duration": "duration_ms",
    "durationms": "duration_ms",
    "length_ms": "duration_ms",
    "length": "duration_ms",
}


def load_data(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load CSV from path."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Data file not found: {p.resolve()}")
    return pd.read_csv(p, encoding="utf-8", low_memory=False, **kwargs)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and snake_case column names."""
    out = df.copy()
    out.columns = [str(c).strip().replace(" ", "_").lower() for c in out.columns]
    return out


def map_expected_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to canonical names:
    1) standardize column names
    2) apply COLUMN_MAP
    3) match remaining columns to EXPECTED_COLUMNS by case-insensitive equality
    """
    out = standardize_column_names(df)
    rename_map: dict[str, str] = {}
    cols = list(out.columns)
    expected_lower = {e.lower(): e for e in EXPECTED_COLUMNS}

    for col in cols:
        low = col.lower()
        if low in COLUMN_MAP:
            target = COLUMN_MAP[low]
            if col != target:
                rename_map[col] = target
        elif low in expected_lower:
            canon = expected_lower[low]
            if col != canon:
                rename_map[col] = canon

    out = out.rename(columns=rename_map)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


def fix_duration_units(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Convert duration_sec to duration_ms if present.
    If duration_ms looks like seconds (max in typical 60–600 s range), convert to ms.
    """
    out = df.copy()
    notes: dict[str, Any] = {}

    if "duration_sec" in out.columns:
        sec = pd.to_numeric(out["duration_sec"], errors="coerce")
        if "duration_ms" in out.columns:
            out = out.drop(columns=["duration_sec"])
            notes["duration_sec_dropped_duration_ms_present"] = True
        else:
            out["duration_ms"] = sec * 1000.0
            out = out.drop(columns=["duration_sec"])
            notes["duration_sec_converted_to_duration_ms"] = True

    if "duration_ms" not in out.columns:
        return out, notes

    s = pd.to_numeric(out["duration_ms"], errors="coerce")
    max_v = float(s.max()) if len(s) else 0.0
    if max_v > 0 and max_v < 25_000.0:
        out["duration_ms"] = s * 1000.0
        notes["duration_values_looked_like_seconds_scaled_to_ms"] = True
    return out, notes


def summarize_raw(df: pd.DataFrame) -> None:
    """Print column names, head, info, missing counts (for notebook inspection)."""
    print("Columns:", list(df.columns))
    print("Shape:", df.shape)
    display(pd.DataFrame({"missing": df.isna().sum(), "pct": (df.isna().mean() * 100).round(2)}))
    pd.set_option("display.max_columns", 30)
    print(df.head())
    print(df.info())


def summarize_raw_no_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return missingness summary for non-notebook use."""
    return pd.DataFrame({"missing": df.isna().sum(), "pct": (df.isna().mean() * 100).round(2)})


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
    Optional columns (genre, track_name, artist_name) are not required.
    """
    if required_cols is None:
        core = ["popularity"] + [c for c in MODEL_NUMERIC_FEATURES if c in df.columns]
        required_cols = list(dict.fromkeys(core))

    missing_before = df[required_cols].isna().sum()
    mask = df[required_cols].notna().all(axis=1)
    out = df.loc[mask].reset_index(drop=True)
    return out, missing_before


def validate_ranges(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Popularity 0–100; Spotify features 0–1; duration_ms > 0.
    Invalid 0–1 features set to NaN; bad popularity/duration rows dropped.
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


def clean_dataframe(
    df: pd.DataFrame,
    *,
    drop_after_validate: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Full pipeline: map columns → fix duration units → dedupe → drop NA on required
    → validate ranges → drop NA on numeric features again → popularity_z → sticky.

    Returns (cleaned_df, stats_dict).
    """
    stats: dict[str, Any] = {}
    out = map_expected_columns(df)
    out, dur_notes = fix_duration_units(out)
    stats.update(dur_notes)

    out, n_dup = remove_duplicates(out)
    stats["duplicates_removed"] = n_dup

    out, missing_report = handle_missing_values(out)
    stats["missingness_before_drop"] = missing_report.to_dict()

    out, val_notes = validate_ranges(out)
    stats["validation_notes"] = val_notes

    if drop_after_validate:
        core = [c for c in MODEL_NUMERIC_FEATURES if c in out.columns]
        out = out.dropna(subset=core).reset_index(drop=True)

    out = create_standardized_popularity(out)
    out, thresh = create_sticky_label(out, percentile=0.8)
    stats["sticky_threshold"] = thresh
    stats["sticky_balance"] = float(out["sticky"].mean()) if len(out) else 0.0
    stats["n_rows_final"] = len(out)

    return out, stats
