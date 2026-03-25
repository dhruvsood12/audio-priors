"""Feature matrix construction and scaling for modeling."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

DEFAULT_AUDIO_FEATURES = [
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


def get_feature_columns(df: pd.DataFrame, include_duration: bool = True) -> list[str]:
    """Return list of numeric modeling columns present in df."""
    cols = []
    for c in DEFAULT_AUDIO_FEATURES:
        if c not in df.columns:
            continue
        if c == "duration_ms" and not include_duration:
            continue
        cols.append(c)
    return cols


def encode_genre_column(
    df: pd.DataFrame,
    genre_col: str = "genre",
    drop_first: bool = True,
    min_freq: int = 2,
) -> tuple[pd.DataFrame, list[str]]:
    """
    One-hot encode genre; drop rare categories into 'genre_other' bucket.
    Returns encoded frame (aligned index) and list of new column names.
    """
    if genre_col not in df.columns:
        return pd.DataFrame(index=df.index), []

    s = df[genre_col].fillna("unknown").astype(str).str.strip()
    vc = s.value_counts()
    rare = vc[vc < min_freq].index
    s = s.where(~s.isin(rare), other="genre_other")

    dummies = pd.get_dummies(s, prefix="genre", drop_first=drop_first, dtype=float)
    return dummies, list(dummies.columns)


def build_feature_matrix(
    df: pd.DataFrame,
    include_genre: bool = False,
    genre_col: str = "genre",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build X: audio features plus optional one-hot genre columns.
    Returns (X, feature_names).
    """
    feat_cols = get_feature_columns(df)
    X = df[feat_cols].copy()

    names = list(feat_cols)
    if include_genre and genre_col in df.columns:
        enc, gcols = encode_genre_column(df, genre_col=genre_col)
        if len(gcols):
            X = pd.concat([X.reset_index(drop=True), enc.reset_index(drop=True)], axis=1)
            names.extend(gcols)

    return X, names


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "sticky",
    include_genre: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return X, y for classification."""
    X, _ = build_feature_matrix(df, include_genre=include_genre)
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")
    y = df[target_col].copy()
    return X, y


def scale_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on train; transform both; return DataFrames with same columns."""
    scaler = StandardScaler()
    cols = list(X_train.columns)
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    X_train_s = pd.DataFrame(X_tr, columns=cols, index=X_train.index)
    X_test_s = pd.DataFrame(X_te, columns=cols, index=X_test.index)
    return X_train_s, X_test_s, scaler
