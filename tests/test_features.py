"""Feature matrix construction, scaling, and rare-genre bucketing."""

from __future__ import annotations

import pandas as pd

from src import features


def test_build_feature_matrix_returns_default_audio_columns(
    synthetic_tracks: pd.DataFrame,
) -> None:
    X, names = features.build_feature_matrix(synthetic_tracks, include_genre=False)
    assert set(names) == set(features.DEFAULT_AUDIO_FEATURES)
    assert X.shape == (len(synthetic_tracks), len(features.DEFAULT_AUDIO_FEATURES))


def test_scale_train_test_produces_zero_mean_unit_std(
    synthetic_tracks: pd.DataFrame,
) -> None:
    X, _ = features.build_feature_matrix(synthetic_tracks)
    X_train = X.iloc[:150]
    X_test = X.iloc[150:]
    X_train_s, _, _ = features.scale_train_test(X_train, X_test)
    means = X_train_s.mean()
    stds = X_train_s.std(ddof=0)
    assert (means.abs() < 1e-9).all()
    assert ((stds - 1.0).abs() < 1e-9).all()


def test_encode_genre_column_buckets_rare_classes() -> None:
    df = pd.DataFrame({"genre": ["pop", "pop", "pop", "pop", "jazz", "rare"]})
    enc, cols = features.encode_genre_column(df, min_freq=2, drop_first=False)
    assert "genre_pop" in cols
    assert "genre_jazz" not in cols
    assert "genre_rare" not in cols
    assert len(enc) == 6
