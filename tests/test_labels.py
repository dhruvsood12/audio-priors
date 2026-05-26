"""Tests for label derivations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from audio_priors.labels import (
    popularity_z,
    popularity_z_by_genre,
    sticky_top_q,
    sticky_top_q_by_genre,
)


def test_popularity_z_has_mean_zero_and_unit_std() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"popularity": rng.integers(0, 101, size=2000)})
    z = popularity_z(df)
    assert abs(z.mean()) < 1e-9
    assert abs(z.std(ddof=0) - 1.0) < 1e-9


def test_popularity_z_handles_constant_column() -> None:
    """A constant popularity column should produce all zeros, not NaN or inf."""
    df = pd.DataFrame({"popularity": [50.0, 50.0, 50.0]})
    z = popularity_z(df)
    assert (z == 0).all()


def test_popularity_z_by_genre_is_centered_within_group() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "popularity": np.concatenate([rng.integers(0, 50, 1000), rng.integers(50, 101, 1000)]),
            "genre": ["pop"] * 1000 + ["rock"] * 1000,
        }
    )
    z = popularity_z_by_genre(df)
    df["z"] = z
    for _, group in df.groupby("genre"):
        assert abs(group["z"].mean()) < 1e-9


def test_sticky_top_q_labels_fraction_q_at_top() -> None:
    """Roughly q fraction of rows get label 1, picking the top by popularity."""
    df = pd.DataFrame({"popularity": list(range(100))})
    label = sticky_top_q(df, q=0.20)
    # 20 of the top values should be 1; 80 of the bottom should be 0.
    assert label.sum() == 20
    assert label.iloc[-1] == 1
    assert label.iloc[0] == 0


def test_sticky_top_q_rejects_q_outside_open_interval() -> None:
    df = pd.DataFrame({"popularity": [1.0, 2.0, 3.0]})
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="q must be in"):
            sticky_top_q(df, q=bad)


def test_sticky_top_q_by_genre_keeps_top_q_within_each_genre() -> None:
    """Top fraction is computed per genre, not globally."""
    df = pd.DataFrame(
        {
            "popularity": list(range(50)) + list(range(50)),
            "genre": ["pop"] * 50 + ["rock"] * 50,
        }
    )
    label = sticky_top_q_by_genre(df, q=0.20)
    # 10 positives in pop and 10 in rock (top 20% of 50 each).
    assert label.iloc[:50].sum() == 10
    assert label.iloc[50:].sum() == 10


def test_sticky_top_q_treats_nan_popularity_as_zero() -> None:
    df = pd.DataFrame({"popularity": [1.0, 2.0, 3.0, 4.0, np.nan, np.nan]})
    label = sticky_top_q(df, q=0.20)
    # NaN should never be labeled 1.
    assert label.iloc[4] == 0
    assert label.iloc[5] == 0
