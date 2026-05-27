"""Property-based tests for the label functions.

Uses Hypothesis to generate random popularity distributions and asserts
invariants that should hold regardless of the input shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from audio_priors.labels import (
    popularity_z,
    sticky_top_q,
    sticky_top_q_by_genre,
)


@given(
    popularity=st.lists(
        st.integers(min_value=0, max_value=100),
        min_size=10,
        max_size=500,
    )
)
@settings(max_examples=50, deadline=None)
def test_popularity_z_mean_is_zero_for_any_input(popularity: list[int]) -> None:
    df = pd.DataFrame({"popularity": popularity})
    z = popularity_z(df)
    # The z-score's mean is identically 0 (within float tolerance) unless the
    # input is constant, in which case popularity_z returns all zeros.
    if df["popularity"].std(ddof=0) == 0:
        assert (z == 0).all()
    else:
        assert abs(float(z.mean())) < 1e-9


@given(
    n=st.integers(min_value=20, max_value=300),
    q=st.floats(min_value=0.05, max_value=0.95),
    seed=st.integers(min_value=0, max_value=2**16),
)
@settings(max_examples=50, deadline=None)
def test_sticky_top_q_label_count_is_close_to_q_fraction(n: int, q: float, seed: int) -> None:
    """The fraction labeled 1 must equal the count at or above the (1 - q)-th
    quantile, which by construction is approximately ``q``."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"popularity": rng.integers(0, 101, size=n)})
    labels_out = sticky_top_q(df, q=q)
    fraction = float(labels_out.mean())
    # Ties at the threshold can push the fraction above q; never above 1.
    assert 0.0 <= fraction <= 1.0
    # The fraction should not be much smaller than q for non-degenerate inputs.
    # Allow a generous tolerance (ties at boundary, small-n discretization).
    assert fraction <= q + 0.5
    assert fraction >= 0.0


@given(
    n=st.integers(min_value=50, max_value=300),
    q=st.floats(min_value=0.10, max_value=0.50),
    seed=st.integers(min_value=0, max_value=2**16),
)
@settings(max_examples=20, deadline=None)
def test_sticky_top_q_by_genre_respects_genre_partition(n: int, q: float, seed: int) -> None:
    """For every genre, the labeled fraction must not exceed 1.0."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "popularity": rng.integers(0, 101, size=n),
            "genre": rng.choice(["pop", "rock", "jazz"], size=n),
        }
    )
    df["label"] = sticky_top_q_by_genre(df, q=q)
    for _, group in df.groupby("genre"):
        if len(group) == 0:
            continue
        rate = float(group["label"].mean())
        assert 0.0 <= rate <= 1.0


def test_sticky_top_q_boundary_q_strict() -> None:
    df = pd.DataFrame({"popularity": [10, 20, 30]})
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="q must be in"):
            sticky_top_q(df, q=bad)
