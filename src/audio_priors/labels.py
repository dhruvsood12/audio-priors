"""Stickiness label derivations from popularity.

Four functions:

- :func:`popularity_z`: global z-score.
- :func:`popularity_z_by_genre`: per-genre z-score.
- :func:`sticky_top_q`: binary, ``1`` for tracks in the top ``q`` fraction.
- :func:`sticky_top_q_by_genre`: binary, ``1`` for the top ``q`` within each
  genre. Mitigates genre-popularity confounding.

``q`` is a fraction in ``(0, 1)``. ``sticky_top_q(df, q=0.20)`` labels
the top 20% of tracks. NaN popularity yields NaN z-score and ``0`` label
(rows with no signal are not stickiness positives).
"""

from __future__ import annotations

import pandas as pd

_UNKNOWN_GENRE = "__unknown__"


def popularity_z(df: pd.DataFrame, col: str = "popularity") -> pd.Series:
    """Global z-score of ``col``. Degenerate variance returns zeros."""

    s = df[col].astype(float)
    mu = s.mean()
    sigma = s.std(ddof=0)
    if pd.isna(sigma) or sigma == 0:
        return pd.Series(0.0, index=df.index)
    return (s - mu) / sigma


def popularity_z_by_genre(
    df: pd.DataFrame,
    col: str = "popularity",
    group: str = "genre",
) -> pd.Series:
    """Per-genre z-score of ``col``. NaN genres go into a single bucket."""

    s = df[col].astype(float)
    g = df[group].fillna(_UNKNOWN_GENRE)

    def _zs(x: pd.Series) -> pd.Series:
        sigma = x.std(ddof=0)
        if pd.isna(sigma) or sigma == 0:
            return pd.Series(0.0, index=x.index)
        return (x - x.mean()) / sigma

    return s.groupby(g, group_keys=False).transform(_zs)


def sticky_top_q(
    df: pd.DataFrame,
    q: float,
    col: str = "popularity",
) -> pd.Series:
    """Binary label: ``1`` if ``col`` is in the top ``q`` fraction.

    Uses the ``(1 - q)``-th quantile as the cutoff. NaN popularity yields
    ``0`` (a row with no popularity signal is not a stickiness positive).
    """

    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")
    threshold = df[col].quantile(1.0 - q)
    return (df[col] >= threshold).fillna(False).astype(int)


def sticky_top_q_by_genre(
    df: pd.DataFrame,
    q: float,
    col: str = "popularity",
    group: str = "genre",
) -> pd.Series:
    """Binary label: ``1`` if ``col`` is in the top ``q`` fraction within genre.

    Each genre gets its own ``(1 - q)``-th quantile cutoff. Tracks with NaN
    genre fall into a shared ``__unknown__`` bucket. NaN popularity yields
    ``0``.
    """

    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")
    s = df[col].astype(float)
    g = df[group].fillna(_UNKNOWN_GENRE)
    thresholds = s.groupby(g, group_keys=False).transform(lambda x: x.quantile(1.0 - q))
    return (s >= thresholds).fillna(False).astype(int)
