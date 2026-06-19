"""Stickiness label derivations from popularity.

- :func:`popularity_z`: global z-score.
- :func:`popularity_z_by_genre`: per-genre z-score.
- :func:`popularity_threshold`: the ``(1 - q)``-th popularity quantile.
- :func:`sticky_at_threshold`: binary label against a frozen cutoff.
- :func:`sticky_top_q`: binary, ``1`` for tracks in the top ``q`` fraction
  of the WHOLE frame (deployment and EDA use only; see its docstring).
- :func:`sticky_top_q_train_threshold`: the evaluation-safe variant; the
  cutoff is fit on train rows only and applied everywhere.
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


def popularity_threshold(
    df: pd.DataFrame,
    q: float,
    col: str = "popularity",
) -> float:
    """The ``(1 - q)``-th quantile of ``col`` over the rows of ``df``.

    Fit this on the train fence only when the labels feed an evaluation;
    a full-corpus fit leaks test popularity into the test labels.
    """

    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")
    return float(df[col].quantile(1.0 - q))


def sticky_at_threshold(
    df: pd.DataFrame,
    threshold: float,
    col: str = "popularity",
) -> pd.Series:
    """Binary label: ``1`` where ``col`` >= ``threshold``. NaN yields ``0``."""

    return (df[col] >= threshold).fillna(False).astype(int)


def sticky_top_q(
    df: pd.DataFrame,
    q: float,
    col: str = "popularity",
) -> pd.Series:
    """Binary label: ``1`` if ``col`` is in the top ``q`` fraction.

    Uses the ``(1 - q)``-th quantile as the cutoff. NaN popularity yields
    ``0`` (a row with no popularity signal is not a stickiness positive).

    The quantile is fit on every row of ``df``. That is the right behavior
    for deployment artifacts and EDA over a fixed corpus, and the wrong
    behavior for train/test evaluation: there, use
    :func:`sticky_top_q_train_threshold` so test rows never move the
    cutoff that defines their own labels.
    """

    return sticky_at_threshold(df, popularity_threshold(df, q, col=col), col=col)


def sticky_top_q_train_threshold(
    df: pd.DataFrame,
    train_index: pd.Index,
    q: float,
    col: str = "popularity",
) -> tuple[pd.Series, float]:
    """Labels for ALL rows of ``df`` from a threshold fit on train rows only.

    ``train_index`` selects the train fence by index label. The returned
    threshold is the ``(1 - q)``-th popularity quantile of that fence, and
    the returned series labels every row of ``df`` against it, so test
    labels depend on train popularity only.
    """

    threshold = popularity_threshold(df.loc[train_index], q, col=col)
    return sticky_at_threshold(df, threshold, col=col), threshold


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
