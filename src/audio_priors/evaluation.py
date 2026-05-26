"""Bootstrap confidence intervals, per-genre AUC, and threshold sweep.

Every function operates on raw arrays of ``y_true`` and probability scores;
nothing here knows about the model that produced the scores.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

_UNKNOWN_GENRE = "__unknown__"


def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for any scoring metric.

    Returns ``(point_estimate, ci_lower, ci_upper)`` at the
    ``(alpha/2, 1 - alpha/2)`` quantiles. Resamples that contain only one
    class (which would make AUC undefined) are skipped.
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    point = float(metric_fn(y_true, y_score))

    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        samples.append(float(metric_fn(yt, y_score[idx])))

    low = float(np.percentile(samples, 100.0 * alpha / 2.0))
    high = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
    return point, low, high


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Sweep thresholds and return ``(threshold, F1)`` at the F1-maximum."""

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall have one more element than thresholds.
    denom = precision[:-1] + recall[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2.0 * precision[:-1] * recall[:-1] / denom, 0.0)
    idx = int(np.argmax(f1))
    return float(thresholds[idx]), float(f1[idx])


def f1_at_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """F1 at the threshold that maximizes F1, computed on the same array.

    Useful as a ``metric_fn`` for :func:`bootstrap_metric`.
    """

    threshold, _ = best_f1_threshold(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    return float(f1_score(y_true, y_pred))


def per_genre_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    genres: pd.Series,
    min_count: int = 20,
) -> pd.DataFrame:
    """Per-genre AUC table. Filters out genres with fewer than ``min_count``
    samples or single-class slices (where AUC is undefined)."""

    df = pd.DataFrame(
        {
            "y": np.asarray(y_true),
            "score": np.asarray(y_score),
            "genre": genres.fillna(_UNKNOWN_GENRE).astype(str).values,
        }
    )
    rows: list[dict[str, object]] = []
    for name, group in df.groupby("genre"):
        if len(group) < min_count or group["y"].nunique() < 2:
            continue
        auc = roc_auc_score(group["y"].to_numpy(), group["score"].to_numpy())
        rows.append(
            {
                "genre": name,
                "n": len(group),
                "n_positive": int(group["y"].sum()),
                "auc": float(auc),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["genre", "n", "n_positive", "auc"])
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def evaluate_with_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """Run bootstrap CIs for ROC-AUC, PR-AUC, F1 at optimal threshold, Brier."""

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    return {
        "roc_auc": bootstrap_metric(
            y_true, y_score, roc_auc_score, n_resamples=n_resamples, seed=seed
        ),
        "pr_auc": bootstrap_metric(
            y_true, y_score, average_precision_score, n_resamples=n_resamples, seed=seed
        ),
        "f1": bootstrap_metric(
            y_true, y_score, f1_at_optimal_threshold, n_resamples=n_resamples, seed=seed
        ),
        "brier": bootstrap_metric(
            y_true, y_score, brier_score_loss, n_resamples=n_resamples, seed=seed
        ),
    }
