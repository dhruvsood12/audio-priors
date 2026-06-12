"""Bootstrap confidence intervals, per-genre AUC, and threshold handling.

Every function operates on raw arrays of ``y_true`` and probability scores;
nothing here knows about the model that produced the scores.

Bootstrap unit. With ``groups=None`` the bootstrap resamples rows. With a
``groups`` array it resamples groups (draw groups with replacement, keep
all of each sampled group's rows), which is the right unit when rows
cluster, e.g. tracks by one artist under an artist-grouped protocol; a row
bootstrap there understates interval width. Resamples that contain a
single class are skipped deterministically and counted, never silently
dropped.

Decision thresholds. :func:`pick_f1_threshold_on_train` chooses a
threshold on a validation slice INSIDE the train fence; the threshold is
then frozen and :func:`f1_at_threshold` scores test data at that fixed
value. :func:`f1_in_sample_oracle` optimizes the threshold on the arrays
it scores and is an in-sample upper bound, reported only as a single
labeled number, never with an interval (an interval around an oracle
estimates the variance of cheating).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import partial

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

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def _resample_indices(
    n: int,
    n_resamples: int,
    seed: int,
    groups: np.ndarray | None,
) -> Iterator[np.ndarray]:
    """Yield bootstrap index arrays: rows, or whole groups when given.

    The row path draws ``n`` rows with replacement per resample, exactly
    matching the v0.1 RNG sequence so legacy runs reproduce. The group
    path draws ``n_groups`` groups with replacement and gathers each
    sampled group's rows.
    """

    rng = np.random.default_rng(seed)
    if groups is None:
        for _ in range(n_resamples):
            yield rng.integers(0, n, size=n)
        return

    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv)
    starts = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts)[:-1]])
    n_groups = len(counts)
    for _ in range(n_resamples):
        gidx = rng.integers(0, n_groups, size=n_groups)
        c = counts[gidx]
        total = int(c.sum())
        rep_starts = np.repeat(starts[gidx], c)
        rep_cum = np.repeat(np.cumsum(c) - c, c)
        offsets = np.arange(total) - rep_cum
        yield order[rep_starts + offsets]


def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: MetricFn,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    groups: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for any scoring metric.

    Returns ``(point_estimate, ci_lower, ci_upper)`` at the
    ``(alpha/2, 1 - alpha/2)`` quantiles. Resamples that contain only one
    class (which would make AUC undefined) are skipped. With ``groups``,
    the resampling unit is the group, not the row.
    """

    point, low, high, _ = bootstrap_metric_detailed(
        y_true, y_score, metric_fn, n_resamples=n_resamples, seed=seed, alpha=alpha, groups=groups
    )
    return point, low, high


def bootstrap_metric_detailed(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: MetricFn,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    groups: np.ndarray | None = None,
) -> tuple[float, float, float, int]:
    """:func:`bootstrap_metric` plus the count of skipped degenerate resamples."""

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    point = float(metric_fn(y_true, y_score))

    samples: list[float] = []
    skipped = 0
    for idx in _resample_indices(len(y_true), n_resamples, seed, groups):
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            skipped += 1
            continue
        samples.append(float(metric_fn(yt, y_score[idx])))

    low = float(np.percentile(samples, 100.0 * alpha / 2.0))
    high = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0)))
    return point, low, high, skipped


def paired_cluster_bootstrap_delta(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metric_fn: MetricFn,
    groups: np.ndarray | None = None,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """CI for ``metric(a) - metric(b)`` on the SAME rows, paired per resample.

    The honest test for ranking claims: both score vectors are evaluated
    on identical resamples, so between-sample variance cancels. With
    ``groups`` the resampling unit is the group (e.g. artist). Returns
    ``(delta_point, ci_lower, ci_upper)``.
    """

    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    point = float(metric_fn(y_true, score_a)) - float(metric_fn(y_true, score_b))

    deltas: list[float] = []
    for idx in _resample_indices(len(y_true), n_resamples, seed, groups):
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        deltas.append(float(metric_fn(yt, score_a[idx])) - float(metric_fn(yt, score_b[idx])))

    low = float(np.percentile(deltas, 100.0 * alpha / 2.0))
    high = float(np.percentile(deltas, 100.0 * (1.0 - alpha / 2.0)))
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


def f1_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    """F1 of the fixed policy ``predict 1 iff score >= threshold``."""

    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return float(f1_score(y_true, y_pred))


def pick_f1_threshold_on_train(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """The F1-maximizing threshold, to be FROZEN before touching test data.

    Call this with the validation slice inside the train fence, never with
    test arrays: a threshold tuned on test turns the F1 column into an
    in-sample oracle (see :func:`f1_in_sample_oracle`).
    """

    return best_f1_threshold(y_true, y_score)[0]


def f1_in_sample_oracle(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """F1 at the threshold that maximizes F1 on the same arrays.

    An in-sample upper bound, not a deployable policy. Report it only as
    a single labeled number; do not bootstrap it.
    """

    threshold, _ = best_f1_threshold(y_true, y_score)
    return f1_at_threshold(y_true, y_score, threshold)


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


@dataclass(frozen=True)
class EvalReport:
    """CI'd metrics plus the labeled oracle bound and the degenerate count.

    ``metrics`` maps metric name to ``(point, ci_lower, ci_upper)``. The
    skip count is identical across metrics because every metric replays
    the same seeded resample sequence.
    """

    metrics: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    f1_oracle: float = float("nan")
    n_degenerate_resamples: int = 0


def evaluate_with_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
    f1_threshold: float | None = None,
    groups: np.ndarray | None = None,
) -> EvalReport:
    """Bootstrap CIs for ROC-AUC, PR-AUC, Brier, and fixed-threshold F1.

    ``f1_threshold`` is the FROZEN decision threshold from
    :func:`pick_f1_threshold_on_train`; it is bound before the bootstrap
    loop runs, so it cannot be refit per resample by construction. F1 at
    0.5 is always reported as the parameter-free fixed policy. The
    in-sample oracle F1 is returned as a single number only.
    """

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    metric_fns: dict[str, MetricFn] = {
        "roc_auc": roc_auc_score,
        "pr_auc": average_precision_score,
        "brier": brier_score_loss,
        "f1_at_05": partial(f1_at_threshold, threshold=0.5),
    }
    if f1_threshold is not None:
        metric_fns["f1_frozen"] = partial(f1_at_threshold, threshold=f1_threshold)

    metrics: dict[str, tuple[float, float, float]] = {}
    skipped = 0
    for name, fn in metric_fns.items():
        point, low, high, skipped = bootstrap_metric_detailed(
            y_true, y_score, fn, n_resamples=n_resamples, seed=seed, groups=groups
        )
        metrics[name] = (point, low, high)

    return EvalReport(
        metrics=metrics,
        f1_oracle=f1_in_sample_oracle(y_true, y_score),
        n_degenerate_resamples=skipped,
    )
