"""Tests for the evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from audio_priors.evaluation import (
    best_f1_threshold,
    bootstrap_metric,
    evaluate_with_ci,
    f1_at_optimal_threshold,
    per_genre_auc,
)


def _toy_scored(seed: int = 0, n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    # Scores partly informative, partly noisy.
    score = y * 0.6 + rng.uniform(0, 0.4, size=n)
    return y.astype(int), score.astype(float)


def test_bootstrap_metric_interval_contains_point_estimate() -> None:
    y, s = _toy_scored()
    point, lo, hi = bootstrap_metric(y, s, roc_auc_score, n_resamples=500, seed=42)
    assert lo <= point <= hi


def test_bootstrap_metric_is_reproducible_under_fixed_seed() -> None:
    y, s = _toy_scored()
    a = bootstrap_metric(y, s, roc_auc_score, n_resamples=200, seed=7)
    b = bootstrap_metric(y, s, roc_auc_score, n_resamples=200, seed=7)
    assert a == b


def test_best_f1_threshold_recovers_perfect_threshold() -> None:
    """A perfect-ranker score yields F1 = 1.0 at the natural cutoff."""
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    threshold, f1 = best_f1_threshold(y, s)
    assert f1 == 1.0
    assert 0.3 < threshold <= 0.7


def test_f1_at_optimal_threshold_matches_best_f1() -> None:
    y = np.array([0, 1, 0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.4, 0.6, 0.2, 0.8])
    _, f1_best = best_f1_threshold(y, s)
    f1_callable = f1_at_optimal_threshold(y, s)
    assert abs(f1_best - f1_callable) < 1e-12


def test_per_genre_auc_filters_small_buckets() -> None:
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 1] * 10)
    s = y * 0.5 + 0.1  # informative
    g = pd.Series(["big"] * 60 + ["small"] * 40)  # both above min_count=20
    table = per_genre_auc(y, s, g, min_count=20)
    assert {"big", "small"}.issubset(set(table["genre"]))

    # With min_count=80, only 'big' (60) is dropped too; both are < 80.
    table_strict = per_genre_auc(y, s, g, min_count=80)
    assert table_strict.empty


def test_evaluate_with_ci_returns_four_metrics() -> None:
    y, s = _toy_scored()
    out = evaluate_with_ci(y, s, n_resamples=200, seed=42)
    assert set(out.keys()) == {"roc_auc", "pr_auc", "f1", "brier"}
    for metric, (point, lo, hi) in out.items():
        assert lo <= point <= hi, f"{metric}: {lo} <= {point} <= {hi} failed"
