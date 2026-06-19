"""Tests for the evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from audio_priors.evaluation import (
    best_f1_threshold,
    bootstrap_metric,
    evaluate_with_ci,
    f1_at_threshold,
    f1_in_sample_oracle,
    paired_cluster_bootstrap_delta,
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


def test_f1_in_sample_oracle_matches_best_f1() -> None:
    y = np.array([0, 1, 0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.4, 0.6, 0.2, 0.8])
    _, f1_best = best_f1_threshold(y, s)
    f1_callable = f1_in_sample_oracle(y, s)
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


def test_evaluate_with_ci_returns_fixed_policy_metrics() -> None:
    y, s = _toy_scored()
    rep = evaluate_with_ci(y, s, n_resamples=200, seed=42)
    assert set(rep.metrics.keys()) == {"roc_auc", "pr_auc", "brier", "f1_at_05"}
    for metric, (point, lo, hi) in rep.metrics.items():
        assert lo <= point <= hi, f"{metric}: {lo} <= {point} <= {hi} failed"
    # The oracle is a single labeled number, never a CI triple.
    assert isinstance(rep.f1_oracle, float)
    assert rep.f1_oracle >= rep.metrics["f1_at_05"][0] - 1e-12


def test_evaluate_with_ci_adds_frozen_f1_when_threshold_passed() -> None:
    y, s = _toy_scored()
    rep = evaluate_with_ci(y, s, n_resamples=200, seed=42, f1_threshold=0.5)
    assert "f1_frozen" in rep.metrics
    assert rep.metrics["f1_frozen"][0] == f1_at_threshold(y, s, 0.5)


def test_cluster_bootstrap_reproducible_and_contains_point() -> None:
    y, s = _toy_scored()
    groups = np.array([f"g{i % 40}" for i in range(len(y))])
    a = bootstrap_metric(y, s, roc_auc_score, n_resamples=200, seed=7, groups=groups)
    b = bootstrap_metric(y, s, roc_auc_score, n_resamples=200, seed=7, groups=groups)
    assert a == b
    point, lo, hi = a
    assert lo <= point <= hi


def test_paired_delta_is_zero_against_itself() -> None:
    y, s = _toy_scored()
    point, lo, hi = paired_cluster_bootstrap_delta(y, s, s, roc_auc_score, n_resamples=100)
    assert point == 0.0
    assert lo == 0.0 and hi == 0.0


def test_paired_delta_detects_a_better_scorer() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=600)
    good = y * 0.8 + rng.uniform(0, 0.2, size=600)
    bad = rng.uniform(0, 1, size=600)
    point, lo, _ = paired_cluster_bootstrap_delta(y, good, bad, roc_auc_score, n_resamples=300)
    assert point > 0
    assert lo > 0  # strictly positive interval: the ranking claim is supported
