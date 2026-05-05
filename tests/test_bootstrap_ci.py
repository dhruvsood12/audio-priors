"""Bootstrap percentile CI on ROC-AUC."""

from __future__ import annotations

import numpy as np

from src import modeling


def test_bootstrap_ci_contains_point_auc_for_perfect_ranker() -> None:
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    y_score = y.astype(float)
    point, lower, upper = modeling.bootstrap_auc_ci(y, y_score, n_resamples=200)
    assert point == 1.0
    assert lower <= point <= upper


def test_bootstrap_ci_is_reproducible_with_same_seed() -> None:
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    y_score = rng.uniform(0, 1, size=n)
    a = modeling.bootstrap_auc_ci(y, y_score, n_resamples=200, seed=42)
    b = modeling.bootstrap_auc_ci(y, y_score, n_resamples=200, seed=42)
    assert a == b


def test_bootstrap_ci_widens_for_random_scores() -> None:
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, size=n)
    y_score = rng.uniform(0, 1, size=n)
    point, lower, upper = modeling.bootstrap_auc_ci(y, y_score, n_resamples=300)
    assert 0.0 <= lower <= point <= upper <= 1.0
    assert upper - lower > 0.05
