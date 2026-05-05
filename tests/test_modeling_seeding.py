"""Deterministic seeding for logistic regression and random forest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import modeling


def _toy_data(seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    n = 200
    X = pd.DataFrame(rng.normal(0, 1, size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((X["f0"] + 0.5 * X["f1"] + rng.normal(0, 0.5, n) > 0).astype(int))
    return X, y


def test_logistic_regression_is_deterministic_for_fixed_seed() -> None:
    X, y = _toy_data()
    m1 = modeling.train_logistic_regression(X, y, random_state=42)
    m2 = modeling.train_logistic_regression(X, y, random_state=42)
    assert np.allclose(m1.predict_proba(X), m2.predict_proba(X))


def test_random_forest_is_deterministic_for_fixed_seed() -> None:
    X, y = _toy_data()
    m1 = modeling.train_random_forest(X, y, random_state=42)
    m2 = modeling.train_random_forest(X, y, random_state=42)
    assert np.allclose(m1.predict_proba(X), m2.predict_proba(X))
