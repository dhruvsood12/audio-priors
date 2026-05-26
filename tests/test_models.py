"""Tests for the model trainers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from audio_priors.models import (
    train_genre_prior,
    train_logistic,
    train_random_forest,
)


def _toy_xy(seed: int = 0, n: int = 400) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Three audio-feature columns + a genre column + a binary target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(0, 1, size=(n, 3)), columns=["a", "b", "c"])
    # Target depends on the first feature so models can learn it.
    y_raw = X["a"] + 0.5 * X["b"] + rng.normal(0, 0.3, size=n)
    y = pd.Series((y_raw > 0).astype(int))
    genres = pd.Series(rng.choice(["pop", "rock", "jazz"], size=n))
    return X, y, genres


def test_genre_prior_recovers_per_group_base_rate() -> None:
    g = pd.Series(["pop"] * 10 + ["rock"] * 10)
    y = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    gp = train_genre_prior(g, y)
    assert abs(gp.genre_rates["pop"] - 0.2) < 1e-9
    assert abs(gp.genre_rates["rock"] - 0.6) < 1e-9


def test_genre_prior_falls_back_to_global_rate_for_unknown_genre() -> None:
    g_train = pd.Series(["pop", "pop", "rock", "rock"])
    y_train = pd.Series([1, 0, 1, 1])
    gp = train_genre_prior(g_train, y_train)
    # 'jazz' is unseen at fit time, so it should use the global mean (3/4 = 0.75).
    g_test = pd.Series(["jazz"])
    proba = gp.predict_proba(g_test)
    assert abs(proba[0, 1] - 0.75) < 1e-9


def test_genre_prior_proba_rows_sum_to_one() -> None:
    g = pd.Series(["pop", "rock", "rock"])
    y = pd.Series([1, 0, 1])
    gp = train_genre_prior(g, y)
    proba = gp.predict_proba(pd.Series(["pop", "rock", "unseen"]))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_train_logistic_returns_fitted_pipeline_with_predict_proba() -> None:
    X, y, _ = _toy_xy()
    m = train_logistic(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_train_random_forest_is_seed_deterministic() -> None:
    X, y, _ = _toy_xy()
    m1 = train_random_forest(X, y, n_estimators=50)
    m2 = train_random_forest(X, y, n_estimators=50)
    p1 = m1.predict_proba(X)
    p2 = m2.predict_proba(X)
    assert np.allclose(p1, p2)


def test_train_logistic_beats_chance_on_a_learnable_target() -> None:
    """The toy target is constructed from the features; logistic should beat 0.5 AUC."""
    from sklearn.metrics import roc_auc_score

    X, y, _ = _toy_xy(seed=1)
    m = train_logistic(X, y)
    score = m.predict_proba(X)[:, 1]
    assert roc_auc_score(y, score) > 0.7
