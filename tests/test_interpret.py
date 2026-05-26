"""Tests for the Phase 4 interpretability helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from audio_priors.interpret import (
    calibration_with_isotonic,
    logistic_coefficients_with_ci,
    permutation_importance_with_ci,
    plot_logistic_coefficients,
    plot_permutation_importance,
)


def _toy_dataset(seed: int = 0, n: int = 600) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(0, 1, size=(n, 4)), columns=["a", "b", "c", "d"])
    # Make the target depend on a and b so the bootstrap recovers nonzero coefs.
    y_raw = 1.2 * X["a"] - 0.8 * X["b"] + rng.normal(0, 0.4, size=n)
    y = pd.Series((y_raw > 0).astype(int))
    return X, y


def test_logistic_coefficients_with_ci_returns_expected_shape() -> None:
    X, y = _toy_dataset()
    df = logistic_coefficients_with_ci(X, y, n_resamples=50, seed=0)
    assert set(df["feature"]) == {"a", "b", "c", "d"}
    for col in ("coef", "ci_lower", "ci_upper", "abs_coef"):
        assert col in df.columns
    # CI must contain the point estimate for every feature.
    assert (df["ci_lower"] <= df["coef"]).all()
    assert (df["coef"] <= df["ci_upper"]).all()


def test_logistic_coefficients_capture_target_dependence() -> None:
    """The bootstrap should produce a CI for ``a`` that excludes zero."""
    X, y = _toy_dataset()
    df = logistic_coefficients_with_ci(X, y, n_resamples=200, seed=0)
    a_row = df.loc[df["feature"] == "a"].iloc[0]
    # ``a`` drives the target, so its CI should not contain zero.
    assert a_row["ci_lower"] > 0 or a_row["ci_upper"] < 0


def test_permutation_importance_with_ci_schema_and_order() -> None:
    X, y = _toy_dataset()
    model = LogisticRegression(class_weight="balanced", max_iter=2000).fit(X, y)
    df = permutation_importance_with_ci(model, X, y, n_repeats=10, seed=0)
    assert set(df["feature"]) == {"a", "b", "c", "d"}
    for col in ("importance_mean", "importance_std", "ci_lower", "ci_upper"):
        assert col in df.columns
    # Sorted descending by importance_mean.
    assert list(df["importance_mean"]) == sorted(df["importance_mean"], reverse=True)


def test_calibration_with_isotonic_returns_brier_pair() -> None:
    """The function returns Brier before and after, plus curve data."""
    X, y = _toy_dataset(seed=1)
    X_train, X_test = X.iloc[:400], X.iloc[400:]
    y_train, y_test = y.iloc[:400], y.iloc[400:]
    model = LogisticRegression(class_weight="balanced", max_iter=2000)
    cal = calibration_with_isotonic(model, X_train, y_train, X_test, y_test, n_bins=5)
    assert {"brier_before", "brier_after", "brier_improvement"}.issubset(cal.keys())
    assert 0.0 < cal["brier_before"] < 1.0
    assert 0.0 < cal["brier_after"] < 1.0
    assert abs(cal["brier_improvement"] - (cal["brier_before"] - cal["brier_after"])) < 1e-12
    assert len(cal["prob_pred_before"]) == len(cal["prob_true_before"])


def test_plot_logistic_coefficients_writes_a_png(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "feature": ["a", "b", "c"],
            "coef": [1.0, -0.5, 0.2],
            "ci_lower": [0.8, -0.7, -0.1],
            "ci_upper": [1.2, -0.3, 0.5],
        }
    )
    out = plot_logistic_coefficients(df, tmp_path / "coef.png")
    assert out.is_file()
    assert out.stat().st_size > 0


def test_plot_permutation_importance_writes_a_png(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "feature": ["a", "b", "c"],
            "importance_mean": [0.05, 0.02, 0.01],
            "importance_std": [0.005, 0.003, 0.001],
            "ci_lower": [0.04, 0.015, 0.008],
            "ci_upper": [0.06, 0.025, 0.012],
        }
    )
    out = plot_permutation_importance(df, tmp_path / "perm.png")
    assert out.is_file()
    assert out.stat().st_size > 0
