"""Model training and evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    max_iter: int = 2000,
    random_state: int = 42,
    class_weight: str | None = None,
) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
        class_weight=class_weight,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int = 42,
    class_weight: str | None = None,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_majority_class_baseline(y_train: pd.Series) -> DummyClassifier:
    """Always predict the majority class. Floors accuracy; AUC is 0.5 by construction."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit(np.zeros((len(y_train), 1)), y_train)
    return model


def train_stratified_random_baseline(y_train: pd.Series, random_state: int = 42) -> DummyClassifier:
    """Sample labels from the empirical class distribution. Floors AUC at 0.5."""
    model = DummyClassifier(strategy="stratified", random_state=random_state)
    model.fit(np.zeros((len(y_train), 1)), y_train)
    return model


def evaluate_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Return accuracy, precision, recall, F1, ROC-AUC for binary classification."""
    y_pred = model.predict(X_test)
    proba_fn = getattr(model, "predict_proba", None)
    if callable(proba_fn):
        proba = proba_fn(X_test)
        pos = np.where(model.classes_ == 1)[0][0]
        y_score = proba[:, pos]
    else:
        y_score = y_pred.astype(float)

    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_test, y_score))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def bootstrap_auc_ci(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    Bootstrap percentile CI on ROC-AUC.

    Returns (point_auc, lower, upper). Resamples whose y_true contains a single
    class are skipped, since AUC is undefined there.
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    n = len(y_true_arr)
    rng = np.random.default_rng(seed)

    point = float(roc_auc_score(y_true_arr, y_score_arr))

    aucs: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true_arr[idx])) < 2:
            continue
        aucs.append(float(roc_auc_score(y_true_arr[idx], y_score_arr[idx])))

    if not aucs:
        return point, float("nan"), float("nan")

    aucs_arr = np.asarray(aucs)
    lower = float(np.quantile(aucs_arr, alpha / 2))
    upper = float(np.quantile(aucs_arr, 1 - alpha / 2))
    return point, lower, upper


def calibration_data(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Per-bin observed-vs-predicted probabilities for reliability plotting."""
    proba = model.predict_proba(X_test)
    pos = np.where(model.classes_ == 1)[0][0]
    y_score = proba[:, pos]
    prob_true, prob_pred = calibration_curve(y_test, y_score, n_bins=n_bins)
    return pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true})


def permutation_importance_table(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str] | None = None,
    n_repeats: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Sorted permutation importance from sklearn.inspection.permutation_importance."""
    if feature_names is None:
        feature_names = list(X_test.columns)
    result = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=seed)
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def genre_stratified_auc(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    genre_series: pd.Series,
    min_count: int = 30,
) -> pd.DataFrame:
    """Per-genre AUC and sample count for groups with min_count rows and both classes."""
    proba = model.predict_proba(X_test)
    pos = np.where(model.classes_ == 1)[0][0]
    y_score = proba[:, pos]

    df = pd.DataFrame(
        {
            "genre": np.asarray(genre_series),
            "y_true": np.asarray(y_test),
            "y_score": y_score,
        }
    )

    rows: list[dict[str, Any]] = []
    for genre, sub in df.groupby("genre"):
        if len(sub) < min_count or sub["y_true"].nunique() < 2:
            continue
        rows.append(
            {
                "genre": str(genre),
                "n": len(sub),
                "n_sticky": int(sub["y_true"].sum()),
                "auc": float(roc_auc_score(sub["y_true"], sub["y_score"])),
            }
        )

    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def build_classification_report_df(results_by_model: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Tidy comparison table: rows = models, cols = metrics."""
    rows = []
    for name, m in results_by_model.items():
        row = {"model": name, **m}
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_regression(
    model: LinearRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    return {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mse)),
    }


def confusion_matrix_array(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> np.ndarray:
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)


def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified split when possible; falls back if a class is too small."""
    if y.nunique() < 2:
        raise ValueError("Target needs at least two classes for classification.")
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)
