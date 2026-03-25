"""Model training and evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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
