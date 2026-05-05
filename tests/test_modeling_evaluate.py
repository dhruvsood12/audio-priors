"""evaluate_classifier output shape and known-perfect-ranker AUC."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import modeling


class _PerfectRanker:
    """Stand-in classifier whose probability for class 1 equals the first feature."""

    classes_ = np.array([0, 1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X.iloc[:, 0].astype(int).to_numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p1 = X.iloc[:, 0].to_numpy().astype(float)
        return np.column_stack([1 - p1, p1])


def test_evaluate_classifier_returns_expected_keys() -> None:
    X = pd.DataFrame({"f": [0, 1, 0, 1]})
    y = pd.Series([0, 1, 0, 1])
    out = modeling.evaluate_classifier(_PerfectRanker(), X, y)
    assert set(out.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_evaluate_classifier_perfect_ranker_scores_one() -> None:
    X = pd.DataFrame({"f": [0, 1, 0, 1]})
    y = pd.Series([0, 1, 0, 1])
    out = modeling.evaluate_classifier(_PerfectRanker(), X, y)
    assert out["roc_auc"] == 1.0
    assert out["accuracy"] == 1.0
    assert out["f1"] == 1.0
