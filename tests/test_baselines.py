"""Baseline classifiers as sanity floors for accuracy and AUC."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import modeling


def test_majority_class_baseline_predicts_majority() -> None:
    y = pd.Series([0] * 80 + [1] * 20)
    model = modeling.train_majority_class_baseline(y)
    pred = model.predict(np.zeros((100, 1)))
    assert (pred == 0).all()


def test_majority_class_baseline_accuracy_equals_majority_frequency() -> None:
    y = pd.Series([0] * 80 + [1] * 20)
    model = modeling.train_majority_class_baseline(y)
    metrics = modeling.evaluate_classifier(model, pd.DataFrame(np.zeros((100, 1))), y)
    assert metrics["accuracy"] == 0.8


def test_stratified_random_baseline_is_reproducible() -> None:
    y = pd.Series([0] * 80 + [1] * 20)
    m1 = modeling.train_stratified_random_baseline(y, random_state=42)
    m2 = modeling.train_stratified_random_baseline(y, random_state=42)
    X = np.zeros((100, 1))
    assert np.array_equal(m1.predict(X), m2.predict(X))
