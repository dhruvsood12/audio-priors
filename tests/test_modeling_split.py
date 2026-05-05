"""Stratified train/test split behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import modeling


def test_stratified_split_preserves_class_proportions() -> None:
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame(rng.normal(0, 1, size=(n, 3)), columns=["a", "b", "c"])
    y = pd.Series([0] * 400 + [1] * 100)

    _, _, y_train, y_test = modeling.train_test_split_stratified(
        X, y, test_size=0.2, random_state=42
    )

    assert abs(y_train.mean() - 0.2) < 0.02
    assert abs(y_test.mean() - 0.2) < 0.02


def test_stratified_split_raises_on_single_class_target() -> None:
    X = pd.DataFrame({"a": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 0, 0])
    with pytest.raises(ValueError, match="at least two classes"):
        modeling.train_test_split_stratified(X, y)
