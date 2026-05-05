"""End-to-end smoke: clean -> features -> split -> train -> evaluate."""

from __future__ import annotations

import pandas as pd

from src import data_prep, features, modeling


def test_pipeline_runs_and_returns_well_formed_metrics(
    synthetic_tracks: pd.DataFrame,
) -> None:
    cleaned, _ = data_prep.clean_dataframe(synthetic_tracks)
    assert "sticky" in cleaned.columns

    X, y = features.split_features_target(cleaned)
    assert len(X) == len(y) > 0

    X_train, X_test, y_train, y_test = modeling.train_test_split_stratified(
        X, y, test_size=0.25, random_state=42
    )
    X_train_s, X_test_s, _ = features.scale_train_test(X_train, X_test)
    model = modeling.train_logistic_regression(X_train_s, y_train, random_state=42)
    metrics = modeling.evaluate_classifier(model, X_test_s, y_test)

    assert {"accuracy", "precision", "recall", "f1", "roc_auc"}.issubset(metrics.keys())
    for v in metrics.values():
        assert 0.0 <= v <= 1.0
