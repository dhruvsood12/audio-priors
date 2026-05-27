"""Tests for the Streamlit app helpers.

The Streamlit UI is exercised by booting the app in headless mode and
inspecting it (see ``outputs/figures/demo.png`` and the Phase 6 report).
This module covers the pure helper functions that are easier to test
in isolation than to exercise through the live UI.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def app_module():
    """Import ``app.streamlit_app`` without booting Streamlit."""
    repo_root = Path(__file__).resolve().parents[1]
    app_root = repo_root / "app"
    sys.path.insert(0, str(app_root))
    try:
        module = importlib.import_module("streamlit_app")
        yield module
    finally:
        sys.path.remove(str(app_root))


def test_feature_slider_spec_covers_every_feature(app_module) -> None:
    spec = app_module.feature_slider_spec()
    assert set(spec.keys()) == set(app_module.FEATURE_COLS)
    for col, (lo, hi, default) in spec.items():
        assert lo < default < hi, f"{col} default {default} outside ({lo}, {hi})"


def test_predict_sticky_returns_probability(app_module) -> None:
    class _StubModel:
        def predict_proba(self, X):
            return np.array([[0.4, 0.6]])

    prob = app_module.predict_sticky(_StubModel(), np.zeros(len(app_module.FEATURE_COLS)))
    assert 0.0 <= prob <= 1.0
    assert prob == pytest.approx(0.6)


def test_nearest_tracks_returns_expected_columns(app_module) -> None:
    import faiss
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(40, len(app_module.FEATURE_COLS)))
    scaler = StandardScaler().fit(X)
    standardized = scaler.transform(X)
    norm = np.linalg.norm(standardized, axis=1, keepdims=True)
    embeddings = (standardized / np.where(norm == 0, 1.0, norm)).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    metadata = pd.DataFrame(
        {
            "track_name": [f"track_{i}" for i in range(40)],
            "artist_name": [f"artist_{i % 8}" for i in range(40)],
            "genre": rng.choice(["pop", "rock", "jazz"], size=40),
            "popularity": rng.integers(0, 101, size=40),
        }
    )

    query = X[0]
    out = app_module.nearest_tracks(scaler, index, metadata, query, k=5)
    assert list(out.columns) == ["similarity", "track_name", "artist_name", "genre", "popularity"]
    assert len(out) == 5
    # First neighbor of the corpus's own first row is itself, similarity ≈ 1.
    assert out.iloc[0]["similarity"] > 0.99


def test_missing_artifacts_reports_absent_files(app_module, tmp_path, monkeypatch) -> None:
    """If artifacts are missing, _missing_artifacts must report them."""
    monkeypatch.setattr(app_module, "MODELS_DIR", tmp_path)
    missing = app_module._missing_artifacts()
    expected_names = {
        "lightgbm.pkl",
        "scaler.pkl",
        "faiss.index",
        "embeddings.npy",
        "metadata.parquet",
    }
    assert {p.name for p in missing} == expected_names
