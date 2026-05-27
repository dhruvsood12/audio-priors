"""End-to-end pipeline tests against synthetic data.

These tests exercise the full data layer plus a smoke training pass
without touching the real Kaggle parquet, so they run fast under
``pytest -q`` and on Phase 8 CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from audio_priors import data as ap_data
from audio_priors import features, labels, models
from audio_priors.recommend import build_corpus, l2_normalize


def _synthetic_corpus(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """A small frame mirroring the maharshipandya raw schema."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "track_name": [f"track_{i}" for i in range(n)],
            "artists": [f"artist_{i % 30}" for i in range(n)],
            "track_genre": rng.choice(["pop", "rock", "jazz", "hip hop", "electronic"], size=n),
            "popularity": rng.integers(0, 101, size=n),
            "danceability": np.clip(rng.beta(2, 2, size=n), 0, 1),
            "energy": np.clip(rng.beta(2, 2, size=n), 0, 1),
            "valence": np.clip(rng.beta(2, 2, size=n), 0, 1),
            "acousticness": np.clip(rng.beta(2, 3, size=n), 0, 1),
            "instrumentalness": np.clip(rng.beta(1, 8, size=n), 0, 1),
            "liveness": np.clip(rng.beta(1, 4, size=n), 0, 1),
            "speechiness": np.clip(rng.beta(1, 5, size=n), 0, 1),
            "loudness": rng.uniform(-30, 0, size=n),
            "tempo": rng.uniform(60, 200, size=n),
            "duration_ms": rng.integers(60_000, 480_000, size=n),
        }
    )


def test_data_harmonize_dedup_validate_round_trip(tmp_path) -> None:
    """Harmonize a synthetic maharshipandya frame into canonical shape."""

    raw = _synthetic_corpus(n=200)
    csv_path = tmp_path / "spotify.csv"
    raw.to_csv(csv_path, index=False)

    canonical = ap_data.load_maharshipandya(csv_path)
    assert "track_name" in canonical.columns
    assert "artist_name" in canonical.columns
    assert "genre" in canonical.columns
    assert canonical["source"].iloc[0] == "maharshipandya"

    dedupped, n_dropped = ap_data.deduplicate(canonical)
    assert len(dedupped) + n_dropped == len(canonical)

    cleaned, range_notes = ap_data.validate_ranges(dedupped)
    # Synthetic data is in-range so no rows should drop, no NaN coercion.
    assert len(cleaned) == len(dedupped)
    assert isinstance(range_notes, dict)


def test_full_label_features_train_evaluate_pipeline(tmp_path) -> None:
    """Label → features → split → train → evaluate, end-to-end."""

    raw = _synthetic_corpus(n=2000, seed=7)
    csv_path = tmp_path / "spotify.csv"
    raw.to_csv(csv_path, index=False)

    canonical = ap_data.load_maharshipandya(csv_path)
    cleaned, _ = ap_data.deduplicate(canonical)
    cleaned, _ = ap_data.validate_ranges(cleaned)
    cleaned["sticky"] = labels.sticky_top_q(cleaned, q=0.20)

    X = features.select_audio_features(cleaned)
    y = cleaned["sticky"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    model = models.train_logistic(X_train, y_train)
    score = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, score)
    # No real signal in random synthetic data; AUC near 0.5 is expected.
    assert 0.0 <= auc <= 1.0


def test_recommender_corpus_builds_against_demo_features() -> None:
    """build_corpus produces a usable FAISS index over synthetic features."""

    raw = _synthetic_corpus(n=300, seed=3)
    # Compute the sticky label off the raw frame; load_maharshipandya is not
    # needed here because we already have canonical-looking columns by name.
    raw["sticky"] = labels.sticky_top_q(raw.rename(columns={"track_genre": "genre"}), q=0.20)
    X = raw[list(features.AUDIO_FEATURES)]
    # Pre-standardize so the embedding behaves; cheaper than constructing the helper.
    embedding = l2_normalize((X - X.mean()) / (X.std(ddof=0) + 1e-9))
    corpus = build_corpus(
        X,
        raw["track_genre"],
        raw["artists"],
        raw["sticky"],
        raw["popularity"],
        X_train_embedded=embedding,
    )
    # FAISS index reports the right shape and produces self-similar neighbors.
    assert corpus.index.ntotal == len(raw)
    sims, idx = corpus.index.search(embedding[:1].astype("float32"), 5)
    assert idx[0, 0] == 0
    assert sims[0, 0] > 0.99
