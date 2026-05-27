"""audio-priors Streamlit demo (Phase 6).

Two input modes:

A. **Track search**: pick a track from the corpus, see the predicted
   sticky probability plus a SHAP contribution bar per audio feature.
B. **Manual sliders**: set the ten audio features by hand, see the
   predicted sticky probability and the top-10 nearest tracks by
   audio similarity (FAISS cosine).

All artifacts live under ``outputs/models/`` and are produced by
``python scripts/prepare_app.py`` before launch. The app caches the
model, scaler, FAISS index, and metadata so boot stays under the
brief's 5-second bar after the first run.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "outputs" / "models"

FEATURE_COLS = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "loudness",
    "tempo",
    "duration_ms",
]


def feature_slider_spec() -> dict[str, tuple[float, float, float]]:
    """Per-feature (min, max, default) tuples for the slider mode."""

    return {
        "danceability": (0.0, 1.0, 0.6),
        "energy": (0.0, 1.0, 0.6),
        "valence": (0.0, 1.0, 0.5),
        "acousticness": (0.0, 1.0, 0.3),
        "instrumentalness": (0.0, 1.0, 0.05),
        "liveness": (0.0, 1.0, 0.15),
        "speechiness": (0.0, 1.0, 0.08),
        "loudness": (-30.0, 0.0, -7.0),
        "tempo": (40.0, 240.0, 120.0),
        "duration_ms": (30000.0, 600000.0, 210000.0),
    }


def _missing_artifacts() -> list[Path]:
    paths = [
        MODELS_DIR / "lightgbm.pkl",
        MODELS_DIR / "scaler.pkl",
        MODELS_DIR / "faiss.index",
        MODELS_DIR / "embeddings.npy",
        MODELS_DIR / "metadata.parquet",
    ]
    return [p for p in paths if not p.is_file()]


@st.cache_resource(show_spinner="Loading LightGBM…")
def load_model() -> object:
    return pickle.loads((MODELS_DIR / "lightgbm.pkl").read_bytes())


@st.cache_resource(show_spinner="Loading scaler…")
def load_scaler() -> object:
    return pickle.loads((MODELS_DIR / "scaler.pkl").read_bytes())


@st.cache_resource(show_spinner="Loading FAISS index…")
def load_faiss() -> faiss.Index:
    return faiss.read_index(str(MODELS_DIR / "faiss.index"))


@st.cache_resource(show_spinner="Loading SHAP explainer…")
def load_explainer(_model: object) -> object:
    import shap

    return shap.TreeExplainer(_model)


@st.cache_data(show_spinner="Loading track metadata…")
def load_metadata() -> pd.DataFrame:
    return pd.read_parquet(MODELS_DIR / "metadata.parquet")


def predict_sticky(model: object, features: np.ndarray) -> float:
    """Single-row sticky probability from a fitted classifier."""

    proba = model.predict_proba(features.reshape(1, -1))[0, 1]  # type: ignore[attr-defined]
    return float(proba)


def shap_contribution_plot(explainer: object, features: np.ndarray):
    """Horizontal bar of per-feature SHAP contributions for one row."""

    sv = explainer(features.reshape(1, -1))  # type: ignore[operator]
    contributions = pd.Series(sv.values[0], index=FEATURE_COLS).sort_values()
    colors = ["steelblue" if v >= 0 else "coral" for v in contributions]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(contributions.index, contributions.values, color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("SHAP contribution (+ pushes toward sticky)")
    ax.set_title("Per-feature contribution")
    fig.tight_layout()
    return fig


def nearest_tracks(
    scaler: object,
    index: faiss.Index,
    metadata: pd.DataFrame,
    features: np.ndarray,
    k: int = 10,
) -> pd.DataFrame:
    """Return the K nearest training tracks by audio similarity."""

    standardized = scaler.transform(features.reshape(1, -1))  # type: ignore[attr-defined]
    norm = np.linalg.norm(standardized, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    query = (standardized / norm).astype(np.float32)
    sims, idx = index.search(query, k)
    out = metadata.iloc[idx[0]][["track_name", "artist_name", "genre", "popularity"]].copy()
    out.insert(0, "similarity", sims[0].round(4))
    return out.reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="audio-priors demo", layout="wide")
    st.title("audio-priors")
    st.caption(
        "Audio-feature priors for cold-start music recommendation. "
        "LightGBM sticky-probability + FAISS cosine retrieval over the "
        "10-feature audio embedding."
    )

    missing = _missing_artifacts()
    if missing:
        st.error(
            "Missing pre-built artifacts:\n"
            + "\n".join(f"- `{p.relative_to(ROOT)}`" for p in missing)
            + "\n\nRun `python scripts/prepare_app.py` first."
        )
        st.stop()

    metadata = load_metadata()
    model = load_model()
    scaler = load_scaler()

    mode = st.radio(
        "Input mode",
        ("Track search", "Manual sliders"),
        horizontal=True,
        help="A: pick a track and see SHAP. B: set features by hand and see neighbors.",
    )

    if mode == "Track search":
        st.subheader("Track search")
        labels_series = (
            metadata["track_name"].astype(str) + " - " + metadata["artist_name"].astype(str)
        )
        choice = st.selectbox(
            "Pick a track",
            options=labels_series.head(2000),
            index=0,
            help="First 2,000 tracks shown to keep the dropdown snappy.",
        )
        row_idx = labels_series.eq(choice).idxmax()
        row = metadata.iloc[row_idx]
        features = row[FEATURE_COLS].to_numpy(dtype=float)

        col_a, col_b = st.columns([1, 2])
        with col_a:
            prob = predict_sticky(model, features)
            st.metric("Sticky probability", f"{prob:.3f}")
            st.metric("Popularity (raw)", int(row["popularity"]))
            st.metric("Sticky (q=0.20 label)", "yes" if row["sticky"] else "no")
            st.write(f"**Genre:** {row['genre']}")
        with col_b:
            explainer = load_explainer(model)
            st.pyplot(shap_contribution_plot(explainer, features))

    else:
        st.subheader("Manual sliders")
        spec = feature_slider_spec()
        sliders: dict[str, float] = {}
        slider_cols = st.columns(2)
        for i, col in enumerate(FEATURE_COLS):
            lo, hi, default = spec[col]
            with slider_cols[i % 2]:
                step = (hi - lo) / 100.0
                sliders[col] = st.slider(col, min_value=lo, max_value=hi, value=default, step=step)
        features = np.array([sliders[c] for c in FEATURE_COLS], dtype=float)

        col_a, col_b = st.columns([1, 2])
        with col_a:
            prob = predict_sticky(model, features)
            st.metric("Sticky probability", f"{prob:.3f}")
        with col_b:
            index = load_faiss()
            neighbors = nearest_tracks(scaler, index, metadata, features, k=10)
            st.write("**Top 10 nearest tracks by audio cosine similarity:**")
            st.dataframe(neighbors, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
