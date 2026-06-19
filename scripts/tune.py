"""One-time hyperparameter search; writes the frozen configs/hparams.json.

The v2 protocol freezes hyperparameters so the random-vs-grouped arm delta
isolates the split and nothing else. This script is the only place Optuna
runs: it tunes on the GROUPED arm's train_fit slice with a GroupKFold
inner CV on artist_name, so the tuner never scores a model on artists it
trained on, and it never sees the validation carve that later freezes the
F1 threshold. Every training script loads the resulting file read-only.

The committed JSON is the reproducibility boundary: downstream runs refit
from it deterministically. Re-running this script may land on different
hyperparameters within search noise; that is expected and only matters if
the file is regenerated and recommitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from audio_priors import labels, models, splits

app = typer.Typer(add_completion=False)
ROOT = Path(__file__).resolve().parents[1]

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


@app.command()
def main(
    parquet_path: Path = typer.Option(ROOT / "data" / "processed" / "tracks.parquet"),
    q: float = typer.Option(0.20, help="Top-q fraction for the binary label."),
    test_size: float = typer.Option(0.20, help="Held-out fraction (matches train.py)."),
    val_size: float = typer.Option(0.20, help="Validation fraction of the train side."),
    n_trials: int = typer.Option(30, help="Optuna trials per model."),
    time_budget_s: int = typer.Option(
        1800, help="Generous per-study cap; 30 trials finish well inside it."
    ),
    out_path: Path = typer.Option(ROOT / "configs" / "hparams.json"),
) -> None:
    """Tune LightGBM and XGBoost once on grouped train_fit; freeze to JSON."""

    typer.echo(f"loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=["popularity"]).copy()
    df = df.dropna(subset=FEATURE_COLS).copy()
    df = df.reset_index(drop=True)

    sp = splits.protocol_split(df, "grouped", test_size=test_size, val_size=val_size)
    y_all, threshold = labels.sticky_top_q_train_threshold(df, df.index[sp.train], q)
    X_fit = df.iloc[sp.train_fit][FEATURE_COLS]
    y_fit = y_all.iloc[sp.train_fit]
    groups_fit = df.iloc[sp.train_fit]["artist_name"].astype(str).to_numpy()
    typer.echo(
        f"tuning on grouped train_fit: {len(X_fit):,} rows, "
        f"{pd.unique(groups_fit).size:,} artists, pop threshold {threshold:.1f}"
    )

    typer.echo(f"== lightgbm Optuna ({n_trials} trials, GroupKFold inner CV) ==")
    lgbm = models.train_lightgbm(
        X_fit, y_fit, n_trials=n_trials, time_budget_s=time_budget_s, groups=groups_fit
    )
    lgbm_searched = {
        k: lgbm.get_params()[k]
        for k in (
            "num_leaves",
            "min_child_samples",
            "learning_rate",
            "feature_fraction",
            "bagging_fraction",
        )
    }

    typer.echo(f"== xgboost Optuna ({n_trials} trials, GroupKFold inner CV) ==")
    xgbm = models.train_xgboost(
        X_fit, y_fit, n_trials=n_trials, time_budget_s=time_budget_s, groups=groups_fit
    )
    xgb_searched = {
        k: xgbm.get_params()[k]
        for k in (
            "max_depth",
            "min_child_weight",
            "learning_rate",
            "subsample",
            "colsample_bytree",
        )
    }

    payload = {
        "meta": {
            "tuned_on": "grouped_train_fit",
            "inner_cv": "GroupKFold(5) on artist_name",
            "n_trials": n_trials,
            "sampler_seed": models.RANDOM_STATE,
            "q": q,
            "test_size": test_size,
            "val_size": val_size,
            "n_rows": len(X_fit),
        },
        "lightgbm": lgbm_searched,
        "xgboost": xgb_searched,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    typer.echo(f"wrote {out_path}")


if __name__ == "__main__":
    app()
