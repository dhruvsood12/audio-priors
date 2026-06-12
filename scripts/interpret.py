"""Phase 4 interpretability CLI.

Loads ``data/processed/tracks.parquet``, rebuilds the labeled subset
under the same protocol as ``scripts/train.py`` (grouped split by
default, popularity threshold fit on the train fence only, frozen
hyperparameters from ``configs/hparams.json``), refits the LightGBM, and
runs four interpretability analyses with figures and tables. Keeping the
protocol identical to the metrics table matters: a model card must not
quote attribution and calibration from one experiment beside metrics
from another.

Writes:

- ``outputs/figures/06_shap_summary.png``
- ``outputs/figures/07_shap_bar.png``
- ``outputs/figures/08_logistic_coefficients.png``
- ``outputs/figures/09_permutation_importance.png``
- ``outputs/figures/10_calibration.png``
- ``outputs/tables/logistic_coefficients.csv``
- ``outputs/tables/permutation_importance.csv``
- ``outputs/tables/calibration.json``
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import typer

from audio_priors import interpret, labels, splits

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


_LGBM_FIXED = {
    "n_estimators": 300,
    "bagging_freq": 5,
    "objective": "binary",
    "verbose": -1,
    "random_state": 42,
    "class_weight": "balanced",
    # n_jobs=1: same macOS arm64 OpenMP guard as models.py.
    "n_jobs": 1,
}


@app.command()
def main(
    parquet_path: Path = typer.Option(ROOT / "data" / "processed" / "tracks.parquet"),
    q: float = typer.Option(0.20, help="Top-q fraction for the sticky label."),
    split: str = typer.Option("grouped", help="Protocol arm: grouped or random."),
    test_size: float = typer.Option(0.20, help="Held-out fraction."),
    val_size: float = typer.Option(0.20, help="Validation fraction of the train side."),
    hparams_path: Path = typer.Option(ROOT / "configs" / "hparams.json"),
    coef_resamples: int = typer.Option(200, help="Bootstrap resamples for logistic coefficients."),
    perm_repeats: int = typer.Option(30, help="Permutation importance repeats."),
    figures_dir: Path = typer.Option(ROOT / "outputs" / "figures"),
    tables_dir: Path = typer.Option(ROOT / "outputs" / "tables"),
) -> None:
    """Train LightGBM, then run SHAP / coefficients / permutation / calibration."""

    typer.echo(f"loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=["popularity"]).copy()
    df = df.dropna(subset=FEATURE_COLS).copy()
    df = df.reset_index(drop=True)

    sp = splits.protocol_split(df, split, test_size=test_size, val_size=val_size)
    y_all, pop_thr = labels.sticky_top_q_train_threshold(df, df.index[sp.train], q)
    # No decision threshold is picked here, so the model trains on the full
    # train fence (fit+val), matching what the calibration tables describe.
    X_train = df.iloc[sp.train][FEATURE_COLS]
    y_train = y_all.iloc[sp.train]
    X_test = df.iloc[sp.test][FEATURE_COLS]
    y_test = y_all.iloc[sp.test]
    typer.echo(
        f"arm {split} | train: {len(X_train):,} | test: {len(X_test):,} | "
        f"pop threshold {pop_thr:.1f} | train pos {y_train.mean():.4f}"
    )

    searched = json.loads(hparams_path.read_text())["lightgbm"]
    typer.echo(f"training LightGBM (frozen hyperparameters from {hparams_path.name})")
    lgbm = lgb.LGBMClassifier(**{**searched, **_LGBM_FIXED})
    lgbm.fit(X_train, y_train)

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("computing SHAP values")
    shap_expl = interpret.shap_values_tree(lgbm, X_test)
    interpret.plot_shap_summary(shap_expl, figures_dir / "06_shap_summary.png")
    interpret.plot_shap_bar(shap_expl, figures_dir / "07_shap_bar.png")

    typer.echo("bootstrapping logistic coefficients")
    coef_df = interpret.logistic_coefficients_with_ci(X_train, y_train, n_resamples=coef_resamples)
    coef_df.to_csv(tables_dir / "logistic_coefficients.csv", index=False)
    interpret.plot_logistic_coefficients(coef_df, figures_dir / "08_logistic_coefficients.png")

    typer.echo(f"permutation importance ({perm_repeats} repeats)")
    perm_df = interpret.permutation_importance_with_ci(lgbm, X_test, y_test, n_repeats=perm_repeats)
    perm_df.to_csv(tables_dir / "permutation_importance.csv", index=False)
    interpret.plot_permutation_importance(perm_df, figures_dir / "09_permutation_importance.png")

    typer.echo("calibration + isotonic recalibration")
    cal = interpret.calibration_with_isotonic(
        lgb.LGBMClassifier(**{**searched, **_LGBM_FIXED}),
        X_train,
        y_train,
        X_test,
        y_test,
    )
    (tables_dir / "calibration.json").write_text(json.dumps(cal, indent=2) + "\n")
    interpret.plot_calibration(cal, figures_dir / "10_calibration.png")

    # Top-3 overlap summary printed for the phase report.
    shap_mean_abs = pd.Series(
        shap_expl.values.__abs__().mean(axis=0),
        index=list(X_test.columns),
    ).sort_values(ascending=False)
    perm_top = perm_df.sort_values("importance_mean", ascending=False).head(3)["feature"].tolist()
    shap_top = shap_mean_abs.head(3).index.tolist()
    overlap = sorted(set(shap_top) & set(perm_top))

    typer.echo("")
    typer.echo(f"SHAP top 3:        {shap_top}")
    typer.echo(f"permutation top 3: {perm_top}")
    typer.echo(f"overlap:           {overlap} ({len(overlap)}/3)")
    typer.echo(
        f"calibration Brier: {cal['brier_before']:.5f} -> {cal['brier_after']:.5f} "
        f"(delta = {cal['brier_improvement']:+.5f})"
    )


if __name__ == "__main__":
    app()
