"""Reproduce the v0.1.0 evaluation protocol exactly; write legacy_metrics.csv.

The v2 protocol changed the split, the label threshold fence, the F1
column, and the training fraction at once, so the v2 random arm is NOT
expected to match the v0.1 numbers. This script is the controlled
reference: it re-runs the v0.1 protocol byte-for-byte (full-corpus label
quantile, stratified random split, per-run Optuna search with a
stratified inner CV, row bootstrap, F1 at the test-optimal threshold)
and is the only path expected to reproduce the published v0.1 table.

Known-leaky by design; exists so the three-row comparison in the docs
(legacy / v2-random / v2-grouped) is measured, not asserted.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from audio_priors import evaluation, labels, models

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

MODELS = ("genre_prior", "logistic", "random_forest", "lightgbm", "xgboost")


def _legacy_evaluate(
    y_true: pd.Series, score: pd.Series, n_resamples: int
) -> dict[str, tuple[float, float, float]]:
    """The v0.1 metric set: row bootstrap, oracle F1 bootstrapped (leaky)."""

    y = y_true.to_numpy()
    s = score.to_numpy()
    return {
        "roc_auc": evaluation.bootstrap_metric(y, s, roc_auc_score, n_resamples=n_resamples),
        "pr_auc": evaluation.bootstrap_metric(
            y, s, average_precision_score, n_resamples=n_resamples
        ),
        "f1": evaluation.bootstrap_metric(
            y, s, evaluation.f1_in_sample_oracle, n_resamples=n_resamples
        ),
        "brier": evaluation.bootstrap_metric(y, s, brier_score_loss, n_resamples=n_resamples),
    }


@app.command()
def main(
    parquet_path: Path = typer.Option(ROOT / "data" / "processed" / "tracks.parquet"),
    q: float = typer.Option(0.20),
    test_size: float = typer.Option(0.20),
    n_resamples: int = typer.Option(1000),
    lgbm_trials: int = typer.Option(30),
    xgb_trials: int = typer.Option(30),
    time_budget_s: int = typer.Option(300),
    model: str = typer.Option("all"),
    out_path: Path = typer.Option(ROOT / "outputs" / "tables" / "legacy_metrics.csv"),
) -> None:
    """Re-run the v0.1 protocol; the output should byte-match the v0.1 table."""

    typer.echo(f"loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=["popularity"]).copy()
    df["y"] = labels.sticky_top_q(df, q=q)  # full-corpus fence: the v0.1 leak, on purpose
    df = df.dropna(subset=FEATURE_COLS).copy()
    typer.echo(f"labeled rows: {len(df):,} | positive rate: {df['y'].mean():.4f}")

    X = df[FEATURE_COLS]
    y = df["y"]
    g = df["genre"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    g_train = g.loc[X_train.index]
    g_test = g.loc[X_test.index]
    typer.echo(f"train: {len(X_train):,} | test: {len(X_test):,}")

    want = MODELS if model == "all" else (model,)
    rows: list[dict[str, object]] = []

    for name in want:
        typer.echo(f"== {name} (legacy protocol) ==")
        if name == "genre_prior":
            gp = models.train_genre_prior(g_train, y_train)
            score = pd.Series(gp.predict_proba(g_test)[:, 1], index=y_test.index)
        elif name == "logistic":
            score = pd.Series(
                models.train_logistic(X_train, y_train).predict_proba(X_test)[:, 1],
                index=y_test.index,
            )
        elif name == "random_forest":
            score = pd.Series(
                models.train_random_forest(X_train, y_train).predict_proba(X_test)[:, 1],
                index=y_test.index,
            )
        elif name == "lightgbm":
            m = models.train_lightgbm(
                X_train, y_train, n_trials=lgbm_trials, time_budget_s=time_budget_s
            )
            score = pd.Series(m.predict_proba(X_test)[:, 1], index=y_test.index)
        elif name == "xgboost":
            mx = models.train_xgboost(
                X_train, y_train, n_trials=xgb_trials, time_budget_s=time_budget_s
            )
            score = pd.Series(mx.predict_proba(X_test)[:, 1], index=y_test.index)
        else:
            raise typer.BadParameter(f"unknown model: {name}")

        ci = _legacy_evaluate(y_test, score, n_resamples)
        row: dict[str, object] = {"model": name, "q": q}
        for metric, (point, lo, hi) in ci.items():
            row[metric] = point
            row[f"{metric}_ci_lower"] = lo
            row[f"{metric}_ci_upper"] = hi
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    typer.echo(f"wrote {out_path}")


if __name__ == "__main__":
    app()
