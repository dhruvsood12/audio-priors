"""Train every Phase 3 model, evaluate with bootstrap CIs, write metrics.csv.

Default invocation runs the full panel and writes
``outputs/tables/metrics.csv`` and ``outputs/tables/per_genre_auc.csv``.
The ``--model`` flag subsets to a single model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from sklearn.model_selection import train_test_split

from audio_priors import data as ap_data  # noqa: F401  (kept for symmetry; data module loadable)
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


def _ci_row(
    name: str, label_q: float, ci: dict[str, tuple[float, float, float]]
) -> dict[str, object]:
    row: dict[str, object] = {"model": name, "q": label_q}
    for metric, (point, lo, hi) in ci.items():
        row[metric] = point
        row[f"{metric}_ci_lower"] = lo
        row[f"{metric}_ci_upper"] = hi
    return row


@app.command()
def main(
    parquet_path: Path = typer.Option(ROOT / "data" / "processed" / "tracks.parquet"),
    label: str = typer.Option(
        "sticky_top_q", help="Label function name (sticky_top_q only for now)."
    ),
    q: float = typer.Option(0.20, help="Top-q fraction for the binary label."),
    test_size: float = typer.Option(0.20, help="Stratified hold-out fraction."),
    n_resamples: int = typer.Option(1000, help="Bootstrap resamples per metric."),
    lgbm_trials: int = typer.Option(30, help="Optuna trials for LightGBM."),
    xgb_trials: int = typer.Option(30, help="Optuna trials for XGBoost."),
    time_budget_s: int = typer.Option(300, help="Seconds per Optuna study."),
    model: str = typer.Option(
        "all", help="One of genre_prior, logistic, random_forest, lightgbm, xgboost, all."
    ),
    metrics_path: Path = typer.Option(ROOT / "outputs" / "tables" / "metrics.csv"),
    per_genre_path: Path = typer.Option(ROOT / "outputs" / "tables" / "per_genre_auc.csv"),
) -> None:
    """Train models, bootstrap CIs, write metrics.csv and per-genre AUC."""

    typer.echo(f"loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=["popularity"]).copy()
    if label != "sticky_top_q":
        raise typer.BadParameter(f"unsupported label: {label}")
    df["y"] = labels.sticky_top_q(df, q=q)
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
    test_scores: dict[str, pd.Series] = {}

    if "genre_prior" in want:
        typer.echo("== genre prior ==")
        gp = models.train_genre_prior(g_train, y_train)
        score = gp.predict_proba(g_test)[:, 1]
        ci = evaluation.evaluate_with_ci(y_test.to_numpy(), score, n_resamples=n_resamples)
        rows.append(_ci_row("genre_prior", q, ci))
        test_scores["genre_prior"] = pd.Series(score, index=y_test.index)

    if "logistic" in want:
        typer.echo("== logistic ==")
        m = models.train_logistic(X_train, y_train)
        score = m.predict_proba(X_test)[:, 1]
        ci = evaluation.evaluate_with_ci(y_test.to_numpy(), score, n_resamples=n_resamples)
        rows.append(_ci_row("logistic", q, ci))
        test_scores["logistic"] = pd.Series(score, index=y_test.index)

    if "random_forest" in want:
        typer.echo("== random forest ==")
        m = models.train_random_forest(X_train, y_train)
        score = m.predict_proba(X_test)[:, 1]
        ci = evaluation.evaluate_with_ci(y_test.to_numpy(), score, n_resamples=n_resamples)
        rows.append(_ci_row("random_forest", q, ci))
        test_scores["random_forest"] = pd.Series(score, index=y_test.index)

    if "lightgbm" in want:
        typer.echo(f"== lightgbm Optuna search ({lgbm_trials} trials, {time_budget_s}s budget) ==")
        m = models.train_lightgbm(
            X_train, y_train, n_trials=lgbm_trials, time_budget_s=time_budget_s
        )
        score = m.predict_proba(X_test)[:, 1]
        ci = evaluation.evaluate_with_ci(y_test.to_numpy(), score, n_resamples=n_resamples)
        rows.append(_ci_row("lightgbm", q, ci))
        test_scores["lightgbm"] = pd.Series(score, index=y_test.index)

    if "xgboost" in want:
        typer.echo(f"== xgboost Optuna search ({xgb_trials} trials, {time_budget_s}s budget) ==")
        m = models.train_xgboost(X_train, y_train, n_trials=xgb_trials, time_budget_s=time_budget_s)
        score = m.predict_proba(X_test)[:, 1]
        ci = evaluation.evaluate_with_ci(y_test.to_numpy(), score, n_resamples=n_resamples)
        rows.append(_ci_row("xgboost", q, ci))
        test_scores["xgboost"] = pd.Series(score, index=y_test.index)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    typer.echo(f"wrote {metrics_path}")

    # Per-genre AUC on the best tuned model (highest roc_auc point estimate
    # among the non-baseline models). Falls back to whatever ran.
    pick = max(
        (r for r in rows if r["model"] != "genre_prior"),
        key=lambda r: float(r["roc_auc"]),
        default=rows[0] if rows else None,
    )
    if pick is not None:
        name = str(pick["model"])
        per_g = evaluation.per_genre_auc(y_test.to_numpy(), test_scores[name].to_numpy(), g_test)
        per_g.insert(0, "model", name)
        per_genre_path.parent.mkdir(parents=True, exist_ok=True)
        per_g.to_csv(per_genre_path, index=False)
        typer.echo(f"wrote {per_genre_path} ({len(per_g)} genres above min_count, model={name})")


if __name__ == "__main__":
    app()
