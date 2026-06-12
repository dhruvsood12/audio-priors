"""Train the model panel under the v2 protocol, write metrics.csv.

Protocol per arm (random | grouped): split first with no stratification,
fit the popularity quantile on the train fence (fit+val) only, derive
labels everywhere from that frozen scalar, fit each model on train_fit
with hyperparameters loaded read-only from configs/hparams.json, freeze
the F1 threshold on the validation carve, then bootstrap test metrics.
The grouped arm resamples artists (cluster bootstrap); the random arm
keeps the row bootstrap as primary and reports the artist-cluster CI
beside it.

Writes ``outputs/tables/metrics.csv`` (one row per model per arm),
``outputs/tables/per_genre_auc.csv`` (best non-baseline model per arm),
and ``outputs/tables/split_delta.csv`` (the paired cluster-bootstrap
grouped-vs-random delta, scored on the grouped test set by both arms'
models).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import roc_auc_score

from audio_priors import evaluation, labels, models, splits

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
METRIC_ORDER = ("roc_auc", "pr_auc", "f1_frozen", "f1_at_05", "brier")


def _ci_row(
    name: str,
    kind: str,
    label_q: float,
    pop_threshold: float,
    hparams_sha: str,
    sizes: dict[str, int],
    rates: dict[str, float],
    n_unseen_genre: int,
    f1_threshold: float,
    rep: evaluation.EvalReport,
    rep_artist: evaluation.EvalReport,
) -> dict[str, object]:
    row: dict[str, object] = {
        "model": name,
        "split": kind,
        "q": label_q,
        "pop_threshold": pop_threshold,
        "hparams_sha": hparams_sha,
        "n_train_fit": sizes["train_fit"],
        "n_val": sizes["val"],
        "n_test": sizes["test"],
        "train_pos_rate": rates["train"],
        "test_pos_rate": rates["test"],
        "n_test_unseen_genre": n_unseen_genre,
        "f1_threshold": f1_threshold,
        "bootstrap_unit": "artist" if kind == "grouped" else "row",
        "n_degenerate_resamples": rep.n_degenerate_resamples,
    }
    for metric in METRIC_ORDER:
        point, lo, hi = rep.metrics[metric]
        _, lo_a, hi_a = rep_artist.metrics[metric]
        row[metric] = point
        row[f"{metric}_ci_lower"] = lo
        row[f"{metric}_ci_upper"] = hi
        row[f"{metric}_ci_lower_artist"] = lo_a
        row[f"{metric}_ci_upper_artist"] = hi_a
    row["f1_oracle"] = rep.f1_oracle
    return row


def _fit_model(
    name: str,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    g_fit: pd.Series,
    hparams: dict[str, Any],
) -> Any:
    if name == "genre_prior":
        return models.train_genre_prior(g_fit, y_fit)
    if name == "logistic":
        return models.train_logistic(X_fit, y_fit)
    if name == "random_forest":
        return models.train_random_forest(X_fit, y_fit)
    if name == "lightgbm":
        return models.train_lightgbm(X_fit, y_fit, params=hparams["lightgbm"])
    if name == "xgboost":
        return models.train_xgboost(X_fit, y_fit, params=hparams["xgboost"])
    raise ValueError(f"unknown model: {name}")


def _score(name: str, model: Any, X: pd.DataFrame, g: pd.Series) -> np.ndarray:
    inputs = g if name == "genre_prior" else X
    scores: np.ndarray = model.predict_proba(inputs)[:, 1]
    return scores


@app.command()
def main(
    parquet_path: Path = typer.Option(ROOT / "data" / "processed" / "tracks.parquet"),
    q: float = typer.Option(0.20, help="Top-q fraction for the binary label."),
    test_size: float = typer.Option(0.20, help="Held-out fraction."),
    val_size: float = typer.Option(0.20, help="Validation fraction of the train side."),
    n_resamples: int = typer.Option(1000, help="Bootstrap resamples per metric."),
    splits_arg: str = typer.Option(
        "random,grouped", "--splits", help="Comma-separated protocol arms to run."
    ),
    model: str = typer.Option(
        "all", help="One of genre_prior, logistic, random_forest, lightgbm, xgboost, all."
    ),
    min_class_count: int = typer.Option(50, help="Minimum positives and negatives per slice."),
    hparams_path: Path = typer.Option(ROOT / "configs" / "hparams.json"),
    metrics_path: Path = typer.Option(ROOT / "outputs" / "tables" / "metrics.csv"),
    per_genre_path: Path = typer.Option(ROOT / "outputs" / "tables" / "per_genre_auc.csv"),
    delta_path: Path = typer.Option(ROOT / "outputs" / "tables" / "split_delta.csv"),
) -> None:
    """Train the panel under each protocol arm, bootstrap CIs, write tables."""

    typer.echo(f"loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=["popularity"]).copy()
    df = df.dropna(subset=FEATURE_COLS).copy()
    df = df.reset_index(drop=True)
    typer.echo(f"labeled rows: {len(df):,}")

    hparams_bytes = hparams_path.read_bytes()
    hparams: dict[str, Any] = json.loads(hparams_bytes)
    hparams_sha = hashlib.sha256(hparams_bytes).hexdigest()[:12]
    typer.echo(f"hparams {hparams_path.name} sha {hparams_sha} (tuned once, refit per arm)")

    arms = tuple(s.strip() for s in splits_arg.split(",") if s.strip())
    for arm in arms:
        if arm not in splits.SPLIT_KINDS:
            raise typer.BadParameter(f"unknown split arm: {arm}")
    want = MODELS if model == "all" else (model,)

    rows: list[dict[str, object]] = []
    per_genre_frames: list[pd.DataFrame] = []
    fitted: dict[str, dict[str, Any]] = {}
    grouped_ctx: dict[str, Any] = {}

    for kind in arms:
        typer.echo(f"=== arm: {kind} ===")
        sp = splits.protocol_split(df, kind, test_size=test_size, val_size=val_size)
        y_all, pop_thr = labels.sticky_top_q_train_threshold(df, df.index[sp.train], q)
        splits.assert_min_class_counts(y_all, sp, min_class_count)

        X_fit = df.iloc[sp.train_fit][FEATURE_COLS]
        y_fit = y_all.iloc[sp.train_fit]
        g_fit = df.iloc[sp.train_fit]["genre"]
        X_val = df.iloc[sp.val][FEATURE_COLS]
        y_val = y_all.iloc[sp.val]
        g_val = df.iloc[sp.val]["genre"]
        X_test = df.iloc[sp.test][FEATURE_COLS]
        y_test = y_all.iloc[sp.test]
        g_test = df.iloc[sp.test]["genre"]
        artists_test = df.iloc[sp.test]["artist_name"].astype(str).to_numpy()

        sizes = {"train_fit": len(sp.train_fit), "val": len(sp.val), "test": len(sp.test)}
        rates = {
            "train": float(y_all.iloc[sp.train].mean()),
            "test": float(y_test.mean()),
        }
        seen_genres = set(g_fit.dropna().astype(str))
        n_unseen = int((~g_test.fillna("__unknown__").astype(str).isin(seen_genres)).sum())
        typer.echo(
            f"split {sizes['train_fit']:,}/{sizes['val']:,}/{sizes['test']:,} | "
            f"pop threshold {pop_thr:.1f} | train pos {rates['train']:.4f} | "
            f"test pos {rates['test']:.4f} | unseen-genre test rows {n_unseen}"
        )

        fitted[kind] = {}
        test_scores: dict[str, np.ndarray] = {}
        for name in want:
            typer.echo(f"== {name} ({kind}) ==")
            m = _fit_model(name, X_fit, y_fit, g_fit, hparams)
            fitted[kind][name] = m
            f1_thr = evaluation.pick_f1_threshold_on_train(
                y_val.to_numpy(), _score(name, m, X_val, g_val)
            )
            s_test = _score(name, m, X_test, g_test)
            test_scores[name] = s_test

            groups_primary = artists_test if kind == "grouped" else None
            rep = evaluation.evaluate_with_ci(
                y_test.to_numpy(),
                s_test,
                n_resamples=n_resamples,
                f1_threshold=f1_thr,
                groups=groups_primary,
            )
            if kind == "grouped":
                rep_artist = rep
            else:
                rep_artist = evaluation.evaluate_with_ci(
                    y_test.to_numpy(),
                    s_test,
                    n_resamples=n_resamples,
                    f1_threshold=f1_thr,
                    groups=artists_test,
                )
            rows.append(
                _ci_row(
                    name,
                    kind,
                    q,
                    pop_thr,
                    hparams_sha,
                    sizes,
                    rates,
                    n_unseen,
                    f1_thr,
                    rep,
                    rep_artist,
                )
            )

        if kind == "grouped":
            grouped_ctx = {
                "y_test": y_test.to_numpy(),
                "X_test": X_test,
                "g_test": g_test,
                "artists_test": artists_test,
            }

        # Per-genre table for this arm's best non-baseline model.
        arm_rows = [r for r in rows if r["split"] == kind and r["model"] != "genre_prior"]
        if arm_rows:
            pick = max(arm_rows, key=lambda r: float(str(r["roc_auc"])))
            name = str(pick["model"])
            per_g = evaluation.per_genre_auc(y_test.to_numpy(), test_scores[name], g_test)
            per_g.insert(0, "split", kind)
            per_g.insert(0, "model", name)
            per_genre_frames.append(per_g)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    typer.echo(f"wrote {metrics_path}")

    if per_genre_frames:
        per_genre_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(per_genre_frames, ignore_index=True).to_csv(per_genre_path, index=False)
        typer.echo(f"wrote {per_genre_path}")

    # Paired arm delta: the grouped test set is artist-disjoint from grouped
    # training, so scoring it with BOTH arms' models isolates what the
    # random-arm model gains from having seen these artists' other tracks.
    if {"random", "grouped"} <= set(arms) and grouped_ctx:
        delta_rows: list[dict[str, object]] = []
        for name in want:
            s_grouped = _score(
                name, fitted["grouped"][name], grouped_ctx["X_test"], grouped_ctx["g_test"]
            )
            s_random = _score(
                name, fitted["random"][name], grouped_ctx["X_test"], grouped_ctx["g_test"]
            )
            point, lo, hi = evaluation.paired_cluster_bootstrap_delta(
                grouped_ctx["y_test"],
                s_random,
                s_grouped,
                roc_auc_score,
                groups=grouped_ctx["artists_test"],
                n_resamples=n_resamples,
            )
            delta_rows.append(
                {
                    "model": name,
                    "metric": "roc_auc",
                    "delta_random_minus_grouped": point,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "bootstrap_unit": "artist",
                    "n_artists": int(pd.unique(grouped_ctx["artists_test"]).size),
                    "n_rows": len(grouped_ctx["y_test"]),
                    "hparams_sha": hparams_sha,
                }
            )
            typer.echo(
                f"delta[{name}] random-minus-grouped AUC on grouped test: "
                f"{point:+.4f} ({lo:+.4f}, {hi:+.4f})"
            )
        delta_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(delta_rows).to_csv(delta_path, index=False)
        typer.echo(f"wrote {delta_path}")


if __name__ == "__main__":
    app()
