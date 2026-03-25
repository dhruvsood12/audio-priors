"""Presentation-ready plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 120


def _maybe_save(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")


def plot_popularity_distribution(
    df: pd.DataFrame,
    col: str = "popularity",
    save_path: str | Path | None = None,
) -> plt.Figure:
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Distribution of Spotify popularity")
    ax.set_xlabel("Popularity (0–100)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_feature_boxplots(
    df: pd.DataFrame,
    features: list[str],
    target_col: str = "sticky",
    save_path: str | Path | None = None,
) -> plt.Figure:
    _setup_style()
    present = [f for f in features if f in df.columns]
    n = len(present)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No features to plot", ha="center", va="center")
        return fig

    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    for i, feat in enumerate(present):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        sns.boxplot(data=df, x=target_col, y=feat, ax=ax, palette="Set2")
        ax.set_title(feat.replace("_", " ").title())
        ax.set_xlabel("Sticky (0/1)")
    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle("Audio features by stickiness label", y=1.02, fontsize=13)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_scatter_with_trend(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    d = df[[x_col, y_col]].dropna()
    sns.scatterplot(data=d, x=x_col, y=y_col, alpha=0.35, ax=ax, color="darkslateblue")
    if len(d) > 2:
        z = np.polyfit(d[x_col], d[y_col], 1)
        p = np.poly1d(z)
        xs = np.linspace(d[x_col].min(), d[x_col].max(), 50)
        ax.plot(xs, p(xs), color="crimson", lw=2, label="Linear trend")
        ax.legend()
    ax.set_title(f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: list[str],
    save_path: str | Path | None = None,
) -> plt.Figure:
    _setup_style()
    present = [c for c in cols if c in df.columns]
    if len(present) < 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "Not enough columns for heatmap", ha="center", va="center")
        return fig
    corr = df[present].corr()
    fig, ax = plt.subplots(figsize=(max(8, len(present) * 0.6), max(6, len(present) * 0.55)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=False)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_feature_mean_comparison(
    df: pd.DataFrame,
    features: list[str],
    group_col: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of mean features for two groups (e.g. top 10% vs bottom 10% by popularity)."""
    _setup_style()
    present = [f for f in features if f in df.columns]
    if not present or group_col not in df.columns:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Insufficient data for comparison", ha="center", va="center")
        return fig

    summary = df.groupby(group_col)[present].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    summary.T.plot(kind="bar", ax=ax, rot=25)
    ax.set_title("Mean audio features: top 10% vs bottom 10% popularity")
    ax.set_ylabel("Mean value")
    ax.legend(title=group_col)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_confusion_matrix_from_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: str | Path | None = None,
) -> plt.Figure:
    _setup_style()
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(pd.unique(pd.concat([y_test, pd.Series(y_pred)])))
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    title: str = "Feature importance",
    save_path: str | Path | None = None,
) -> plt.Figure:
    _setup_style()
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.25)))
    ax.barh(np.array(feature_names)[order], importances[order], color="teal", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


def plot_logistic_coefficients(
    model: Any,
    feature_names: list[str],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Horizontal bar chart of logistic regression coefficients."""
    _setup_style()
    coef = np.ravel(model.coef_)
    order = np.argsort(np.abs(coef))
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.25)))
    colors = np.where(coef[order] >= 0, "steelblue", "coral")
    ax.barh(np.array(feature_names)[order], coef[order], color=colors, alpha=0.9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Logistic regression coefficients (sticky vs not)")
    ax.set_xlabel("Coefficient (log-odds scale)")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig
