"""Interpretability and calibration for the winning model.

Four analyses, each with confidence intervals where applicable:

- :func:`shap_values_tree` wraps SHAP's ``TreeExplainer`` for a tree model.
- :func:`logistic_coefficients_with_ci` bootstraps a balanced logistic
  regression and returns per-feature point estimates with 95% CIs on
  standardized features.
- :func:`permutation_importance_with_ci` calls sklearn's permutation
  importance with ``n_repeats >= 30`` and converts mean/std to a normal
  CI band.
- :func:`calibration_with_isotonic` fits :class:`CalibratedClassifierCV`
  with ``method='isotonic'`` on a 5-fold CV split of the training set,
  evaluates Brier on the test set before and after, and returns enough
  curve data to plot both calibrations.

Five plotting helpers save figures into ``outputs/figures/``: SHAP
beeswarm, mean ``|SHAP|`` bar, logistic coefficient bars with CIs,
permutation importance bars with CIs, and a calibration curve overlay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler


def shap_values_tree(model: Any, X: pd.DataFrame) -> Any:
    """SHAP values for a tree-based model. Returns a ``shap.Explanation``."""

    import shap

    explainer = shap.TreeExplainer(model)
    return explainer(X)


def logistic_coefficients_with_ci(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_resamples: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Bootstrap-resample, refit balanced logistic, return per-feature CIs.

    Standardizes once with the full training set so the coefficients are
    comparable across features.
    """

    feature_names = list(X_train.columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    y = y_train.to_numpy()

    rng = np.random.default_rng(seed)
    n = len(X_train)
    coefs = np.empty((n_resamples, X_scaled.shape[1]), dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
        lr.fit(X_scaled[idx], y[idx])
        coefs[i, :] = lr.coef_.ravel()

    point = coefs.mean(axis=0)
    lo = np.percentile(coefs, 100.0 * alpha / 2.0, axis=0)
    hi = np.percentile(coefs, 100.0 * (1.0 - alpha / 2.0), axis=0)

    out = pd.DataFrame(
        {
            "feature": feature_names,
            "coef": point,
            "ci_lower": lo,
            "ci_upper": hi,
        }
    )
    out["abs_coef"] = out["coef"].abs()
    return out.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def permutation_importance_with_ci(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Permutation importance with a normal-approximation CI on the mean drop."""

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="roc_auc",
        n_jobs=-1,
    )
    means = result.importances_mean
    stds = result.importances_std
    se = stds / np.sqrt(n_repeats)
    return (
        pd.DataFrame(
            {
                "feature": list(X_test.columns),
                "importance_mean": means,
                "importance_std": stds,
                "ci_lower": means - 1.96 * se,
                "ci_upper": means + 1.96 * se,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def calibration_with_isotonic(
    base_estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Fit isotonic recalibration via ``CalibratedClassifierCV`` (5-fold).

    Returns the Brier score before and after, plus enough quantile-binned
    curve points to plot both calibrations against the diagonal.
    """

    base_estimator.fit(X_train, y_train)
    proba_raw = base_estimator.predict_proba(X_test)[:, 1]
    brier_before = float(brier_score_loss(y_test, proba_raw))
    prob_true_b, prob_pred_b = calibration_curve(
        y_test, proba_raw, n_bins=n_bins, strategy="quantile"
    )

    calibrated = CalibratedClassifierCV(base_estimator, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    proba_iso = calibrated.predict_proba(X_test)[:, 1]
    brier_after = float(brier_score_loss(y_test, proba_iso))
    prob_true_a, prob_pred_a = calibration_curve(
        y_test, proba_iso, n_bins=n_bins, strategy="quantile"
    )

    return {
        "brier_before": brier_before,
        "brier_after": brier_after,
        "brier_improvement": brier_before - brier_after,
        "prob_true_before": prob_true_b.tolist(),
        "prob_pred_before": prob_pred_b.tolist(),
        "prob_true_after": prob_true_a.tolist(),
        "prob_pred_after": prob_pred_a.tolist(),
        "n_bins": n_bins,
    }


def plot_shap_summary(shap_explanation: Any, save_path: str | Path) -> Path:
    """SHAP beeswarm summary plot saved to ``save_path``."""

    import shap

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_explanation, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_shap_bar(shap_explanation: Any, save_path: str | Path) -> Path:
    """Mean ``|SHAP|`` bar plot saved to ``save_path``."""

    import shap

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_explanation, plot_type="bar", show=False)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_logistic_coefficients(coef_df: pd.DataFrame, save_path: str | Path) -> Path:
    """Horizontal bar of coefficients with 95% CI whiskers."""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_df = coef_df.sort_values("coef")
    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.35 * len(sorted_df))))
    lower_err = (sorted_df["coef"] - sorted_df["ci_lower"]).to_numpy()
    upper_err = (sorted_df["ci_upper"] - sorted_df["coef"]).to_numpy()
    colors = ["steelblue" if c >= 0 else "coral" for c in sorted_df["coef"]]
    ax.barh(sorted_df["feature"], sorted_df["coef"], color=colors, alpha=0.85)
    ax.errorbar(
        sorted_df["coef"],
        sorted_df["feature"],
        xerr=np.array([lower_err, upper_err]),
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Coefficient (standardized features, 95% CI)")
    ax.set_title("Logistic regression coefficients")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_permutation_importance(imp_df: pd.DataFrame, save_path: str | Path) -> Path:
    """Horizontal bar of permutation importance with CI whiskers."""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_df = imp_df.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.35 * len(sorted_df))))
    lower_err = (sorted_df["importance_mean"] - sorted_df["ci_lower"]).to_numpy()
    upper_err = (sorted_df["ci_upper"] - sorted_df["importance_mean"]).to_numpy()
    ax.barh(sorted_df["feature"], sorted_df["importance_mean"], color="teal", alpha=0.85)
    ax.errorbar(
        sorted_df["importance_mean"],
        sorted_df["feature"],
        xerr=np.array([lower_err, upper_err]),
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    ax.set_xlabel("ROC-AUC drop when feature is permuted (95% CI)")
    ax.set_title("Permutation importance")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_calibration(calib_data: dict[str, Any], save_path: str | Path) -> Path:
    """Calibration curves before and after isotonic recalibration."""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.plot(
        calib_data["prob_pred_before"],
        calib_data["prob_true_before"],
        "o-",
        color="crimson",
        label=f"Raw  (Brier = {calib_data['brier_before']:.4f})",
    )
    ax.plot(
        calib_data["prob_pred_after"],
        calib_data["prob_true_after"],
        "s-",
        color="steelblue",
        label=f"Isotonic  (Brier = {calib_data['brier_after']:.4f})",
    )
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Calibration of predicted probabilities")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path
