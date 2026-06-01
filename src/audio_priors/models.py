"""Training functions for the modeling phase.

Five trainers, all fit on a stratified training split and return objects
with a ``predict_proba`` method:

- :class:`GenrePrior` (via :func:`train_genre_prior`): per-genre base rate
  of stickiness. The baseline audio-only models must beat.
- :func:`train_logistic`: scaled logistic regression with balanced class
  weights.
- :func:`train_random_forest`: random forest with balanced class weights.
- :func:`train_lightgbm`: LightGBM with an Optuna search over
  ``num_leaves``, ``min_child_samples``, ``learning_rate``,
  ``feature_fraction``, ``bagging_fraction``. 30 trials, 5-minute time
  budget, 5-fold stratified CV inside training.
- :func:`train_xgboost`: XGBoost with a comparable Optuna search.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
_UNKNOWN_GENRE = "__unknown__"


class GenrePrior:
    """Predicts the per-genre base rate of the positive class.

    Trained on a ``(genres, y)`` pair, predicts the empirical
    ``P(y=1 | genre)`` for each test row. Unknown genres fall back to the
    global training-set positive rate.
    """

    def __init__(self) -> None:
        self.global_rate: float = 0.0
        self.genre_rates: dict[str, float] = {}

    def fit(self, genres: pd.Series, y: pd.Series) -> GenrePrior:
        self.global_rate = float(y.mean())
        g = genres.fillna(_UNKNOWN_GENRE).astype(str)
        rates = y.groupby(g).mean()
        self.genre_rates = {str(k): float(v) for k, v in rates.items()}
        return self

    def predict_proba(self, genres: pd.Series) -> np.ndarray:
        g = genres.fillna(_UNKNOWN_GENRE).astype(str)
        p1 = g.map(self.genre_rates).fillna(self.global_rate).astype(float).to_numpy()
        return np.column_stack([1.0 - p1, p1])


def train_genre_prior(genres_train: pd.Series, y_train: pd.Series) -> GenrePrior:
    return GenrePrior().fit(genres_train, y_train)


def train_logistic(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_estimators: int = 300,
    max_depth: int | None = None,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def _cv_score_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any],
) -> float:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aucs: list[float] = []
    for tr_idx, va_idx in skf.split(X_train, y_train):
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        scores = model.predict_proba(X_train.iloc[va_idx])[:, 1]
        aucs.append(float(roc_auc_score(y_train.iloc[va_idx], scores)))
    return float(np.mean(aucs))


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_trials: int = 30,
    time_budget_s: int = 300,
) -> lgb.LGBMClassifier:
    """Optuna search over LightGBM hyperparameters; refit best on full train."""

    fixed = {
        "bagging_freq": 5,
        "n_estimators": 300,
        "objective": "binary",
        "verbose": -1,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
        # n_jobs=1 to avoid the macOS arm64 LightGBM OpenMP segfault observed
        # with -1 on this corpus size. Training time stays acceptable (~2s on
        # 65K rows); CI runners do not have this issue but the value is safe
        # on both.
        "n_jobs": 1,
    }

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            **fixed,
        }
        return _cv_score_lgbm(X_train, y_train, params)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=time_budget_s,
        show_progress_bar=False,
    )

    best_params = {**study.best_params, **fixed}
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    return model


def _cv_score_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any],
) -> float:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    aucs: list[float] = []
    for tr_idx, va_idx in skf.split(X_train, y_train):
        model = xgb.XGBClassifier(**params)
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        scores = model.predict_proba(X_train.iloc[va_idx])[:, 1]
        aucs.append(float(roc_auc_score(y_train.iloc[va_idx], scores)))
    return float(np.mean(aucs))


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_trials: int = 30,
    time_budget_s: int = 300,
) -> xgb.XGBClassifier:
    """Optuna search over XGBoost hyperparameters; refit best on full train."""

    n_pos = max(int(y_train.sum()), 1)
    n_neg = int(len(y_train) - n_pos)
    fixed = {
        "n_estimators": 300,
        "objective": "binary:logistic",
        "verbosity": 0,
        "random_state": RANDOM_STATE,
        "scale_pos_weight": float(n_neg) / float(n_pos),
        "tree_method": "hist",
        "eval_metric": "auc",
        # Match the LightGBM defaults; XGBoost has not been seen to segfault
        # on macOS arm64 but keeping the thread count predictable is cheap.
        "n_jobs": 1,
    }

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            **fixed,
        }
        return _cv_score_xgb(X_train, y_train, params)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=time_budget_s,
        show_progress_bar=False,
    )

    best_params = {**study.best_params, **fixed}
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    return model
