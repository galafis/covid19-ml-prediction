"""Model training and management for COVID-19 case prediction.

Supports multiple algorithms via a unified ModelTrainer interface:
- RandomForest (baseline)
- XGBoost (gradient boosting)
- LightGBM (fast gradient boosting)
- Ridge (linear baseline)

All models are wrapped with scikit-learn Pipeline for consistent
pre-processing (imputation + scaling).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from src.config import get_config
from src.feature_engineering import ALL_FEATURES

logger = logging.getLogger(__name__)


ALGORITHMS = ["random_forest", "xgboost", "lightgbm", "ridge"]


class ModelTrainer:
    """Unified training interface for COVID-19 forecasting models."""

    def __init__(self, algorithm: str = "random_forest", **kwargs: Any) -> None:
        """Initialise the trainer.

        Args:
            algorithm: One of 'random_forest', 'xgboost', 'lightgbm', 'ridge'.
            **kwargs: Hyper-parameter overrides forwarded to the estimator.
        """
        if algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}, got '{algorithm}'.")
        self.algorithm = algorithm
        self.kwargs = kwargs
        self.pipeline: Optional[Pipeline] = None
        self.feature_names: list[str] = []
        self.config = get_config()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "new_cases",
        feature_cols: Optional[list[str]] = None,
    ) -> ModelTrainer:
        """Fit the model on the provided feature DataFrame.

        Args:
            df: Feature matrix (output of feature_engineering.build_features).
            target_col: Name of the target column.
            feature_cols: Explicit list of feature columns. Defaults to ALL_FEATURES.

        Returns:
            Self, for method chaining.
        """
        feature_cols = feature_cols or [c for c in ALL_FEATURES if c in df.columns]
        self.feature_names = feature_cols

        X = df[feature_cols].values
        y = df[target_col].values

        estimator = self._build_estimator()
        self.pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
        self.pipeline.fit(X, y)
        logger.info(
            "Trained %s on %d samples with %d features.",
            self.algorithm,
            len(X),
            len(feature_cols),
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions from the fitted pipeline.

        Args:
            df: DataFrame containing at least the training feature columns.

        Returns:
            Numpy array of predictions.
        """
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained yet. Call .train() first.")
        X = df[self.feature_names].values
        return self.pipeline.predict(X)

    def cross_validate(
        self,
        df: pd.DataFrame,
        target_col: str = "new_cases",
        n_splits: int = 5,
    ) -> dict[str, float]:
        """Time-series cross-validation (no future data leakage).

        Args:
            df: Feature matrix.
            target_col: Target column name.
            n_splits: Number of time-series folds.

        Returns:
            Dict with mean and std of MAE, RMSE, and R2 across folds.
        """
        feature_cols = [c for c in ALL_FEATURES if c in df.columns]
        X = df[feature_cols].values
        y = df[target_col].values

        estimator = self._build_estimator()
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = cross_val_score(pipeline, X, y, cv=tscv, scoring="neg_mean_absolute_error")
        mae_scores = -scores
        logger.info(
            "CV MAE: %.2f +/- %.2f",
            mae_scores.mean(),
            mae_scores.std(),
        )
        return {
            "mae_mean": float(mae_scores.mean()),
            "mae_std": float(mae_scores.std()),
        }

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist the trained pipeline to disk."""
        if self.pipeline is None:
            raise RuntimeError("Nothing to save. Train the model first.")
        path = path or Path(self.config.model.artifact_path) / f"{self.algorithm}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"pipeline": self.pipeline, "features": self.feature_names}, fh)
        logger.info("Model saved to %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> ModelTrainer:
        """Load a previously saved trainer from disk."""
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        trainer = cls.__new__(cls)
        trainer.pipeline = payload["pipeline"]
        trainer.feature_names = payload["features"]
        trainer.algorithm = "loaded"
        trainer.config = get_config()
        logger.info("Model loaded from %s", path)
        return trainer

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_estimator(self) -> Any:
        defaults: dict[str, Any] = {
            "random_forest": {
                "n_estimators": self.config.model.n_estimators,
                "max_depth": self.config.model.max_depth,
                "random_state": self.config.model.random_state,
                "n_jobs": -1,
            },
            "xgboost": {
                "n_estimators": self.config.model.n_estimators,
                "max_depth": self.config.model.max_depth or 6,
                "learning_rate": 0.05,
                "random_state": self.config.model.random_state,
                "n_jobs": -1,
            },
            "lightgbm": {
                "n_estimators": self.config.model.n_estimators,
                "max_depth": self.config.model.max_depth or -1,
                "learning_rate": 0.05,
                "random_state": self.config.model.random_state,
                "n_jobs": -1,
                "verbose": -1,
            },
            "ridge": {"alpha": 1.0},
        }
        params = {**defaults.get(self.algorithm, {}), **self.kwargs}

        if self.algorithm == "random_forest":
            return RandomForestRegressor(**params)
        if self.algorithm == "xgboost":
            if not HAS_XGB:
                raise ImportError("xgboost is not installed. Run: pip install xgboost")
            return xgb.XGBRegressor(**params)
        if self.algorithm == "lightgbm":
            if not HAS_LGB:
                raise ImportError("lightgbm is not installed. Run: pip install lightgbm")
            return lgb.LGBMRegressor(**params)
        return Ridge(**{k: v for k, v in params.items() if k == "alpha"})


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics.

    Args:
        y_true: Ground-truth values.
        y_pred: Model predictions.

    Returns:
        Dict with MAE, RMSE, and R2 scores.
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }
