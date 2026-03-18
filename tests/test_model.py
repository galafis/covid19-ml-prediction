"""Unit tests for src.model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import _generate_synthetic_data
from src.feature_engineering import build_features
from src.model import ModelTrainer, compute_metrics


@pytest.fixture(scope="module")
def feature_df() -> pd.DataFrame:
    """Small synthetic feature matrix for fast tests."""
    raw = _generate_synthetic_data(n_countries=3, n_days=60)
    raw["date"] = pd.to_datetime(raw["date"])
    return build_features(raw)


class TestModelTrainer:
    def test_invalid_algorithm_raises(self) -> None:
        with pytest.raises(ValueError, match="algorithm must be one of"):
            ModelTrainer(algorithm="magic_forest")

    def test_predict_before_train_raises(self, feature_df: pd.DataFrame) -> None:
        trainer = ModelTrainer(algorithm="ridge")
        with pytest.raises(RuntimeError, match="not been trained"):
            trainer.predict(feature_df)

    def test_train_returns_self(self, feature_df: pd.DataFrame) -> None:
        trainer = ModelTrainer(algorithm="ridge")
        result = trainer.train(feature_df)
        assert result is trainer

    def test_predict_shape(self, feature_df: pd.DataFrame) -> None:
        trainer = ModelTrainer(algorithm="ridge")
        trainer.train(feature_df)
        preds = trainer.predict(feature_df)
        assert preds.shape == (len(feature_df),)

    def test_random_forest_trains(self, feature_df: pd.DataFrame) -> None:
        trainer = ModelTrainer(algorithm="random_forest", n_estimators=10)
        trainer.train(feature_df)
        preds = trainer.predict(feature_df)
        assert len(preds) == len(feature_df)

    def test_feature_names_populated(self, feature_df: pd.DataFrame) -> None:
        trainer = ModelTrainer(algorithm="ridge")
        trainer.train(feature_df)
        assert len(trainer.feature_names) > 0


class TestComputeMetrics:
    def test_perfect_prediction(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        metrics = compute_metrics(y, y)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)

    def test_returns_expected_keys(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        metrics = compute_metrics(y, y + 0.1)
        assert set(metrics.keys()) == {"mae", "rmse", "r2"}

    def test_mae_positive(self) -> None:
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 28.0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["mae"] > 0
