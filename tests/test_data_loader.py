"""Unit tests for src.data_loader."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_loader import (
    REQUIRED_COLUMNS,
    _generate_synthetic_data,
    validate_schema,
)


class TestGenerateSyntheticData:
    def test_returns_dataframe(self) -> None:
        df = _generate_synthetic_data(n_countries=2, n_days=30)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self) -> None:
        df = _generate_synthetic_data(n_countries=3, n_days=10)
        assert len(df) == 30

    def test_required_columns_present(self) -> None:
        df = _generate_synthetic_data(n_countries=2, n_days=10)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_negative_cases(self) -> None:
        df = _generate_synthetic_data(n_countries=2, n_days=30)
        assert (df["new_cases"] >= 0).all()

    def test_population_positive(self) -> None:
        df = _generate_synthetic_data(n_countries=2, n_days=10)
        assert (df["population"] > 0).all()

    def test_dates_are_datetime(self) -> None:
        df = _generate_synthetic_data(n_countries=2, n_days=10)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])


class TestValidateSchema:
    def test_valid_schema_passes(self) -> None:
        df = _generate_synthetic_data(n_countries=1, n_days=5)
        assert validate_schema(df) is True

    def test_missing_column_raises(self) -> None:
        df = _generate_synthetic_data(n_countries=1, n_days=5)
        df = df.drop(columns=["total_cases"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df)

    def test_empty_dataframe_raises(self) -> None:
        df = pd.DataFrame()
        with pytest.raises(ValueError):
            validate_schema(df)
