"""Feature engineering for COVID-19 prediction models.

Builds time-series features: rolling statistics, lag features, growth rates,
and contextual country-level indicators.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature groups for easy selection
TIME_FEATURES = [
    "day_of_year",
    "week_of_year",
    "month",
    "days_since_first_case",
]
ROLLING_FEATURES = [
    "new_cases_7d_avg",
    "new_cases_14d_avg",
    "new_deaths_7d_avg",
    "cases_std_7d",
]
LAG_FEATURES = [
    "new_cases_lag_1",
    "new_cases_lag_7",
    "new_cases_lag_14",
    "new_deaths_lag_7",
]
GROWTH_FEATURES = [
    "cases_growth_rate_7d",
    "deaths_growth_rate_7d",
    "doubling_time_7d",
]
CONTEXT_FEATURES = [
    "cases_per_million",
    "deaths_per_million",
    "vacc_pct_population",
    "gdp_per_capita",
    "hospital_beds_per_thousand",
]

ALL_FEATURES = (
    TIME_FEATURES
    + ROLLING_FEATURES
    + LAG_FEATURES
    + GROWTH_FEATURES
    + CONTEXT_FEATURES
)


def build_features(
    df: pd.DataFrame,
    target_col: str = "new_cases",
    feature_groups: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Build the full feature matrix from processed OWID data.

    Args:
        df: Processed DataFrame (output of data_loader.load_processed_data).
        target_col: Column to use as the prediction target.
        feature_groups: Subset of feature groups to include. Defaults to all.

    Returns:
        DataFrame with engineered features and target column.
    """
    df = df.sort_values(["location", "date"]).copy()

    df = _add_time_features(df)
    df = _add_rolling_features(df)
    df = _add_lag_features(df)
    df = _add_growth_features(df)
    df = _add_context_features(df)

    if feature_groups:
        selected: list[str] = []
        mapping = {
            "time": TIME_FEATURES,
            "rolling": ROLLING_FEATURES,
            "lag": LAG_FEATURES,
            "growth": GROWTH_FEATURES,
            "context": CONTEXT_FEATURES,
        }
        for group in feature_groups:
            selected.extend(mapping.get(group, []))
        keep_cols = [c for c in selected if c in df.columns] + [target_col]
        df = df[["location", "date"] + keep_cols].copy()
    else:
        feature_cols = [c for c in ALL_FEATURES if c in df.columns]
        df = df[["location", "date"] + feature_cols + [target_col]].copy()

    # Drop rows with NaN introduced by lags/rolling windows
    before = len(df)
    df = df.dropna(subset=[target_col])
    logger.info(
        "Feature matrix: %d rows (%d dropped due to NaN). Columns: %s",
        len(df),
        before - len(df),
        list(df.columns),
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Private feature builders
# ---------------------------------------------------------------------------

def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    first_case = df[df["total_cases"] > 0].groupby("location")["date"].transform("min")
    df["days_since_first_case"] = (df["date"] - first_case).dt.days.clip(lower=0)
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("location")["new_cases"]
    df["new_cases_7d_avg"] = grp.transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["new_cases_14d_avg"] = grp.transform(lambda x: x.rolling(14, min_periods=1).mean())
    df["cases_std_7d"] = grp.transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0))
    df["new_deaths_7d_avg"] = (
        df.groupby("location")["new_deaths"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for lag in [1, 7, 14]:
        df[f"new_cases_lag_{lag}"] = df.groupby("location")["new_cases"].transform(
            lambda x, l=lag: x.shift(l)
        )
    df["new_deaths_lag_7"] = df.groupby("location")["new_deaths"].transform(
        lambda x: x.shift(7)
    )
    return df


def _add_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    for col, name in [
        ("new_cases", "cases_growth_rate_7d"),
        ("new_deaths", "deaths_growth_rate_7d"),
    ]:
        rolling_now = df.groupby("location")[col].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        rolling_prev = df.groupby("location")[col].transform(
            lambda x: x.rolling(7, min_periods=1).mean().shift(7)
        )
        df[name] = np.where(
            rolling_prev > 0, (rolling_now - rolling_prev) / rolling_prev, np.nan
        )

    df["doubling_time_7d"] = np.where(
        df["cases_growth_rate_7d"] > 0,
        np.log(2) / np.log1p(df["cases_growth_rate_7d"]),
        np.nan,
    )
    return df


def _add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    pop = df["population"].replace(0, np.nan)
    df["cases_per_million"] = df["total_cases"] / pop * 1_000_000
    df["deaths_per_million"] = df["total_deaths"] / pop * 1_000_000
    df["vacc_pct_population"] = df["people_vaccinated"] / pop
    # gdp_per_capita and hospital_beds_per_thousand already in OWID schema
    return df
