"""COVID-19 Data Loader.

Loads and validates pandemic data from Our World in Data (OWID) public dataset.
Fallback: generates realistic synthetic data when the source is unavailable.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.config import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OWID_URL = (
    "https://raw.githubusercontent.com/owid/covid-19-data/master/"
    "public/data/owid-covid-data.csv"
)

REQUIRED_COLUMNS = [
    "iso_code",
    "continent",
    "location",
    "date",
    "total_cases",
    "new_cases",
    "total_deaths",
    "new_deaths",
    "total_vaccinations",
    "people_vaccinated",
    "population",
    "gdp_per_capita",
    "hospital_beds_per_thousand",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_data(
    cache_path: Optional[Path] = None,
    force_download: bool = False,
    timeout: int = 30,
) -> pd.DataFrame:
    """Load raw OWID COVID-19 dataset.

    Args:
        cache_path: Local path to cache the downloaded CSV.
        force_download: Re-download even if cached version exists.
        timeout: HTTP request timeout in seconds.

    Returns:
        Raw DataFrame with OWID schema.
    """
    config = get_config()
    cache_path = cache_path or Path(config.data.raw_path) / "owid-covid-data.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_download:
        logger.info("Loading cached data from %s", cache_path)
        return pd.read_csv(cache_path, low_memory=False)

    logger.info("Downloading OWID dataset from %s", OWID_URL)
    try:
        response = requests.get(OWID_URL, timeout=timeout)
        response.raise_for_status()
        cache_path.write_bytes(response.content)
        logger.info("Dataset saved to %s", cache_path)
        return pd.read_csv(cache_path, low_memory=False)
    except requests.RequestException as exc:
        logger.warning("Download failed (%s). Generating synthetic data.", exc)
        return _generate_synthetic_data()


def load_processed_data(
    countries: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load and filter the processed dataset.

    Args:
        countries: ISO-3166 alpha-3 codes or country names to include.
        start_date: Filter rows on or after this date (YYYY-MM-DD).
        end_date: Filter rows on or before this date (YYYY-MM-DD).

    Returns:
        Filtered and type-cast DataFrame.
    """
    df = load_raw_data()
    df = _cast_types(df)
    df = _filter_world_aggregates(df)

    if countries:
        mask = df["location"].isin(countries) | df["iso_code"].isin(countries)
        df = df[mask].copy()
        logger.info("Filtered to %d countries, %d rows.", len(countries), len(df))

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    logger.info("Loaded processed data: %d rows x %d cols.", *df.shape)
    return df.reset_index(drop=True)


def validate_schema(df: pd.DataFrame) -> bool:
    """Check that all required columns are present.

    Args:
        df: DataFrame to validate.

    Returns:
        True if schema is valid.

    Raises:
        ValueError: When required columns are missing.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.debug("Schema validation passed.")
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce date column and numeric types."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    numeric_cols = [
        c for c in df.columns
        if c not in {"iso_code", "continent", "location", "date", "tests_units"}
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


def _filter_world_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove OWID continent/world aggregate rows (iso_code starts with 'OWID_')."""
    mask = ~df["iso_code"].str.startswith("OWID_", na=False)
    return df[mask].copy()


def _generate_synthetic_data(n_countries: int = 20, n_days: int = 730) -> pd.DataFrame:
    """Generate a realistic synthetic COVID-19 dataset for testing/fallback.

    The synthetic data mimics OWID schema with plausible epidemiological curves.
    All values are randomly generated — NOT real epidemiological data.

    Args:
        n_countries: Number of synthetic countries to generate.
        n_days: Number of days of time series per country.

    Returns:
        Synthetic DataFrame following OWID schema.
    """
    rng = np.random.default_rng(42)
    start = pd.Timestamp("2020-01-01")
    dates = pd.date_range(start, periods=n_days, freq="D")

    continents = ["Europe", "Asia", "Americas", "Africa", "Oceania"]
    country_names = [f"Country_{i:02d}" for i in range(n_countries)]
    iso_codes = [f"CC{i:02d}" for i in range(n_countries)]

    rows: list[dict] = []
    for idx, (country, iso) in enumerate(zip(country_names, iso_codes)):
        population = rng.integers(1_000_000, 200_000_000)
        gdp = rng.uniform(1_000, 60_000)
        beds = rng.uniform(0.5, 10.0)
        continent = continents[idx % len(continents)]

        # Simulate epidemic wave with Gaussian-like daily cases
        peak_day = rng.integers(60, 300)
        scale = rng.uniform(10_000, 500_000)
        daily = scale * np.exp(-0.5 * ((np.arange(n_days) - peak_day) / 60) ** 2)
        daily = np.clip(daily + rng.normal(0, scale * 0.05, n_days), 0, None)

        cumulative = np.cumsum(daily)
        deaths = np.cumsum(daily * rng.uniform(0.005, 0.03))
        vacc = np.cumsum(
            np.clip(
                rng.uniform(0, population * 0.003, n_days)
                * (np.arange(n_days) / n_days),
                0,
                population,
            )
        )

        for d_idx, date in enumerate(dates):
            rows.append(
                {
                    "iso_code": iso,
                    "continent": continent,
                    "location": country,
                    "date": date,
                    "total_cases": cumulative[d_idx],
                    "new_cases": daily[d_idx],
                    "total_deaths": deaths[d_idx],
                    "new_deaths": max(deaths[d_idx] - (deaths[d_idx - 1] if d_idx else 0), 0),
                    "total_vaccinations": vacc[d_idx],
                    "people_vaccinated": vacc[d_idx] * 0.9,
                    "population": population,
                    "gdp_per_capita": gdp,
                    "hospital_beds_per_thousand": beds,
                }
            )

    df = pd.DataFrame(rows)
    logger.info(
        "Generated synthetic dataset: %d rows x %d cols.", len(df), df.shape[1]
    )
    return df
