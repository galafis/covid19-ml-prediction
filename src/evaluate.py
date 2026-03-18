"""Model evaluation utilities.

Computes and logs regression metrics, generates a per-country breakdown,
and exports an evaluation report to JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.model import ModelTrainer, compute_metrics

logger = logging.getLogger(__name__)


def evaluate_model(
    trainer: ModelTrainer,
    df: pd.DataFrame,
    target_col: str = "new_cases",
    test_size: float = 0.2,
    output_path: Optional[Path] = None,
) -> dict:
    """Evaluate model on a held-out test split and return a metrics report.

    Args:
        trainer: A fitted ModelTrainer instance.
        df: Feature DataFrame with target column.
        target_col: Name of the target column.
        test_size: Fraction of data used for testing (chronological split).
        output_path: Optional path to write the JSON report.

    Returns:
        Dict containing overall and per-country metrics.
    """
    # Chronological split (not random!) to respect time-series structure
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    y_test = test_df[target_col].values
    y_pred = trainer.predict(test_df)

    overall = compute_metrics(y_test, y_pred)
    logger.info(
        "Overall evaluation | MAE: %.2f | RMSE: %.2f | R2: %.4f",
        overall["mae"],
        overall["rmse"],
        overall["r2"],
    )

    per_country: dict[str, dict] = {}
    if "location" in test_df.columns:
        for country, grp in test_df.groupby("location"):
            y_c = grp[target_col].values
            y_p = trainer.predict(grp)
            per_country[str(country)] = compute_metrics(y_c, y_p)

    report = {
        "algorithm": trainer.algorithm,
        "test_size": test_size,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "overall_metrics": overall,
        "per_country_metrics": per_country,
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        logger.info("Evaluation report saved to %s", output_path)

    return report


def compare_algorithms(
    df: pd.DataFrame,
    algorithms: Optional[list[str]] = None,
    target_col: str = "new_cases",
    test_size: float = 0.2,
) -> pd.DataFrame:
    """Train and evaluate multiple algorithms side by side.

    Args:
        df: Feature DataFrame.
        algorithms: List of algorithm names to compare.
        target_col: Target column name.
        test_size: Test set fraction.

    Returns:
        DataFrame with one row per algorithm and metric columns.
    """
    from src.model import ALGORITHMS  # avoid circular at module level

    algorithms = algorithms or ["ridge", "random_forest"]
    results = []

    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    for algo in algorithms:
        logger.info("Training %s for comparison...", algo)
        try:
            trainer = ModelTrainer(algorithm=algo)
            trainer.train(train_df, target_col=target_col)
            y_pred = trainer.predict(test_df)
            metrics = compute_metrics(test_df[target_col].values, y_pred)
            results.append({"algorithm": algo, **metrics})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s: %s", algo, exc)

    comparison = pd.DataFrame(results).sort_values("mae")
    logger.info("Algorithm comparison:\n%s", comparison.to_string(index=False))
    return comparison
