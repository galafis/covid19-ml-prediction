"""End-to-end COVID-19 ML prediction pipeline.

Usage:
    python -m src.pipeline
    python -m src.pipeline --algorithm xgboost --countries BRA USA DEU
    python -m src.pipeline --compare
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import get_config
from src.data_loader import load_processed_data
from src.evaluate import compare_algorithms, evaluate_model
from src.feature_engineering import build_features
from src.model import ALGORITHMS, ModelTrainer


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="COVID-19 ML Prediction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algorithm",
        choices=ALGORITHMS,
        default="random_forest",
        help="ML algorithm to train",
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=None,
        help="ISO-3 codes or country names to filter (e.g. BRA USA DEU)",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date filter YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date filter YYYY-MM-DD",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all available algorithms instead of training one",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def run_pipeline(
    algorithm: str = "random_forest",
    countries: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    compare: bool = False,
) -> dict:
    """Execute the full training and evaluation pipeline.

    Args:
        algorithm: Algorithm identifier.
        countries: Optional country filter.
        start_date: Optional start date (YYYY-MM-DD).
        end_date: Optional end date (YYYY-MM-DD).
        compare: If True, compare all algorithms.

    Returns:
        Evaluation report dict.
    """
    config = get_config()
    logger = logging.getLogger(__name__)

    logger.info("=== COVID-19 ML Prediction Pipeline ===")
    logger.info("Algorithm: %s | Countries: %s", algorithm, countries or "all")

    # 1. Load data
    logger.info("Step 1/4: Loading data...")
    df = load_processed_data(
        countries=countries,
        start_date=start_date,
        end_date=end_date,
    )

    # 2. Feature engineering
    logger.info("Step 2/4: Engineering features...")
    feature_df = build_features(df)

    # 3. Train and/or compare
    if compare:
        logger.info("Step 3/4: Comparing algorithms...")
        comparison = compare_algorithms(feature_df)
        logger.info("Best algorithm by MAE: %s", comparison.iloc[0]["algorithm"])
        return {"comparison": comparison.to_dict(orient="records")}

    logger.info("Step 3/4: Training %s...", algorithm)
    trainer = ModelTrainer(algorithm=algorithm)
    trainer.train(feature_df)

    # 4. Evaluate
    logger.info("Step 4/4: Evaluating...")
    report_path = Path(config.data.processed_path).parent / "reports" / "evaluation.json"
    report = evaluate_model(trainer, feature_df, output_path=report_path)

    # 5. Save model
    trainer.save()

    logger.info("Pipeline complete. MAE=%.2f R2=%.4f",
                report["overall_metrics"]["mae"],
                report["overall_metrics"]["r2"])
    return report


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)
    run_pipeline(
        algorithm=args.algorithm,
        countries=args.countries,
        start_date=args.start_date,
        end_date=args.end_date,
        compare=args.compare,
    )
