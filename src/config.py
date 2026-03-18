"""Pipeline configuration management.

Handles environment variables, default values, and
configuration validation for the ML pipeline.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class DataConfig:
    """Configuration for data sources and storage."""

    brasil_io_url: str = os.getenv(
        "BRASIL_IO_API_URL",
        "https://brasil.io/api/dataset/covid19/caso_full/data/"
    )
    johns_hopkins_url: str = os.getenv(
        "JOHNS_HOPKINS_BASE_URL",
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
    )
    ibge_url: str = os.getenv(
        "IBGE_API_URL",
        "https://servicodados.ibge.gov.br/api/v3/"
    )
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    model_dir: Path = Path(os.getenv("MODEL_DIR", "./models"))


@dataclass
class ModelConfig:
    """Configuration for model training and evaluation."""

    forecast_horizon: int = int(os.getenv("FORECAST_HORIZON", "14"))
    train_test_split: float = float(os.getenv("TRAIN_TEST_SPLIT", "0.8"))
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "1000"))
    target_columns: List[str] = field(
        default_factory=lambda: ["new_cases", "new_deaths"]
    )
    feature_lags: List[int] = field(
        default_factory=lambda: [1, 3, 7, 14]
    )
    rolling_windows: List[int] = field(
        default_factory=lambda: [7, 14, 21]
    )


@dataclass
class APIConfig:
    """Configuration for the prediction API."""

    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))


@dataclass
class PipelineConfig:
    """Main configuration aggregating all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    mlflow_uri: str = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")

    def validate(self) -> bool:
        """Validate configuration values."""
        assert 0 < self.model.train_test_split < 1, "Split must be between 0 and 1"
        assert self.model.forecast_horizon > 0, "Horizon must be positive"
        assert self.model.batch_size > 0, "Batch size must be positive"
        self.data.data_dir.mkdir(parents=True, exist_ok=True)
        self.data.model_dir.mkdir(parents=True, exist_ok=True)
        return True


def get_config() -> PipelineConfig:
    """Factory function to create and validate configuration."""
    config = PipelineConfig()
    config.validate()
    return config
