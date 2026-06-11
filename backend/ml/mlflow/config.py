"""Shared MLflow configuration for experiment tracking and model registry."""

from pathlib import Path

MLFLOW_TRACKING_URI: str = "file:./ml/mlflow/runs"
MLFLOW_REGISTRY_URI: str = "file:./ml/mlflow/registry"
EXPERIMENT_NAME: str = "fraud-detection"
MODELS_OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "models"

CHAMPION_ALIAS: str = "champion"
CHALLENGER_ALIAS: str = "challenger"

# Promotion gate — challenger must beat champion on PR-AUC by this margin.
PROMOTION_MIN_DELTA: float = 0.01
