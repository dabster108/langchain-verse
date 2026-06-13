"""Shared MLflow configuration for experiment tracking and model registry."""

from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]

# Default local tracking store (view with `mlflow ui` from backend/).
MLFLOW_TRACKING_URI: str = f"file:{BACKEND_ROOT / 'mlruns'}"
MLFLOW_REGISTRY_URI: str = f"file:{BACKEND_ROOT / 'ml' / 'mlflow' / 'registry'}"
MODELS_OUTPUT_DIR: Path = BACKEND_ROOT / "ml" / "models"

CHAMPION_ALIAS: str = "champion"
CHALLENGER_ALIAS: str = "challenger"

# Promotion gate — challenger must beat champion on PR-AUC by this margin.
PROMOTION_MIN_DELTA: float = 0.01
