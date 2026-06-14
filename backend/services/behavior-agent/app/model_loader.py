"""Load serialized ML models and feature metadata from ml/models/."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODELS_DIR = BACKEND_ROOT / "ml" / "models"
DEFAULT_FEATURE_TABLE = BACKEND_ROOT / "datasets_processed" / "feature_table.csv"

# Legacy 10-dim heuristic vector (used when no trained model is on disk).
BEHAVIOR_FEATURE_NAMES: list[str] = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "txn_count_1h",
    "txn_count_24h",
    "avg_amount_7d",
    "merchant_risk_score",
    "device_age_days",
    "is_new_device",
    "distance_from_home_km",
]


@dataclass
class BehaviorModels:
    xgboost: object | None = None
    isolation_forest: object | None = None
    lstm_state: dict | None = None
    meta_learner: object | None = None
    feature_columns: list[str] = field(default_factory=list)
    loaded: bool = False


def _load_joblib(path: Path) -> object | None:
    if not path.exists():
        return None
    return joblib.load(path)


def _load_torch_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _load_feature_columns(base: Path) -> list[str]:
    path = base / "feature_columns.json"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)["feature_columns"]


def load_feature_table_index(path: Path | None = None) -> dict[str, pd.Series]:
    """Index feature_table rows by txn_id for inference lookups."""
    table_path = path or DEFAULT_FEATURE_TABLE
    if not table_path.exists():
        return {}
    df = pd.read_csv(table_path)
    if "txn_id" not in df.columns:
        return {}
    df = df.set_index("txn_id", drop=False)
    return {str(idx): row for idx, row in df.iterrows()}


def vector_from_row(row: pd.Series, feature_columns: list[str]) -> np.ndarray:
    """Build a model input vector from a feature-table row."""
    values: list[float] = []
    for col in feature_columns:
        val = row.get(col, 0)
        if isinstance(val, (bool, np.bool_)):
            values.append(float(val))
        elif pd.isna(val):
            values.append(0.0)
        else:
            values.append(float(val))
    return np.asarray(values, dtype=float)


def load_models(models_dir: Path | None = None) -> BehaviorModels:
    """Load trained artifacts from ml/models/ (missing files are tolerated)."""
    base = models_dir or DEFAULT_MODELS_DIR
    feature_columns = _load_feature_columns(base)

    meta_path = base / "meta_learner_model.pkl"
    meta_payload = _load_joblib(meta_path)
    meta_model = meta_payload.get("model") if isinstance(meta_payload, dict) else meta_payload

    bundle = BehaviorModels(
        xgboost=_load_joblib(base / "xgboost_model.pkl"),
        isolation_forest=_load_joblib(base / "isolation_forest_model.pkl"),
        lstm_state=_load_torch_checkpoint(base / "lstm_model.pt"),
        meta_learner=meta_model,
        feature_columns=feature_columns,
    )
    bundle.loaded = any(
        m is not None
        for m in (bundle.xgboost, bundle.isolation_forest, bundle.lstm_state, bundle.meta_learner)
    )
    if bundle.loaded:
        logger.info("Loaded models from %s (%d feature columns)", base, len(feature_columns))
    else:
        logger.warning("No model artifacts found in %s — using heuristic fallback", base)
    return bundle


def heuristic_risk_score(features: np.ndarray) -> float:
    weights = np.array([0.15, 0.05, 0.05, 0.20, 0.15, 0.10, 0.10, 0.05, 0.10, 0.05])
    normalised = features / (np.abs(features) + 1.0)
    raw = float(np.dot(normalised[: len(weights)], weights))
    return min(max(raw, 0.0), 1.0)


def predict_risk(
    models: BehaviorModels,
    features: np.ndarray,
    *,
    feature_columns: list[str] | None = None,
) -> tuple[float, str]:
    """Return (risk_score, model_used)."""
    matrix = np.asarray(features, dtype=float).reshape(1, -1)
    cols = feature_columns or models.feature_columns
    n_features = matrix.shape[1]

    if models.xgboost is not None and cols and n_features == len(cols):
        proba = models.xgboost.predict_proba(matrix)[0]
        return float(proba[1]), "xgboost"

    if models.isolation_forest is not None and cols and n_features == len(cols):
        score = models.isolation_forest.decision_function(matrix)[0]
        risk = 1.0 - (score + 0.5)
        return min(max(float(risk), 0.0), 1.0), "isolation_forest"

    flat = matrix.reshape(-1)
    return heuristic_risk_score(flat), "heuristic"
