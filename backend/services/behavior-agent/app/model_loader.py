"""Load serialized ML models from the models/ artifact directory at startup."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODELS_DIR = BACKEND_ROOT / "ml" / "models"

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
    random_forest: object | None = None
    isolation_forest: object | None = None
    lstm: object | None = None
    feature_names: list[str] = field(default_factory=lambda: list(BEHAVIOR_FEATURE_NAMES))
    loaded: bool = False


def _load_optional(path: Path) -> object | None:
    if not path.exists():
        return None
    return joblib.load(path)


def load_models(models_dir: Path | None = None) -> BehaviorModels:
    """Load all available model artifacts; missing files are tolerated."""
    base = models_dir or DEFAULT_MODELS_DIR
    bundle = BehaviorModels(
        xgboost=_load_optional(base / "xgboost_model.pkl") or _load_optional(base / "xgboost.joblib"),
        random_forest=_load_optional(base / "random_forest.joblib"),
        isolation_forest=_load_optional(base / "isolation_forest_model.pkl")
        or _load_optional(base / "isolation_forest.joblib"),
        lstm=_load_optional(base / "lstm_model.pt") or _load_optional(base / "lstm.pt"),
    )
    bundle.loaded = any(
        m is not None for m in (bundle.xgboost, bundle.random_forest, bundle.isolation_forest, bundle.lstm)
    )
    return bundle


def heuristic_risk_score(features: np.ndarray) -> float:
    """Fallback scorer when no trained model is available."""
    weights = np.array([0.15, 0.05, 0.05, 0.20, 0.15, 0.10, 0.10, 0.05, 0.10, 0.05])
    normalised = features / (np.abs(features) + 1.0)
    raw = float(np.dot(normalised[: len(weights)], weights))
    return min(max(raw, 0.0), 1.0)


def predict_risk(models: BehaviorModels, features: np.ndarray) -> tuple[float, str]:
    """Return (risk_score, model_used)."""
    matrix = np.asarray(features, dtype=float).reshape(1, -1)

    if models.xgboost is not None:
        proba = models.xgboost.predict_proba(matrix)[0]
        return float(proba[1]), "xgboost"

    if models.random_forest is not None:
        proba = models.random_forest.predict_proba(matrix)[0]
        return float(proba[1]), "random_forest"

    if models.isolation_forest is not None:
        score = models.isolation_forest.decision_function(matrix)[0]
        risk = 1.0 - (score + 0.5)
        return min(max(float(risk), 0.0), 1.0), "isolation_forest"

    flat = matrix.reshape(-1)
    return heuristic_risk_score(flat), "heuristic"
