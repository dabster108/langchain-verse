from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.model_loader import BEHAVIOR_FEATURE_NAMES, predict_risk
from shared.constants.service_names import BEHAVIOR_AGENT
from shared.explainability.shap_utils import (
    compute_shap_for_random_forest,
    compute_shap_for_xgboost,
    contributions_to_explanation,
)
from shared.schemas.risk import AgentRiskResponse, SHAPExplanation

router = APIRouter(prefix="/evaluate", tags=["evaluate"])


class EvaluateRequest(BaseModel):
    transaction_id: str
    features: list[float] = Field(
        ...,
        min_length=len(BEHAVIOR_FEATURE_NAMES),
        max_length=len(BEHAVIOR_FEATURE_NAMES),
        description="Feature vector aligned with BEHAVIOR_FEATURE_NAMES.",
    )


@router.post("/risk", response_model=AgentRiskResponse)
async def evaluate_risk(body: EvaluateRequest, request: Request) -> AgentRiskResponse:
    models = request.app.state.models
    feature_array = np.asarray(body.features, dtype=float)
    started = time.perf_counter()

    risk_score, model_used = predict_risk(models, feature_array)
    latency_ms = int((time.perf_counter() - started) * 1000)

    shap: SHAPExplanation | None = _compute_shap(models, feature_array, model_used)

    confidence = 0.85 if models.loaded else 0.60
    reasons = [f"model={model_used}"]
    if not models.loaded:
        reasons.append("no serialized model found — using heuristic fallback")

    return AgentRiskResponse(
        transaction_id=body.transaction_id,
        agent_name=BEHAVIOR_AGENT,
        risk_score=risk_score,
        confidence_score=confidence,
        reasons=reasons,
        shap=shap,
    )


def _compute_shap(models, features: np.ndarray, model_used: str) -> SHAPExplanation | None:
    matrix = np.asarray(features, dtype=float)
    try:
        if model_used == "xgboost" and models.xgboost is not None:
            return compute_shap_for_xgboost(models.xgboost, matrix, BEHAVIOR_FEATURE_NAMES)
        if model_used == "random_forest" and models.random_forest is not None:
            return compute_shap_for_random_forest(
                models.random_forest, matrix, BEHAVIOR_FEATURE_NAMES
            )
    except Exception:
        pass

    # Heuristic linear attribution when SHAP is unavailable.
    weights = np.array([0.15, 0.05, 0.05, 0.20, 0.15, 0.10, 0.10, 0.05, 0.10, 0.05])
    contrib = matrix.reshape(-1)[: len(weights)] * weights
    return contributions_to_explanation(BEHAVIOR_FEATURE_NAMES, contrib, base_value=0.0)
