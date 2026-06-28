from __future__ import annotations

import statistics
from typing import Any


AgentName = str
Weights = dict[AgentName, float]
Breakdown = dict[str, Any]

LAYER1_WEIGHTS: dict[str, Weights] = {
    "ESEWA_P2P": {"velocity": 0.20, "geo": 0.25, "behavior": 0.55},
    "KHALTI_QR": {"velocity": 0.50, "geo": 0.20, "behavior": 0.30},
    "CARD_POS": {"velocity": 0.40, "geo": 0.40, "behavior": 0.20},
    "ATM_WITHDRAWAL": {"velocity": 0.30, "geo": 0.30, "behavior": 0.25},
    "SWIFT_OUTWARD": {"velocity": 0.20, "geo": 0.55, "behavior": 0.25},
    "RTGS": {"velocity": 0.15, "geo": 0.25, "behavior": 0.30},
    "MOBILE_TOPUP": {"velocity": 0.35, "geo": 0.20, "behavior": 0.30},
    "UTILITY_BILL": {"velocity": 0.30, "geo": 0.20, "behavior": 0.35},
}

DEFAULT_LAYER1_WEIGHTS: Weights = {"velocity": 0.33, "geo": 0.33, "behavior": 0.34}

LAYER2_WEIGHTS: dict[str, Weights] = {
    "Rapid transfers": {"velocity": 0.90, "geo": 0.20, "behavior": 0.60},
    "Fraud ring": {"velocity": 0.40, "geo": 0.95, "behavior": 0.85},
    "Money laundering": {"velocity": 0.45, "geo": 0.92, "behavior": 0.78},
    "Novel pattern": {"velocity": 0.35, "geo": 0.50, "behavior": 0.88},
}

DISAGREEMENT_VARIANCE_THRESHOLD = 0.15
DISAGREEMENT_SCORE_BUMP = 0.15


def classify_fraud_pattern(
    velocity_risk: float,
    geo_risk: float,
    behavior_risk: float,
    *,
    velocity_breakdown: Breakdown | None = None,
    geo_breakdown: Breakdown | None = None,
    behavior_breakdown: Breakdown | None = None,
) -> str:
    """Classify the dominant fraud pattern from agent scores and details."""
    velocity_details = velocity_breakdown or {}
    geo_details = geo_breakdown or {}
    behavior_details = behavior_breakdown or {}

    unique_counterparties_1h = _number(
        velocity_details.get("unique_counterparties_1h")
        or velocity_details.get("unique_recipients_1h")
        or velocity_details.get("unique_recipients_risk")
    )
    impossible_travel = _truthy(
        geo_details.get("impossible_travel")
        or geo_details.get("impossible_travel_risk")
    )
    shared_ip = _truthy(geo_details.get("shared_ip") or geo_details.get("shared_ip_risk"))
    fraud_ring_proximity = _truthy(
        geo_details.get("fraud_ring_proximity")
        or geo_details.get("fraud_ring_proximity_risk")
        or geo_details.get("is_near_fraud_seed")
    )
    dormancy_break = _truthy(
        behavior_details.get("dormancy_break")
        or behavior_details.get("vel_dormancy_break")
        or velocity_details.get("dormancy_break")
        or velocity_details.get("dormancy_break_risk")
    )
    z_score = _number(
        behavior_details.get("z_score")
        or behavior_details.get("vel_z_score_amount")
        or velocity_details.get("z_score_amount")
    )

    if velocity_risk > 0.7 and unique_counterparties_1h >= 3:
        return "Rapid transfers"
    if geo_risk > 0.7 and (impossible_travel or shared_ip or fraud_ring_proximity):
        return "Fraud ring"
    if behavior_risk > 0.6 and dormancy_break and z_score > 3:
        return "Money laundering"
    return "Novel pattern"


def evaluate_synthesis(
    velocity_risk: float,
    velocity_confidence: float,
    geo_risk: float,
    geo_confidence: float,
    behavior_risk: float,
    behavior_confidence: float,
    txn_type: str,
    *,
    velocity_breakdown: Breakdown | None = None,
    geo_breakdown: Breakdown | None = None,
    behavior_breakdown: Breakdown | None = None,
) -> dict[str, Any]:
    """Apply two-layer weights, confidence-weighted fusion, and disagreement adjustment."""
    layer1 = get_layer1_weights(txn_type)
    fraud_pattern = classify_fraud_pattern(
        velocity_risk,
        geo_risk,
        behavior_risk,
        velocity_breakdown=velocity_breakdown,
        geo_breakdown=geo_breakdown,
        behavior_breakdown=behavior_breakdown,
    )
    layer2 = LAYER2_WEIGHTS[fraud_pattern]
    blended = blend_weights(layer1, layer2)

    final_score = confidence_weighted_fusion(
        velocity_risk,
        velocity_confidence,
        geo_risk,
        geo_confidence,
        behavior_risk,
        behavior_confidence,
        blended,
    )
    disagreement_variance = statistics.pvariance([velocity_risk, geo_risk, behavior_risk])
    disagreement_adjusted = disagreement_variance > DISAGREEMENT_VARIANCE_THRESHOLD
    if disagreement_adjusted:
        final_score += DISAGREEMENT_SCORE_BUMP
    final_score = _clamp01(final_score)

    return {
        "final_score": round(final_score, 4),
        "fraud_pattern": fraud_pattern,
        "weights_layer1": _round_weights(layer1),
        "weights_layer2": _round_weights(layer2),
        "weights_blended": _round_weights(blended),
        "agent_scores": {
            "velocity": {"risk": velocity_risk, "confidence": velocity_confidence},
            "geo": {"risk": geo_risk, "confidence": geo_confidence},
            "behavior": {"risk": behavior_risk, "confidence": behavior_confidence},
        },
        "disagreement_variance": round(disagreement_variance, 4),
        "disagreement_adjusted": disagreement_adjusted,
    }


def get_layer1_weights(txn_type: str) -> Weights:
    """Return transaction-type weights, falling back to a balanced default."""
    return dict(LAYER1_WEIGHTS.get(str(txn_type).upper(), DEFAULT_LAYER1_WEIGHTS))


def blend_weights(layer1: Weights, layer2: Weights) -> Weights:
    """Equation 1: 50/50 blend of transaction-type and fraud-pattern weights."""
    return {
        "velocity": 0.5 * layer1["velocity"] + 0.5 * layer2["velocity"],
        "geo": 0.5 * layer1["geo"] + 0.5 * layer2["geo"],
        "behavior": 0.5 * layer1["behavior"] + 0.5 * layer2["behavior"],
    }


def confidence_weighted_fusion(
    velocity_risk: float,
    velocity_confidence: float,
    geo_risk: float,
    geo_confidence: float,
    behavior_risk: float,
    behavior_confidence: float,
    weights: Weights,
) -> float:
    """Equation 2: weighted score with agent confidence in numerator and denominator."""
    numerator = (
        weights["velocity"] * velocity_confidence * velocity_risk
        + weights["geo"] * geo_confidence * geo_risk
        + weights["behavior"] * behavior_confidence * behavior_risk
    )
    denominator = (
        weights["velocity"] * velocity_confidence
        + weights["geo"] * geo_confidence
        + weights["behavior"] * behavior_confidence
    )
    if denominator <= 0:
        return 0.0
    return _clamp01(numerator / denominator)


def _round_weights(weights: Weights) -> Weights:
    return {name: round(value, 4) for name, value in weights.items()}


def _number(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _number(value) > 0


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
