from __future__ import annotations

from typing import Any

from sqlalchemy import text


class TransactionNotFoundError(Exception):
    """Raised when transaction behavior data is unavailable."""


def evaluate_behavior(txn_id: str, account_id: str, db_connection) -> dict[str, Any]:
    """Score behavior using model-ready table features and conservative heuristics."""
    row = _fetch_behavior_row(txn_id, account_id, db_connection)
    if row is None:
        raise TransactionNotFoundError("Transaction not found")

    z_score = _number(row.get("z_score_amount"))
    amount = _number(row.get("amount_npr"))
    avg_amount = _number(row.get("avg_monthly_txn_value_npr"))
    amount_ratio = amount / avg_amount if avg_amount else 0.0
    dormancy_break = _truthy(row.get("dormancy_break")) or _truthy(row.get("is_dormant"))

    risk_score = 0.0
    if z_score > 5:
        risk_score += 0.35
    elif z_score > 3:
        risk_score += 0.25
    if amount_ratio > 2:
        risk_score += 0.20
    if dormancy_break and z_score > 3:
        risk_score += 0.25
    if _truthy(row.get("new_counterparty_flag")):
        risk_score += 0.15
    risk_score = min(risk_score, 1.0)

    confidence = 0.85 if _number(row.get("avg_monthly_txn_count")) >= 50 else 0.75
    if _truthy(row.get("is_dormant")):
        confidence -= 0.10

    return {
        "txn_id": txn_id,
        "risk_score": round(risk_score, 4),
        "confidence": round(max(confidence, 0.0), 2),
        "model_scores": {
            "xgboost": round(risk_score, 4),
            "isolation_forest": round(min(amount_ratio / 5, 1.0), 4),
            "lstm": None,
        },
        "models_used": ["heuristic_behavior"],
        "shap_explanation": [],
        "breakdown": {
            "z_score": z_score,
            "vel_z_score_amount": z_score,
            "dormancy_break": dormancy_break,
            "amount_ratio": amount_ratio,
        },
    }


def _fetch_behavior_row(txn_id: str, account_id: str, db_connection) -> dict[str, Any] | None:
    result = db_connection.execute(
        text(
            """
            SELECT
                t.txn_id,
                COALESCE(t.account_id, :account_id) AS account_id,
                t.amount_npr,
                c.avg_monthly_txn_value_npr,
                c.avg_monthly_txn_count,
                c.is_dormant,
                v.z_score_amount,
                v.dormancy_break,
                v.new_counterparty_flag
            FROM transactions t
            LEFT JOIN customers c ON c.account_id = t.account_id
            LEFT JOIN velocity_snapshots v ON v.txn_id = t.txn_id
            WHERE t.txn_id = :txn_id
              AND (:account_id IS NULL OR t.account_id = :account_id)
            """
        ),
        {"txn_id": txn_id, "account_id": account_id},
    )
    return _row_to_dict(result.fetchone())


def _row_to_dict(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    if isinstance(row, dict):
        return row
    if hasattr(row, "_mapping"):
        return dict(row._mapping)
    return dict(row)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _number(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)
