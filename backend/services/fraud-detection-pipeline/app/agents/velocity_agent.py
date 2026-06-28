from __future__ import annotations

from typing import Any

from sqlalchemy import text


class TransactionNotFoundError(Exception):
    """Raised when velocity data is unavailable for a transaction."""


def evaluate_velocity(txn_id: str, account_id: str, db_connection) -> dict[str, Any]:
    """Score short-window transfer velocity using Postgres snapshots."""
    row = _fetch_velocity_row(txn_id, account_id, db_connection)
    if row is None:
        raise TransactionNotFoundError("Transaction not found")

    z_score = _number(row.get("z_score_amount"))
    txn_count_1m = int(_number(row.get("txn_count_1m")))
    txn_count_1h = int(_number(row.get("txn_count_1h")))
    unique_counterparties_1h = int(_number(row.get("unique_counterparties_1h")))

    breakdown = {
        "z_score_risk": 0.40 if z_score > 5 else 0.30 if z_score > 3.5 else 0.0,
        "txn_count_risk": min((0.25 if txn_count_1m >= 3 else 0.0) + (0.15 if txn_count_1h >= 8 else 0.0), 0.30),
        "new_counterparty_risk": 0.20 if _truthy(row.get("new_counterparty_flag")) else 0.0,
        "dormancy_break_risk": 0.25 if _truthy(row.get("dormancy_break")) and z_score > 3 else 0.0,
        "unique_recipients_risk": 0.15 if unique_counterparties_1h >= 3 else 0.0,
        "unique_counterparties_1h": unique_counterparties_1h,
        "z_score_amount": z_score,
        "dormancy_break": _truthy(row.get("dormancy_break")),
    }
    risk_score = min(
        breakdown["z_score_risk"]
        + breakdown["txn_count_risk"]
        + breakdown["new_counterparty_risk"]
        + breakdown["dormancy_break_risk"]
        + breakdown["unique_recipients_risk"],
        1.0,
    )
    confidence = 0.95 if row.get("account_id") else 0.75

    return {
        "txn_id": txn_id,
        "risk_score": round(risk_score, 4),
        "confidence": confidence,
        "breakdown": breakdown,
    }


def _fetch_velocity_row(txn_id: str, account_id: str, db_connection) -> dict[str, Any] | None:
    result = db_connection.execute(
        text(
            """
            SELECT
                COALESCE(v.account_id, t.account_id, :account_id) AS account_id,
                v.z_score_amount,
                v.txn_count_1m,
                v.txn_count_1h,
                v.unique_counterparties_1h,
                v.new_counterparty_flag,
                v.dormancy_break
            FROM velocity_snapshots v
            LEFT JOIN transactions t ON t.txn_id = v.txn_id
            WHERE v.txn_id = :txn_id
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
