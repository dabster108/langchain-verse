from __future__ import annotations

import json
import logging
import time
from typing import Any

from sqlalchemy import text

from app.agents import behavior_agent, decision_agent, geo_agent, synthesis_agent, velocity_agent


logger = logging.getLogger("fraud-detection-pipeline")


def evaluate_transaction(
    txn_id: str,
    account_id: str,
    db_connection,
    neo4j_driver=None,
) -> dict[str, Any]:
    """Run the complete fraud detection pipeline for one transaction."""
    started = time.perf_counter()
    txn_type = _fetch_txn_type(txn_id, account_id, db_connection)

    velocity = velocity_agent.evaluate_velocity(txn_id, account_id, db_connection)
    geo = geo_agent.evaluate_geo(txn_id, account_id, db_connection, neo4j_driver)
    behavior = behavior_agent.evaluate_behavior(txn_id, account_id, db_connection)

    synthesis = synthesis_agent.evaluate_synthesis(
        velocity["risk_score"],
        velocity["confidence"],
        geo["risk_score"],
        geo["confidence"],
        behavior["risk_score"],
        behavior["confidence"],
        txn_type,
        velocity_breakdown=velocity.get("breakdown"),
        geo_breakdown={**geo.get("breakdown", {}), **geo.get("fraud_ring_details", {})},
        behavior_breakdown=behavior.get("breakdown"),
    )
    decision = decision_agent.evaluate_decision(
        final_score=synthesis["final_score"],
        fraud_pattern=synthesis["fraud_pattern"],
        account_id=account_id,
        txn_id=txn_id,
        db_connection=db_connection,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    result = {
        "txn_id": txn_id,
        "final_verdict": decision["verdict"],
        "final_score": synthesis["final_score"],
        "fraud_pattern": synthesis["fraud_pattern"],
        "latency_ms": latency_ms,
        "agent_outputs": {
            "velocity": velocity,
            "geo": geo,
            "behavior": behavior,
        },
        "synthesis": {
            "weights_layer1": synthesis["weights_layer1"],
            "weights_layer2": synthesis["weights_layer2"],
            "weights_blended": synthesis["weights_blended"],
            "disagreement_variance": synthesis["disagreement_variance"],
            "disagreement_adjusted": synthesis["disagreement_adjusted"],
        },
        "decision": decision,
    }
    _write_audit_log(txn_id, account_id, result, db_connection)
    return result


def _fetch_txn_type(txn_id: str, account_id: str, db_connection) -> str:
    result = db_connection.execute(
        text(
            """
            SELECT txn_type
            FROM transactions
            WHERE txn_id = :txn_id
              AND (:account_id IS NULL OR account_id = :account_id)
            """
        ),
        {"txn_id": txn_id, "account_id": account_id},
    )
    row = _row_to_dict(result.fetchone())
    if row is None:
        raise velocity_agent.TransactionNotFoundError("Transaction not found")
    return _normalise_txn_type(row.get("txn_type"))


def _write_audit_log(txn_id: str, account_id: str, result: dict[str, Any], db_connection) -> None:
    try:
        db_connection.execute(
            text(
                """
                INSERT INTO fraud_audit_log (txn_id, account_id, final_score, final_verdict, payload)
                VALUES (:txn_id, :account_id, :final_score, :final_verdict, CAST(:payload AS JSONB))
                """
            ),
            {
                "txn_id": txn_id,
                "account_id": account_id,
                "final_score": result["final_score"],
                "final_verdict": result["final_verdict"],
                "payload": json.dumps(result, default=str),
            },
        )
        if hasattr(db_connection, "commit"):
            db_connection.commit()
    except Exception:
        logger.warning("Skipping audit log write; fraud_audit_log table may be unavailable", exc_info=True)


def _normalise_txn_type(txn_type: Any) -> str:
    mapping = {
        "p2p_transfer": "ESEWA_P2P",
        "P2P_TRANSFER": "ESEWA_P2P",
        "merchant_payment": "KHALTI_QR",
        "MERCHANT_PAYMENT": "KHALTI_QR",
        "atm_withdrawal": "ATM_WITHDRAWAL",
        "ATM_WITHDRAWAL": "ATM_WITHDRAWAL",
        "bill_payment": "UTILITY_BILL",
        "BILL_PAYMENT": "UTILITY_BILL",
    }
    return mapping.get(str(txn_type), str(txn_type).upper())


def _row_to_dict(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    if isinstance(row, dict):
        return row
    if hasattr(row, "_mapping"):
        return dict(row._mapping)
    return dict(row)
