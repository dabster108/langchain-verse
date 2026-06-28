from __future__ import annotations

from typing import Any

from sqlalchemy import text


class TransactionNotFoundError(Exception):
    """Raised when geo data is unavailable for a transaction."""


def evaluate_geo(txn_id: str, account_id: str, db_connection, neo4j_driver=None) -> dict[str, Any]:
    """Score geographic, device, and graph fraud signals."""
    row = _fetch_geo_row(txn_id, account_id, db_connection)
    if row is None:
        raise TransactionNotFoundError("Transaction not found")

    breakdown = {
        "impossible_travel_risk": 0.50 if _truthy(row.get("impossible_travel")) else 0.0,
        "vpn_tor_risk": 0.30 if _truthy(row.get("is_tor")) else 0.20 if _truthy(row.get("is_vpn")) else 0.0,
        "datacenter_risk": 0.15 if _truthy(row.get("is_datacenter")) else 0.0,
        "shared_ip_risk": 0.0,
        "fraud_ring_proximity_risk": 0.0,
        "impossible_travel": _truthy(row.get("impossible_travel")),
        "shared_ip": False,
        "fraud_ring_proximity": False,
    }
    fraud_ring_details = {
        "is_near_fraud_seed": False,
        "nearest_fraud_node_distance_hops": None,
        "nearest_fraud_node_id": None,
    }

    if neo4j_driver is not None:
        graph = _graph_checks(account_id, neo4j_driver)
        breakdown["shared_ip_risk"] = min(graph["shared_account_count"] * 0.20, 0.20)
        breakdown["shared_ip"] = graph["shared_account_count"] > 0
        breakdown["fraud_ring_proximity_risk"] = _fraud_ring_risk(graph["fraud_distance"])
        breakdown["fraud_ring_proximity"] = breakdown["fraud_ring_proximity_risk"] > 0
        fraud_ring_details = {
            "is_near_fraud_seed": graph["fraud_distance"] in {1, 2, 3},
            "nearest_fraud_node_distance_hops": graph["fraud_distance"],
            "nearest_fraud_node_id": graph["fraud_node"],
        }

    risk_score = min(
        breakdown["impossible_travel_risk"]
        + breakdown["vpn_tor_risk"]
        + breakdown["datacenter_risk"]
        + breakdown["shared_ip_risk"]
        + breakdown["fraud_ring_proximity_risk"],
        1.0,
    )
    confidence = 0.98 if breakdown["impossible_travel"] else 0.95
    if neo4j_driver is None:
        confidence = min(confidence, 0.75)

    return {
        "txn_id": txn_id,
        "risk_score": round(risk_score, 4),
        "confidence": confidence,
        "breakdown": breakdown,
        "fraud_ring_details": fraud_ring_details,
    }


def _fetch_geo_row(txn_id: str, account_id: str, db_connection) -> dict[str, Any] | None:
    result = db_connection.execute(
        text(
            """
            SELECT
                COALESCE(g.account_id, t.account_id, :account_id) AS account_id,
                g.impossible_travel,
                g.is_vpn,
                g.is_tor,
                g.is_datacenter
            FROM geo_events g
            LEFT JOIN transactions t ON t.txn_id = g.txn_id
            WHERE g.txn_id = :txn_id
            """
        ),
        {"txn_id": txn_id, "account_id": account_id},
    )
    return _row_to_dict(result.fetchone())


def _graph_checks(account_id: str, neo4j_driver) -> dict[str, Any]:
    try:
        with neo4j_driver.session() as session:
            shared_record = session.run(
                """
                MATCH (a:Account {id: $account_id})-[*1..1]-(other:Account)
                WHERE other.id <> $account_id
                RETURN count(distinct other) as shared_account_count
                """,
                {"account_id": account_id},
            ).single()
            fraud_record = session.run(
                """
                MATCH (a:Account {id: $account_id}), (fraud:Account {is_fraud_seed: true})
                MATCH p = shortestPath((a)-[*1..4]-(fraud))
                RETURN fraud.id as fraud_node, length(p) as distance
                ORDER BY distance ASC LIMIT 1
                """,
                {"account_id": account_id},
            ).single()
    except Exception:
        return {"shared_account_count": 0, "fraud_distance": None, "fraud_node": None}

    return {
        "shared_account_count": int(_record_get(shared_record, "shared_account_count") or 0),
        "fraud_distance": _record_get(fraud_record, "distance"),
        "fraud_node": _record_get(fraud_record, "fraud_node"),
    }


def _fraud_ring_risk(distance: int | None) -> float:
    if distance == 1:
        return 0.35
    if distance == 2:
        return 0.25
    if distance == 3:
        return 0.10
    return 0.0


def _record_get(record: Any, key: str) -> Any:
    if record is None:
        return None
    if isinstance(record, dict):
        return record.get(key)
    return record[key]


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
