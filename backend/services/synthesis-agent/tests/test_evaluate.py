import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_synthesise_risk() -> None:
    transport = ASGITransport(app=app)
    payload = {
        "transaction_id": "txn-004",
        "transaction_type": "p2p_transfer",
        "velocity": {"risk_score": 0.7, "confidence": 0.9, "latency_ms": 12},
        "geo": {"risk_score": 0.3, "confidence": 0.8, "latency_ms": 20},
        "behavior": {"risk_score": 0.5, "confidence": 0.85, "latency_ms": 45},
    }
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/evaluate/synthesise", json=payload)
    assert response.status_code == 200
    body = response.json()
    result = body["result"]
    assert 0.0 <= result["final_score"] <= 1.0
    assert result["fraud_pattern"] in (
        "rapid_transfers",
        "fraud_ring",
        "money_laundering",
        "novel_pattern",
    )
