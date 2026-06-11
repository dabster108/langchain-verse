import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_evaluate_velocity_risk() -> None:
    transport = ASGITransport(app=app)
    payload = {
        "transaction_id": "txn-002",
        "txn_count_1h": 8,
        "txn_count_24h": 12,
        "amount": 75000.0,
    }
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/evaluate/risk", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["risk_score"] > 0.0
    assert "high hourly transaction count" in body["reasons"]
