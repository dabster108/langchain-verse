import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_evaluate_geo_risk() -> None:
    transport = ASGITransport(app=app)
    payload = {
        "transaction_id": "txn-001",
        "distance_from_home_km": 120.0,
        "is_new_location": True,
        "ring_proximity_score": 0.2,
    }
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/evaluate/risk", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "txn-001"
    assert 0.0 <= body["risk_score"] <= 1.0
