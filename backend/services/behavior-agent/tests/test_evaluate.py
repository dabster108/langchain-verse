import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.model_loader import BEHAVIOR_FEATURE_NAMES


@pytest.mark.asyncio
async def test_evaluate_behavior_risk_with_shap() -> None:
    transport = ASGITransport(app=app)
    payload = {
        "transaction_id": "txn-003",
        "features": [1000.0, 14.0, 2.0, 3.0, 10.0, 500.0, 0.1, 30.0, 0.0, 5.0],
    }
    assert len(payload["features"]) == len(BEHAVIOR_FEATURE_NAMES)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/evaluate/risk", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["shap"] is not None
    assert len(body["shap"]["feature_names"]) == len(BEHAVIOR_FEATURE_NAMES)
