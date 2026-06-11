import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_decision_thresholds() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        pass_resp = await client.post(
            "/evaluate/decision",
            json={"transaction_id": "txn-005", "final_score": 0.10},
        )
        otp_resp = await client.post(
            "/evaluate/decision",
            json={"transaction_id": "txn-006", "final_score": 0.50},
        )
        block_resp = await client.post(
            "/evaluate/decision",
            json={"transaction_id": "txn-007", "final_score": 0.90},
        )
    assert pass_resp.json()["decision"] == "PASS"
    assert otp_resp.json()["decision"] == "OTP"
    assert block_resp.json()["decision"] == "BLOCK"


@pytest.mark.asyncio
async def test_otp_dual_path_initiate() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/evaluate/otp/initiate",
            json={
                "transaction_id": "txn-otp-1",
                "user_id": "user-1",
                "phone": "+9779800000000",
                "email": "user@example.com",
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["sms_status"] == "pending"
    assert body["email_status"] == "pending"
