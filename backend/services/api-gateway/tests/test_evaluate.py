import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_evaluate_ping() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/evaluate/ping")
    assert response.status_code == 200
    assert response.json()["evaluate"] == "ready"
