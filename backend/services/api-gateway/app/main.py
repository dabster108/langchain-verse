from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from shared.constants.service_names import API_GATEWAY
from shared.routers.health import health_router

VELOCITY_AGENT_URL = os.environ.get("VELOCITY_AGENT_URL", "http://velocity-agent:8001")
GEO_AGENT_URL = os.environ.get("GEO_AGENT_URL", "http://geo-agent:8002")

app = FastAPI(
    title="Fraud Detection API Gateway",
    version="0.1.0",
    description="Public entrypoint for transaction risk requests.",
)

app.include_router(health_router(API_GATEWAY))


class EvaluateRequest(BaseModel):
    txn_id: str = Field(..., min_length=1)
    account_id: str = Field(..., min_length=1)


@app.post("/evaluate/velocity")
def evaluate_velocity(body: EvaluateRequest) -> dict[str, Any]:
    return _post_agent(f"{VELOCITY_AGENT_URL}/evaluate", body.model_dump())


@app.post("/evaluate/geo")
def evaluate_geo(body: EvaluateRequest) -> dict[str, Any]:
    return _post_agent(f"{GEO_AGENT_URL}/evaluate", body.model_dump())


@app.post("/evaluate/both")
def evaluate_both(body: EvaluateRequest) -> dict[str, Any]:
    started = time.perf_counter()
    payload = body.model_dump()
    velocity = _post_agent(f"{VELOCITY_AGENT_URL}/evaluate", payload)
    geo = _post_agent(f"{GEO_AGENT_URL}/evaluate", payload)
    latency_ms = int((time.perf_counter() - started) * 1000)

    return {
        "txn_id": body.txn_id,
        "account_id": body.account_id,
        "agents": {
            "velocity": velocity,
            "geo": geo,
        },
        "latency_ms": latency_ms,
    }


def _post_agent(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8") or exc.reason
        raise HTTPException(status_code=exc.code, detail=detail) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=503, detail=f"Agent unavailable: {exc.reason}") from exc
