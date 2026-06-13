from fastapi import FastAPI

from shared.constants.service_names import API_GATEWAY
from shared.routers.health import health_router

app = FastAPI(
    title="Fraud Detection API Gateway",
    version="0.1.0",
    description="Public entrypoint for transaction risk requests.",
)

app.include_router(health_router(API_GATEWAY))
