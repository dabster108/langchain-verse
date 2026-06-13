from fastapi import FastAPI

from app.routers.evaluate import router as evaluate_router
from shared.constants.service_names import VELOCITY_AGENT
from shared.routers.health import health_router

app = FastAPI(
    title="Velocity Agent",
    version="0.1.0",
    description="Transaction velocity fraud risk microservice.",
)

app.include_router(health_router(VELOCITY_AGENT))
app.include_router(evaluate_router)
