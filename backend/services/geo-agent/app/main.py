from fastapi import FastAPI

from app.routers.evaluate import router as evaluate_router
from shared.constants.service_names import GEO_AGENT
from shared.routers.health import health_router

app = FastAPI(
    title="Geo Agent",
    version="0.1.0",
    description="Location and graph-context fraud risk microservice.",
)

app.include_router(health_router(GEO_AGENT))
app.include_router(evaluate_router)
