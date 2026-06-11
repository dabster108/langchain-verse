from fastapi import FastAPI

from app.routers.evaluate import router as evaluate_router
from app.routers.health import router as health_router

app = FastAPI(
    title="Synthesis Agent",
    version="0.1.0",
    description=(
        "Two-layer dynamic weight blending and confidence-weighted "
        "score synthesis across Velocity, Geo, and Behavior agents."
    ),
)

app.include_router(health_router)
app.include_router(evaluate_router)
