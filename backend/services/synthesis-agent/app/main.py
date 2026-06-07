from fastapi import FastAPI

from app.routers.health import router as health_router


app = FastAPI(
    title="Synthesis Agent",
    version="0.1.0",
    description="Future service for combining agent risk scores.",
)

app.include_router(health_router)
