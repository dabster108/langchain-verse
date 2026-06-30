from fastapi import FastAPI

from app.routers.health import router as health_router


app = FastAPI(
    title="Geo Agent",
    version="0.1.0",
    description="Location and graph-context fraud risk microservice.",
)

app.include_router(health_router)
