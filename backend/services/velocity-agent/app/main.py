from fastapi import FastAPI

from app.routers.health import router as health_router


app = FastAPI(
    title="Velocity Agent",
    version="0.1.0",
    description="Transaction velocity fraud risk microservice.",
)

app.include_router(health_router)
