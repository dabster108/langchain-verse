from fastapi import FastAPI

from app.routers.health import router as health_router


app = FastAPI(
    title="Fraud Detection API Gateway",
    version="0.1.0",
    description="Future public entrypoint for transaction risk requests.",
)

app.include_router(health_router)
