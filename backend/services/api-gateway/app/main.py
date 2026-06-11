from fastapi import FastAPI

from app.routers.evaluate import router as evaluate_router
from app.routers.health import router as health_router

app = FastAPI(
    title="Fraud Detection API Gateway",
    version="0.1.0",
    description="Future public entrypoint for transaction risk requests.",
)

app.include_router(health_router)
app.include_router(evaluate_router)
