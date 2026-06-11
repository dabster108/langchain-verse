from fastapi import FastAPI

from app.routers.evaluate import router as evaluate_router
from app.routers.health import router as health_router

app = FastAPI(
    title="Decision & OTP Service",
    version="0.1.0",
    description=(
        "PASS/OTP/BLOCK threshold logic and dual-path OTP interlock "
        "(Sparrow SMS + email, 3-minute window)."
    ),
)

app.include_router(health_router)
app.include_router(evaluate_router)
