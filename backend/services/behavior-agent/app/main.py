import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.model_loader import DEFAULT_MODELS_DIR, load_models
from app.routers.evaluate import router as evaluate_router
from shared.constants.service_names import BEHAVIOR_AGENT
from shared.routers.health import health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = Path(os.environ.get("MODELS_DIR", str(DEFAULT_MODELS_DIR)))
    app.state.models = load_models(models_dir)
    yield


app = FastAPI(
    title="Behavior Agent",
    version="0.1.0",
    description="ML behavior fraud risk microservice with SHAP explainability.",
    lifespan=lifespan,
)

app.include_router(health_router(BEHAVIOR_AGENT))
app.include_router(evaluate_router)
