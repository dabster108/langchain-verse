import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.model_loader import (
    DEFAULT_FEATURE_TABLE,
    DEFAULT_MODELS_DIR,
    load_feature_table_index,
    load_models,
)
from app.routers.evaluate import router as evaluate_router
from shared.constants.service_names import BEHAVIOR_AGENT
from shared.routers.health import health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    models_dir = Path(os.environ.get("MODELS_DIR", str(DEFAULT_MODELS_DIR)))
    feature_table = Path(os.environ.get("FEATURE_TABLE_PATH", str(DEFAULT_FEATURE_TABLE)))

    app.state.models = load_models(models_dir)
    app.state.feature_index = load_feature_table_index(feature_table)
    yield


app = FastAPI(
    title="Behavior Agent",
    version="0.1.0",
    description="ML behavior fraud risk microservice with SHAP explainability.",
    lifespan=lifespan,
)

app.include_router(health_router(BEHAVIOR_AGENT))
app.include_router(evaluate_router)
