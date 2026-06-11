import pytest

from app.main import app
from app.model_loader import load_models


@pytest.fixture(autouse=True)
def _load_models() -> None:
    app.state.models = load_models()
