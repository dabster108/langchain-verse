from fastapi import APIRouter


router = APIRouter(prefix="/evaluate", tags=["evaluate"])


@router.get("/ping")
async def evaluate_ping() -> dict[str, str]:
    return {"service": "api-gateway", "evaluate": "ready"}
