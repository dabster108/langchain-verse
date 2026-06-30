from pydantic import BaseModel, Field


class AgentRiskResponse(BaseModel):
    transaction_id: str
    agent_name: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasons: list[str] = []
