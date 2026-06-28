from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool

from app.agents import behavior_agent, decision_agent, geo_agent, velocity_agent
from app.orchestrator import evaluate_transaction

try:
    from neo4j import GraphDatabase
except ImportError:  # Neo4j is optional for local Postgres-only development.
    GraphDatabase = None


logger = logging.getLogger("fraud-detection-pipeline")
logging.basicConfig(level=logging.INFO)

BACKEND_DIR = Path(__file__).resolve().parents[3]
load_dotenv(BACKEND_DIR / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL")
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

engine = (
    create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )
    if DATABASE_URL
    else None
)

neo4j_driver = (
    GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    if GraphDatabase is not None and NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD
    else None
)

app = FastAPI(
    title="Unified Fraud Detection Pipeline",
    version="0.1.0",
    description="Single FastAPI service with Velocity, Geo, Behavior, Synthesis, and Decision agents.",
)


class EvaluateRequest(BaseModel):
    txn_id: str = Field(..., min_length=1)
    account_id: str = Field(..., min_length=1)


class VerifyOTPRequest(BaseModel):
    otp_session_id: str = Field(..., min_length=1)
    sms_otp: str | None = Field(default=None, min_length=6, max_length=6)
    email_otp: str | None = Field(default=None, min_length=6, max_length=6)


@app.on_event("startup")
def startup_check_connections() -> None:
    app.state.db_available = False
    app.state.neo4j_available = False

    if engine is None:
        logger.error("DATABASE_URL is not configured")
    else:
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            app.state.db_available = True
        except Exception:
            logger.error("Postgres connection failed:\n%s", traceback.format_exc())

    if neo4j_driver is not None:
        try:
            neo4j_driver.verify_connectivity()
            app.state.neo4j_available = True
        except Exception:
            logger.warning("Neo4j unavailable; Geo Agent will run Postgres-only:\n%s", traceback.format_exc())


@app.on_event("shutdown")
def shutdown_connections() -> None:
    if neo4j_driver is not None:
        neo4j_driver.close()


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "fraud-detection-pipeline"}


@app.post("/evaluate")
def evaluate(body: EvaluateRequest) -> dict:
    if engine is None or not getattr(app.state, "db_available", False):
        raise HTTPException(status_code=503, detail="Database unavailable")

    graph = neo4j_driver if getattr(app.state, "neo4j_available", False) else None
    try:
        with engine.connect() as connection:
            return evaluate_transaction(body.txn_id, body.account_id, connection, graph)
    except (
        velocity_agent.TransactionNotFoundError,
        geo_agent.TransactionNotFoundError,
        behavior_agent.TransactionNotFoundError,
    ):
        raise HTTPException(status_code=404, detail="Transaction not found") from None
    except decision_agent.InvalidScoreError:
        raise HTTPException(status_code=400, detail="Invalid score") from None
    except SQLAlchemyError:
        logger.error("Database error while evaluating txn_id=%s:\n%s", body.txn_id, traceback.format_exc())
        app.state.db_available = False
        raise HTTPException(status_code=503, detail="Database unavailable") from None
    except Exception:
        logger.error("Unexpected error while evaluating txn_id=%s:\n%s", body.txn_id, traceback.format_exc())
        raise


@app.post("/verify-otp")
def verify_otp(body: VerifyOTPRequest) -> dict:
    if engine is None or not getattr(app.state, "db_available", False):
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        with engine.connect() as connection:
            return decision_agent.verify_otp_session(
                body.otp_session_id,
                body.sms_otp,
                body.email_otp,
                connection,
            )
    except decision_agent.OTPExpiredError:
        raise HTTPException(status_code=410, detail="OTP expired, request new verification") from None
    except decision_agent.OTPNotFoundError:
        raise HTTPException(status_code=404, detail="OTP session not found") from None
    except SQLAlchemyError:
        logger.error("Database error while verifying otp_session_id=%s:\n%s", body.otp_session_id, traceback.format_exc())
        raise HTTPException(status_code=503, detail="Database unavailable") from None


@app.get("/otp-status/{otp_session_id}")
def otp_status(otp_session_id: str) -> dict:
    if engine is None or not getattr(app.state, "db_available", False):
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        with engine.connect() as connection:
            return decision_agent.get_otp_status(otp_session_id, connection)
    except decision_agent.OTPNotFoundError:
        raise HTTPException(status_code=404, detail="OTP session not found") from None
    except SQLAlchemyError:
        logger.error("Database error while checking otp_session_id=%s:\n%s", otp_session_id, traceback.format_exc())
        raise HTTPException(status_code=503, detail="Database unavailable") from None


@app.post("/otp-resend/{otp_session_id}")
def otp_resend(otp_session_id: str) -> dict:
    if engine is None or not getattr(app.state, "db_available", False):
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        with engine.connect() as connection:
            return decision_agent.resend_otp_session(otp_session_id, connection)
    except decision_agent.OTPNotFoundError:
        raise HTTPException(status_code=404, detail="OTP session not found") from None
    except SQLAlchemyError:
        logger.error("Database error while resending otp_session_id=%s:\n%s", otp_session_id, traceback.format_exc())
        raise HTTPException(status_code=503, detail="Database unavailable") from None
