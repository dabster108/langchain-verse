from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any


TAU_LOW = 0.30
TAU_HIGH = 0.70


def evaluate_decision(final_score: float) -> dict[str, Any]:
    """Map the synthesis score to PASS, OTP, or BLOCK and create OTP metadata if needed."""
    if final_score < TAU_LOW:
        return {
            "verdict": "PASS",
            "otp_session_id": None,
            "otp_channels": [],
            "otp_expires_at": None,
        }
    if final_score < TAU_HIGH:
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=3)
        return {
            "verdict": "OTP",
            "otp_session_id": f"OTP-{secrets.token_hex(3).upper()}",
            "otp_channels": ["SMS", "EMAIL"],
            "otp_expires_at": expires_at.isoformat().replace("+00:00", "Z"),
        }
    return {
        "verdict": "BLOCK",
        "otp_session_id": None,
        "otp_channels": [],
        "otp_expires_at": None,
    }
