from __future__ import annotations

import logging
import math
import secrets
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import text


logger = logging.getLogger("fraud-detection-pipeline.decision")

THRESHOLD_PASS = 0.30
THRESHOLD_OTP = 0.70
OTP_EXPIRY_SECONDS = 180
MAX_OTP_ATTEMPTS = 3


class InvalidScoreError(ValueError):
    """Raised when synthesis returns a non-finite score."""


class OTPExpiredError(Exception):
    """Raised when an OTP session has expired."""


class OTPNotFoundError(Exception):
    """Raised when an OTP session id is unknown."""


def evaluate_decision(
    final_score: float,
    fraud_pattern: str,
    account_id: str,
    txn_id: str,
    db_connection,
) -> dict[str, Any]:
    """Convert a synthesis score to PASS, OTP, or BLOCK and trigger side effects."""
    if final_score is None or not math.isfinite(float(final_score)):
        raise InvalidScoreError("Invalid score")

    final_score = float(final_score)
    if final_score < THRESHOLD_PASS:
        verdict = "PASS"
    elif final_score <= THRESHOLD_OTP:
        verdict = "OTP"
    else:
        verdict = "BLOCK"

    logger.info(
        "Decision verdict=%s final_score=%s account_id=%s txn_id=%s",
        verdict,
        final_score,
        account_id,
        txn_id,
    )

    if verdict == "PASS":
        return {
            "verdict": "PASS",
            "final_score": round(final_score, 4),
            "threshold_used": "PASS (< 0.30)",
            "message": "Transaction approved, no additional verification required",
        }

    if verdict == "OTP":
        return _create_otp_session(final_score, account_id, txn_id, db_connection)

    _flag_account_for_review(
        account_id,
        f"Transaction {txn_id} blocked due to fraud_pattern={fraud_pattern}",
        db_connection,
        flag_type="KYC_REVIEW_REQUIRED",
    )
    logger.warning(
        "Transaction BLOCKED for account_id=%s fraud_pattern=%s",
        account_id,
        fraud_pattern,
    )
    return {
        "verdict": "BLOCK",
        "final_score": round(final_score, 4),
        "threshold_used": "BLOCK (> 0.70)",
        "fraud_pattern": fraud_pattern,
        "account_status": "FLAGGED_FOR_REVIEW",
        "message": "Transaction blocked. Account flagged for manual KYC review.",
    }


def verify_otp_session(
    otp_session_id: str,
    sms_otp: str | None,
    email_otp: str | None,
    db_connection,
) -> dict[str, Any]:
    """Verify both OTP channels, auto-blocking expired or repeatedly failed sessions."""
    _ensure_decision_tables(db_connection)
    session = _fetch_otp_session(otp_session_id, db_connection)
    now = datetime.utcnow()

    if now > session["expires_at"]:
        _expire_session(otp_session_id, db_connection)
        raise OTPExpiredError("OTP expired, request new verification")

    channel_1_status = session.get("channel_1_status") or "PENDING"
    channel_2_status = session.get("channel_2_status") or "PENDING"
    attempt_count_ch1 = int(session.get("attempt_count_ch1") or 0)
    attempt_count_ch2 = int(session.get("attempt_count_ch2") or 0)

    if sms_otp is not None:
        if secrets.compare_digest(str(sms_otp), str(session.get("channel_1_otp"))):
            channel_1_status = "VERIFIED"
        else:
            channel_1_status = "FAILED"
            attempt_count_ch1 += 1

    if email_otp is not None:
        if secrets.compare_digest(str(email_otp), str(session.get("channel_2_otp"))):
            channel_2_status = "VERIFIED"
        else:
            channel_2_status = "FAILED"
            attempt_count_ch2 += 1

    sim_swap_suspected = channel_1_status == "FAILED" and channel_2_status == "VERIFIED"
    final_decision = None
    message = "OTP verification pending"

    if channel_1_status == "VERIFIED" and channel_2_status == "VERIFIED":
        final_decision = "ALLOWED"
        message = "Both OTP codes verified. Transaction approved."
    elif attempt_count_ch1 >= MAX_OTP_ATTEMPTS or attempt_count_ch2 >= MAX_OTP_ATTEMPTS:
        final_decision = "BLOCKED"
        message = "OTP verification failed too many times. Transaction blocked."
        _flag_account_for_review(
            session["account_id"],
            f"OTP failed 3+ times for otp_session_id={otp_session_id}",
            db_connection,
            flag_type="KYC_REVIEW_REQUIRED",
        )
    elif sim_swap_suspected:
        final_decision = "BLOCKED"
        message = "SMS OTP failed while email succeeded. SIM-swap suspected."
        _flag_account_for_review(
            session["account_id"],
            f"SIM-swap suspected for otp_session_id={otp_session_id}",
            db_connection,
            flag_type="SIM_SWAP_DETECTED",
        )

    _update_otp_verification(
        otp_session_id,
        channel_1_status,
        channel_2_status,
        attempt_count_ch1,
        attempt_count_ch2,
        final_decision,
        sim_swap_suspected,
        db_connection,
    )

    logger.info(
        "OTP verified for otp_session_id=%s channel_1=%s channel_2=%s",
        otp_session_id,
        channel_1_status,
        channel_2_status,
    )
    return {
        "otp_session_id": otp_session_id,
        "channel_1_status": channel_1_status,
        "channel_2_status": channel_2_status,
        "final_decision": final_decision,
        "sim_swap_suspected": sim_swap_suspected,
        "message": message,
    }


def get_otp_status(otp_session_id: str, db_connection) -> dict[str, Any]:
    """Return OTP channel statuses and remaining lifetime."""
    _ensure_decision_tables(db_connection)
    session = _fetch_otp_session(otp_session_id, db_connection)
    now = datetime.utcnow()
    seconds_left = max(int((session["expires_at"] - now).total_seconds()), 0)
    return {
        "otp_session_id": otp_session_id,
        "channel_1_status": session.get("channel_1_status"),
        "channel_2_status": session.get("channel_2_status"),
        "final_decision": session.get("final_decision"),
        "sim_swap_suspected": bool(session.get("sim_swap_suspected")),
        "expires_at": _iso_z(session["expires_at"]),
        "seconds_remaining": seconds_left,
    }


def resend_otp_session(otp_session_id: str, db_connection) -> dict[str, Any]:
    """Expire the old OTP session and issue a replacement session."""
    _ensure_decision_tables(db_connection)
    session = _fetch_otp_session(otp_session_id, db_connection)
    _expire_session(otp_session_id, db_connection)
    return _create_otp_session(
        final_score=0.0,
        account_id=session["account_id"],
        txn_id=session["txn_id"],
        db_connection=db_connection,
        message="OTP resent. Enter the new SMS and email codes.",
    )


def _create_otp_session(
    final_score: float,
    account_id: str,
    txn_id: str,
    db_connection,
    *,
    message: str = "Transaction requires verification. Enter OTP codes from SMS and email.",
) -> dict[str, Any]:
    _ensure_decision_tables(db_connection)
    contact = _fetch_customer_contact(account_id, db_connection)
    now = datetime.utcnow()
    expires_at = now + timedelta(seconds=OTP_EXPIRY_SECONDS)
    otp_session_id = f"OTP-{secrets.token_hex(3).upper()}"
    sms_code = _generate_otp_code()
    email_code = _generate_otp_code()

    try:
        logger.info(
            "OTP SMS sent to account_id=%s phone=%s",
            account_id,
            _last4(contact.get("phone")),
        )
        logger.info(
            "OTP EMAIL sent to account_id=%s email=%s",
            account_id,
            _mask_email(contact.get("email")),
        )
        logger.info("OTP SMS sent to account_id=%s expires_in_seconds=%s", account_id, OTP_EXPIRY_SECONDS)
    except Exception:
        logger.exception("Mock OTP sending failed for account_id=%s", account_id)

    db_connection.execute(
        text(
            """
            INSERT INTO otp_sessions (
                otp_session_id,
                txn_id,
                account_id,
                created_at,
                expires_at,
                channel_1_otp,
                channel_1_sent_at,
                channel_1_status,
                channel_2_otp,
                channel_2_sent_at,
                channel_2_status,
                final_decision,
                sim_swap_suspected,
                attempt_count_ch1,
                attempt_count_ch2
            )
            VALUES (
                :otp_session_id,
                :txn_id,
                :account_id,
                :created_at,
                :expires_at,
                :channel_1_otp,
                :channel_1_sent_at,
                'PENDING',
                :channel_2_otp,
                :channel_2_sent_at,
                'PENDING',
                NULL,
                false,
                0,
                0
            )
            """
        ),
        {
            "otp_session_id": otp_session_id,
            "txn_id": txn_id,
            "account_id": account_id,
            "created_at": now,
            "expires_at": expires_at,
            "channel_1_otp": sms_code,
            "channel_1_sent_at": now,
            "channel_2_otp": email_code,
            "channel_2_sent_at": now,
        },
    )
    _commit_if_supported(db_connection)

    return {
        "verdict": "OTP",
        "final_score": round(final_score, 4),
        "threshold_used": "OTP (0.30-0.70)",
        "otp_session_id": otp_session_id,
        "otp_channels": ["SMS", "EMAIL"],
        "otp_expires_in_seconds": OTP_EXPIRY_SECONDS,
        "otp_expires_at": _iso_z(expires_at),
        "message": message,
    }


def _ensure_decision_tables(db_connection) -> None:
    db_connection.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS otp_sessions (
                otp_session_id VARCHAR PRIMARY KEY,
                txn_id VARCHAR NOT NULL,
                account_id VARCHAR NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMP NOT NULL,
                channel_1_otp VARCHAR,
                channel_1_sent_at TIMESTAMP,
                channel_1_verified_at TIMESTAMP,
                channel_1_status VARCHAR DEFAULT 'PENDING',
                channel_2_otp VARCHAR,
                channel_2_sent_at TIMESTAMP,
                channel_2_verified_at TIMESTAMP,
                channel_2_status VARCHAR DEFAULT 'PENDING',
                final_decision VARCHAR,
                sim_swap_suspected BOOLEAN DEFAULT FALSE,
                attempt_count_ch1 INT DEFAULT 0,
                attempt_count_ch2 INT DEFAULT 0
            )
            """
        )
    )
    db_connection.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS account_flags (
                flag_id SERIAL PRIMARY KEY,
                account_id VARCHAR NOT NULL,
                flag_type VARCHAR,
                created_at TIMESTAMP DEFAULT NOW(),
                reason VARCHAR
            )
            """
        )
    )
    _commit_if_supported(db_connection)


def _fetch_customer_contact(account_id: str, db_connection) -> dict[str, Any]:
    try:
        result = db_connection.execute(
            text(
                """
                SELECT account_id, phone, email
                FROM customers
                WHERE account_id = :account_id
                """
            ),
            {"account_id": account_id},
        )
        row = _row_to_dict(result.fetchone())
        if row is None:
            logger.warning("Account account_id=%s not found in customers; continuing OTP flow", account_id)
            return {"phone": None, "email": None}
        return row
    except Exception:
        logger.warning("Customer contact lookup failed for account_id=%s; continuing OTP flow", account_id)
        return {"phone": None, "email": None}


def _fetch_otp_session(otp_session_id: str, db_connection) -> dict[str, Any]:
    result = db_connection.execute(
        text(
            """
            SELECT *
            FROM otp_sessions
            WHERE otp_session_id = :otp_session_id
            """
        ),
        {"otp_session_id": otp_session_id},
    )
    row = _row_to_dict(result.fetchone())
    if row is None:
        raise OTPNotFoundError("OTP session not found")
    return row


def _update_otp_verification(
    otp_session_id: str,
    channel_1_status: str,
    channel_2_status: str,
    attempt_count_ch1: int,
    attempt_count_ch2: int,
    final_decision: str | None,
    sim_swap_suspected: bool,
    db_connection,
) -> None:
    now = datetime.utcnow()
    db_connection.execute(
        text(
            """
            UPDATE otp_sessions
            SET
                channel_1_status = :channel_1_status,
                channel_1_verified_at = CASE WHEN :channel_1_status = 'VERIFIED' THEN :now ELSE channel_1_verified_at END,
                channel_2_status = :channel_2_status,
                channel_2_verified_at = CASE WHEN :channel_2_status = 'VERIFIED' THEN :now ELSE channel_2_verified_at END,
                attempt_count_ch1 = :attempt_count_ch1,
                attempt_count_ch2 = :attempt_count_ch2,
                final_decision = COALESCE(:final_decision, final_decision),
                sim_swap_suspected = :sim_swap_suspected
            WHERE otp_session_id = :otp_session_id
            """
        ),
        {
            "otp_session_id": otp_session_id,
            "channel_1_status": channel_1_status,
            "channel_2_status": channel_2_status,
            "attempt_count_ch1": attempt_count_ch1,
            "attempt_count_ch2": attempt_count_ch2,
            "final_decision": final_decision,
            "sim_swap_suspected": sim_swap_suspected,
            "now": now,
        },
    )
    _commit_if_supported(db_connection)


def _expire_session(otp_session_id: str, db_connection) -> None:
    db_connection.execute(
        text(
            """
            UPDATE otp_sessions
            SET channel_1_status = 'EXPIRED',
                channel_2_status = 'EXPIRED',
                final_decision = 'BLOCKED'
            WHERE otp_session_id = :otp_session_id
            """
        ),
        {"otp_session_id": otp_session_id},
    )
    _commit_if_supported(db_connection)


def _flag_account_for_review(
    account_id: str,
    reason: str,
    db_connection,
    *,
    flag_type: str,
) -> None:
    _ensure_decision_tables(db_connection)
    db_connection.execute(
        text(
            """
            INSERT INTO account_flags (account_id, flag_type, reason)
            VALUES (:account_id, :flag_type, :reason)
            """
        ),
        {"account_id": account_id, "flag_type": flag_type, "reason": reason},
    )
    _commit_if_supported(db_connection)
    logger.warning("Account account_id=%s flagged for review due to %s", account_id, reason)


def _generate_otp_code() -> str:
    return "".join(str(secrets.randbelow(10)) for _ in range(6))


def _last4(value: Any) -> str:
    text_value = str(value or "0000")
    return text_value[-4:].rjust(4, "*")


def _mask_email(value: Any) -> str:
    text_value = str(value or "unknown@example.com")
    if "@" not in text_value:
        return "***"
    local, domain = text_value.split("@", 1)
    return f"{local[:1]}***@{domain}"


def _iso_z(value: datetime) -> str:
    return value.isoformat(timespec="seconds") + "Z"


def _row_to_dict(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    if isinstance(row, dict):
        return row
    if hasattr(row, "_mapping"):
        return dict(row._mapping)
    return dict(row)


def _commit_if_supported(db_connection) -> None:
    if hasattr(db_connection, "commit"):
        db_connection.commit()
