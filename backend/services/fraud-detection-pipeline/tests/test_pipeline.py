import sys
from datetime import datetime, timedelta
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

from app import orchestrator
from app.agents import behavior_agent, decision_agent, geo_agent, synthesis_agent, velocity_agent


class FakeResult:
    def __init__(self, row=None):
        self.row = row

    def fetchone(self):
        return self.row


class FakeConnection:
    def __init__(self, txn_type="ESEWA_P2P"):
        self.txn_type = txn_type
        self.otp_sessions = {}
        self.account_flags = []

    def execute(self, statement, params=None):
        query = str(statement)
        params = params or {}
        if "SELECT txn_type" in query:
            return FakeResult({"txn_type": self.txn_type})
        if "SELECT account_id, phone, email" in query:
            return FakeResult(
                {
                    "account_id": params["account_id"],
                    "phone": "9800001234",
                    "email": "user@example.com",
                }
            )
        if "INSERT INTO otp_sessions" in query:
            self.otp_sessions[params["otp_session_id"]] = dict(params)
            self.otp_sessions[params["otp_session_id"]].update(
                {
                    "channel_1_status": "PENDING",
                    "channel_2_status": "PENDING",
                    "final_decision": None,
                    "sim_swap_suspected": False,
                    "attempt_count_ch1": 0,
                    "attempt_count_ch2": 0,
                }
            )
            return FakeResult({})
        if "SELECT *" in query and "FROM otp_sessions" in query:
            return FakeResult(self.otp_sessions.get(params["otp_session_id"]))
        if "UPDATE otp_sessions" in query and "channel_1_status = 'EXPIRED'" in query:
            session = self.otp_sessions[params["otp_session_id"]]
            session["channel_1_status"] = "EXPIRED"
            session["channel_2_status"] = "EXPIRED"
            session["final_decision"] = "BLOCKED"
            return FakeResult({})
        if "UPDATE otp_sessions" in query:
            session = self.otp_sessions[params["otp_session_id"]]
            session.update(
                {
                    "channel_1_status": params["channel_1_status"],
                    "channel_2_status": params["channel_2_status"],
                    "attempt_count_ch1": params["attempt_count_ch1"],
                    "attempt_count_ch2": params["attempt_count_ch2"],
                    "sim_swap_suspected": params["sim_swap_suspected"],
                }
            )
            if params.get("final_decision"):
                session["final_decision"] = params["final_decision"]
            return FakeResult({})
        if "INSERT INTO account_flags" in query:
            self.account_flags.append(dict(params))
            return FakeResult({})
        return FakeResult({})

    def commit(self):
        return None


def _agent_output(risk, confidence=0.9, breakdown=None):
    return {
        "risk_score": risk,
        "confidence": confidence,
        "breakdown": breakdown or {},
    }


def test_end_to_end_transaction_reaches_final_verdict(monkeypatch):
    monkeypatch.setattr(velocity_agent, "evaluate_velocity", lambda *args: _agent_output(0.3, 0.9))
    monkeypatch.setattr(geo_agent, "evaluate_geo", lambda *args: _agent_output(0.4, 0.9))
    monkeypatch.setattr(behavior_agent, "evaluate_behavior", lambda *args: {**_agent_output(0.5, 0.9), "shap_explanation": []})

    result = orchestrator.evaluate_transaction("TXN-1", "ACC-1", FakeConnection(), None)

    assert result["txn_id"] == "TXN-1"
    assert result["final_verdict"] in {"PASS", "OTP", "BLOCK"}
    assert "synthesis" in result
    assert "decision" in result


def test_known_fraud_case_gets_block_or_otp(monkeypatch):
    monkeypatch.setattr(
        velocity_agent,
        "evaluate_velocity",
        lambda *args: _agent_output(0.85, 0.95, {"unique_counterparties_1h": 4}),
    )
    monkeypatch.setattr(
        geo_agent,
        "evaluate_geo",
        lambda *args: {
            **_agent_output(0.9, 0.95, {"fraud_ring_proximity_risk": 0.35}),
            "fraud_ring_details": {"is_near_fraud_seed": True},
        },
    )
    monkeypatch.setattr(behavior_agent, "evaluate_behavior", lambda *args: {**_agent_output(0.8, 0.9), "shap_explanation": []})

    result = orchestrator.evaluate_transaction("TXN-FRAUD", "ACC-1", FakeConnection("SWIFT_OUTWARD"), None)

    assert result["final_verdict"] in {"OTP", "BLOCK"}
    assert result["final_score"] >= 0.7 or result["final_verdict"] == "OTP"


def test_normal_case_gets_pass(monkeypatch):
    monkeypatch.setattr(velocity_agent, "evaluate_velocity", lambda *args: _agent_output(0.05, 0.95))
    monkeypatch.setattr(geo_agent, "evaluate_geo", lambda *args: {**_agent_output(0.04, 0.95), "fraud_ring_details": {}})
    monkeypatch.setattr(behavior_agent, "evaluate_behavior", lambda *args: {**_agent_output(0.08, 0.95), "shap_explanation": []})

    result = orchestrator.evaluate_transaction("TXN-NORMAL", "ACC-1", FakeConnection("ESEWA_P2P"), None)

    assert result["final_verdict"] == "PASS"
    assert result["final_score"] < 0.30


def test_synthesis_weights_prioritize_by_transaction_type():
    esewa = synthesis_agent.get_layer1_weights("ESEWA_P2P")
    swift = synthesis_agent.get_layer1_weights("SWIFT_OUTWARD")

    assert esewa["behavior"] == 0.55
    assert swift["geo"] == 0.55


def test_high_geo_risk_detects_fraud_ring_pattern():
    pattern = synthesis_agent.classify_fraud_pattern(
        0.2,
        0.8,
        0.4,
        geo_breakdown={"fraud_ring_proximity_risk": 0.35},
    )

    assert pattern == "Fraud ring"


def test_disagreement_check_bumps_score():
    result = synthesis_agent.evaluate_synthesis(
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        "CARD_POS",
        geo_breakdown={"impossible_travel": True},
    )

    assert result["disagreement_adjusted"] is True
    assert result["final_score"] > 0.15


def test_decision_agent_creates_otp_for_middle_band():
    db = FakeConnection()

    decision = decision_agent.evaluate_decision(0.5, "Novel pattern", "ACC-1", "TXN-1", db)

    assert decision["verdict"] == "OTP"
    assert decision["otp_session_id"].startswith("OTP-")
    assert decision["otp_channels"] == ["SMS", "EMAIL"]


def test_pass_verdict_has_no_otp():
    decision = decision_agent.evaluate_decision(0.2, "Novel pattern", "ACC-1", "TXN-1", FakeConnection())

    assert decision["verdict"] == "PASS"
    assert decision["threshold_used"] == "PASS (< 0.30)"


def test_block_verdict_flags_account():
    db = FakeConnection()

    decision = decision_agent.evaluate_decision(0.8, "Fraud ring", "ACC-1", "TXN-1", db)

    assert decision["verdict"] == "BLOCK"
    assert decision["account_status"] == "FLAGGED_FOR_REVIEW"
    assert db.account_flags[-1]["flag_type"] == "KYC_REVIEW_REQUIRED"


def test_otp_verification_success_allows_transaction():
    db = FakeConnection()
    decision = decision_agent.evaluate_decision(0.5, "Novel pattern", "ACC-1", "TXN-1", db)
    session = db.otp_sessions[decision["otp_session_id"]]

    result = decision_agent.verify_otp_session(
        decision["otp_session_id"],
        session["channel_1_otp"],
        session["channel_2_otp"],
        db,
    )

    assert result["channel_1_status"] == "VERIFIED"
    assert result["channel_2_status"] == "VERIFIED"
    assert result["final_decision"] == "ALLOWED"


def test_otp_verification_failure_three_times_auto_blocks():
    db = FakeConnection()
    decision = decision_agent.evaluate_decision(0.5, "Novel pattern", "ACC-1", "TXN-1", db)

    result = {}
    for _ in range(3):
        result = decision_agent.verify_otp_session(decision["otp_session_id"], "000000", "111111", db)

    assert result["final_decision"] == "BLOCKED"
    assert db.account_flags[-1]["flag_type"] == "KYC_REVIEW_REQUIRED"


def test_otp_expiration_blocks_session():
    db = FakeConnection()
    decision = decision_agent.evaluate_decision(0.5, "Novel pattern", "ACC-1", "TXN-1", db)
    db.otp_sessions[decision["otp_session_id"]]["expires_at"] = datetime.utcnow() - timedelta(seconds=1)

    try:
        decision_agent.verify_otp_session(decision["otp_session_id"], "123456", "654321", db)
    except decision_agent.OTPExpiredError:
        pass

    assert db.otp_sessions[decision["otp_session_id"]]["final_decision"] == "BLOCKED"


def test_sim_swap_detection_when_sms_fails_but_email_succeeds():
    db = FakeConnection()
    decision = decision_agent.evaluate_decision(0.5, "Novel pattern", "ACC-1", "TXN-1", db)
    session = db.otp_sessions[decision["otp_session_id"]]

    result = decision_agent.verify_otp_session(
        decision["otp_session_id"],
        "000000",
        session["channel_2_otp"],
        db,
    )

    assert result["sim_swap_suspected"] is True
    assert result["final_decision"] == "BLOCKED"
    assert db.account_flags[-1]["flag_type"] == "SIM_SWAP_DETECTED"
