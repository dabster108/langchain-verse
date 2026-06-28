import sys
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

    def execute(self, statement, params):
        query = str(statement)
        if "SELECT txn_type" in query:
            return FakeResult({"txn_type": self.txn_type})
        return FakeResult({})


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
    decision = decision_agent.evaluate_decision(0.5)

    assert decision["verdict"] == "OTP"
    assert decision["otp_session_id"].startswith("OTP-")
    assert decision["otp_channels"] == ["SMS", "EMAIL"]
