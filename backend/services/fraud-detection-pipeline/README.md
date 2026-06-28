# Fraud Detection Pipeline

This is the unified backend service: one FastAPI process with five internal Python agents.

```text
POST /evaluate
  -> Velocity Agent
  -> Geo Agent
  -> Behavior Agent
  -> Synthesis Agent
  -> Decision/OTP Agent
```

It replaces the multi-port microservice runtime for local and deployment simplicity. The service runs on port `8001`.

## Architecture

- `app/agents/velocity_agent.py`: velocity and burst-risk checks.
- `app/agents/geo_agent.py`: impossible travel, VPN/Tor, datacenter IP, and optional Neo4j graph checks.
- `app/agents/behavior_agent.py`: behavior score module for the unified pipeline.
- `app/agents/synthesis_agent.py`: math-only two-layer score fusion.
- `app/agents/decision_agent.py`: PASS / OTP / BLOCK plus OTP metadata.
- `app/orchestrator.py`: simple chain that calls agents and writes an audit row when the audit table exists.

## Run

```bash
cd backend/services/fraud-detection-pipeline
uvicorn app.main:app --reload --port 8001
```

Health check:

```bash
curl http://localhost:8001/health
```

Evaluate one transaction:

```bash
curl -X POST http://localhost:8001/evaluate \
  -H "Content-Type: application/json" \
  -d '{"txn_id": "TXN-20260101-00000001", "account_id": "ACC-0000001"}'
```

## Decision And OTP

The Decision Agent is the final gate from paper §IV-F:

- `PASS`: final score `< 0.30`; transaction is approved without extra verification.
- `OTP`: final score `0.30-0.70`; transaction requires dual-path OTP.
- `BLOCK`: final score `> 0.70`; transaction is blocked and the account is flagged for KYC review.

For OTP, both SMS and EMAIL codes must verify before the transaction is allowed. If either channel expires or reaches 3 failed attempts, the transaction is blocked. If SMS fails but email succeeds, the session is marked with `sim_swap_suspected=true`.

Verify OTP:

```bash
curl -X POST http://localhost:8001/verify-otp \
  -H "Content-Type: application/json" \
  -d '{
    "otp_session_id": "OTP-A3F9B2",
    "sms_otp": "123456",
    "email_otp": "654321"
  }'
```

Check OTP status:

```bash
curl http://localhost:8001/otp-status/OTP-A3F9B2
```

Resend OTP:

```bash
curl -X POST http://localhost:8001/otp-resend/OTP-A3F9B2
```

## Synthesis Weights

The Synthesis Agent applies Equation 1 from the paper by blending transaction-type weights and fraud-pattern weights:

```text
w_velocity = 0.5 * w1_velocity + 0.5 * w2_velocity
w_geo = 0.5 * w1_geo + 0.5 * w2_geo
w_behavior = 0.5 * w1_behavior + 0.5 * w2_behavior
```

Then it applies Equation 2, confidence-weighted fusion:

```text
S =
  (w_velocity * c_velocity * r_velocity
   + w_geo * c_geo * r_geo
   + w_behavior * c_behavior * r_behavior)
  /
  (w_velocity * c_velocity
   + w_geo * c_geo
   + w_behavior * c_behavior)
```

If the population variance of the three agent risk scores is greater than `0.15`, the service adds `0.15` to `S` and clamps the final score to `[0, 1]`.

## Latency Target

Expected approximate latency:

- Velocity: `~2ms`
- Geo: `~50ms`
- Behavior: `~100ms`
- Synthesis: `~5ms`
- Decision: `~2ms`
- Total: `~160ms`

## Testing

```bash
pytest tests/
```

Manual cases to try once Postgres/Neo4j data is loaded:

- Normal transaction -> `PASS`
- High `z_score_amount` + new counterparty -> moderate risk, usually `OTP`
- Impossible travel -> high Geo risk
- COMM-042 ring member -> Fraud ring pattern and high final risk
