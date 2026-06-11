# Decision & OTP Service

Owns the PASS / OTP / BLOCK threshold layer (τ_low=0.30, τ_high=0.70) and the dual-path OTP interlock (mock Sparrow SMS + email, 3-minute verification window).

## Endpoints

- `GET /health`
- `POST /evaluate/decision` — classify a synthesised score
- `POST /evaluate/otp/initiate` — send dual-path OTP codes
- `POST /evaluate/otp/verify` — verify SMS and/or email codes

## Run locally

```bash
cd backend/services/decision-otp-service
uv run python main.py
```
