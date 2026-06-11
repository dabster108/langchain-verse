# Synthesis Agent

Two-layer dynamic weight blending across Velocity, Geo, and Behavior agents.

## Modules

| Module | Purpose |
|--------|---------|
| `app/pattern_classifier.py` | Classify fraud pattern (rapid transfers, fraud ring, money laundering, novel) |
| `app/weights.py` | Layer 1 + Layer 2 weight tables and 50/50 blending |
| `app/synthesis.py` | Confidence-weighted fusion (Eq. 2) and disagreement-variance check |

## Endpoints

- `GET /health`
- `POST /evaluate/synthesise` — combine three agent verdicts into a `SynthesisResult`

## Run locally

```bash
cd backend/services/synthesis-agent
uv run python main.py
```
