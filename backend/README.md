# Backend

Backend services for the Agentic Multi-Model Fraud Detection Framework for Nepal Digital Payments.

## Services

- `api-gateway`: future public API entrypoint.
- `geo-agent`: location and graph-context risk service.
- `velocity-agent`: transaction velocity risk service.
- `behavior-agent`: ML behavior risk service.
- `synthesis-agent`: future score aggregation service.

Each service is independently runnable with `uv` from its own folder.

```bash
cd backend/services/geo-agent
uv run python main.py
```

## Infrastructure

- PostgreSQL stores transactions, agent outputs, and final decisions.
- Redis stores fast temporary state and supports Pub/Sub channels.
- Neo4j stores graph relationships between users, devices, merchants, IPs, and locations.
