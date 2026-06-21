# Fraud Detection System Knowledge Notes

This file explains the current backend in a simple way so teammates/friends can understand what each part does, which agent is responsible for which task, what has been done for Neo4j, what PostgreSQL is meant to do, and how much of the backend is complete right now.

## 1. Project Idea

The backend is designed as a real-time fraud detection system for digital payment transactions.

Basic flow:

1. A transaction comes into the API Gateway.
2. The transaction is checked by multiple independent agents.
3. Each agent gives a fraud risk score and confidence score.
4. The Synthesis Agent combines those scores into one final score.
5. The Decision/OTP Service decides whether the transaction should `PASS`, require `OTP`, or be `BLOCKED`.
6. In the complete version, important decisions and explanations should be stored in PostgreSQL for audit/history.

The design is based on an agentic multi-model fraud detection architecture. Each agent looks at fraud from a different angle instead of relying on only one model.

## 2. Backend Folder Structure

The main backend folder is `backend/`.

### `backend/services/`

This is the most important folder for the runtime system. It contains all FastAPI microservices.

Each service is written as a separate app under:

```text
backend/services/<service-name>/app/
```

Current services:

- `api-gateway`
- `velocity-agent`
- `geo-agent`
- `behavior-agent`
- `synthesis-agent`
- `decision-otp-service`

Each service has its own `main.py`, and most services expose an endpoint through `routers/evaluate.py`.

### `backend/shared/`

This folder contains code reused by all services.

Important parts:

- `shared/config/settings.py` contains environment-based settings for PostgreSQL, Redis, and Neo4j.
- `shared/constants/service_names.py` stores standard service names.
- `shared/constants/channels.py` stores Redis Stream channel names.
- `shared/schemas/transaction.py` defines the transaction payload shape.
- `shared/schemas/risk.py` defines shared risk response models, fraud patterns, decision actions, and synthesis output models.
- `shared/schemas/events.py` defines event envelopes for stream-based communication.
- `shared/utils/redis_pubsub.py` contains Redis Streams helper functions.
- `shared/utils/serialization.py` contains JSON serialization helpers.
- `shared/explainability/shap_utils.py` contains SHAP explanation utilities.
- `shared/routers/health.py` provides shared health check routes.

This folder exists so all services use the same contracts and do not define different request/response shapes.

### `backend/ml/`

This folder contains the offline machine learning pipeline.

Important parts:

- `ml/features/` cleans raw data and builds the feature table.
- `ml/training/` trains models such as XGBoost, Isolation Forest, LSTM, and the meta learner.
- `ml/models/` is where trained model artifacts should be stored. It is gitignored except for placeholder files.
- `ml/mlflow/` stores MLflow tracking configuration.

The Behavior Agent uses trained model files from `ml/models/` when they exist. If the trained model files are missing, it can fall back to a heuristic mode.

### `backend/scripts/`

This folder contains operational scripts and loaders.

Current important script:

- `scripts/load_neo4j/loader.py` loads processed account graph data into Neo4j.

There is also documentation for the loader in:

- `scripts/load_neo4j/README.md`

### `backend/docker/`

This folder contains database/cache configuration files used by Docker.

Current files:

- `docker/neo4j/constraints.cypher`
- `docker/redis/redis.conf`

Important note: `docker-compose.yml` currently references `docker/postgres/init.sql`, but that file/folder is not present right now. That means PostgreSQL is configured in Docker Compose, but the actual database initialization SQL is not completed in the repository at the moment.

### `backend/eval/`

This folder is for offline validation and evaluation.

Current files:

- `eval/offline_validation.py`
- `eval/offline_validation.ipynb`

This is meant to test fraud detection performance against prepared datasets and compare metrics such as AUROC, precision, recall, F1, and PR-AUC.

### `backend/tests/`

This folder contains pytest tests.

Current test file:

- `tests/test_services.py`

The tests check:

- All service health endpoints.
- Geo Agent risk endpoint.
- Velocity Agent risk endpoint.
- Behavior Agent risk endpoint with SHAP output.
- Synthesis Agent endpoint.
- Decision/OTP Service thresholds.

### Root Backend Files

Important root files:

- `backend/docker-compose.yml` starts PostgreSQL, Redis, Neo4j, and all backend services.
- `backend/Dockerfile` builds one common service image.
- `backend/run_service.py` runs a selected service.
- `backend/train.py` runs the ML feature pipeline and training.
- `backend/pyproject.toml` contains Python dependencies and project configuration.
- `backend/README.md` explains the intended architecture.

## 3. Which Agent Does Which Task

### API Gateway

Folder:

```text
backend/services/api-gateway/
```

Purpose:

- This should be the public entry point for transaction risk requests.
- In the full system, it should receive a transaction and send work to the agent services.

Current status:

- Only the FastAPI app and `/health` endpoint are implemented.
- Real transaction ingestion and orchestration are not implemented yet.

### Velocity Agent

Folder:

```text
backend/services/velocity-agent/
```

Purpose:

- Detects fraud based on transaction speed and amount.
- It is meant to catch suspicious bursts, such as many transactions in a short time.

Current implementation:

- Endpoint: `/evaluate/risk`
- Uses:
  - `txn_count_1h`
  - `txn_count_24h`
  - `amount`
- Calculates a risk score using a simple heuristic:
  - High hourly transaction count increases risk.
  - Large transaction amount increases risk.

Current limitation:

- The design says this should use Redis counters for real-time transaction velocity.
- Right now, the endpoint receives counts directly in the request and does not calculate them from Redis yet.

### Geo Agent

Folder:

```text
backend/services/geo-agent/
```

Purpose:

- Detects fraud using location and graph context.
- It is meant to identify suspicious location changes, distance from normal user location, and closeness to fraud rings.

Current implementation:

- Endpoint: `/evaluate/risk`
- Uses:
  - `distance_from_home_km`
  - `is_new_location`
  - `ring_proximity_score`
- Calculates risk using distance, new location, and ring proximity.

Current limitation:

- The design says this should use Neo4j for graph-based fraud ring detection.
- Right now, the Geo Agent does not query Neo4j directly. It expects `ring_proximity_score` to already be provided.

### Behavior Agent

Folder:

```text
backend/services/behavior-agent/
```

Purpose:

- Detects fraud using user behavior and machine learning.
- This is the main ML-based risk agent.

Current implementation:

- Endpoint: `/evaluate/risk`
- Loads model artifacts if available:
  - XGBoost
  - Isolation Forest
  - LSTM
  - feature columns
  - feature table
- Produces a fraud risk score.
- Produces SHAP-style explanations.
- Falls back to heuristic mode if trained model artifacts are missing.

Current limitation:

- The model system is production-shaped, but actual quality depends on trained artifacts and dataset size.
- The local ML README notes that the current small dataset has around 1,000 rows and very few fraud examples, so metrics can be unstable.

### Synthesis Agent

Folder:

```text
backend/services/synthesis-agent/
```

Purpose:

- Combines the output of Velocity Agent, Geo Agent, and Behavior Agent.
- This is the decision fusion layer.

Current implementation:

- Endpoint: `/evaluate/synthesise`
- Classifies the fraud pattern:
  - `rapid_transfers`
  - `fraud_ring`
  - `money_laundering`
  - `novel_pattern`
- Applies two-layer weights:
  - Layer 1 weights based on transaction type.
  - Layer 2 weights based on fraud pattern.
- Blends the weights 50/50.
- Uses confidence-weighted scoring:

```text
final_score = sum(weight * risk_score * confidence) / sum(weight * confidence)
```

- Checks disagreement between agents. If agent scores disagree too much, it pushes the decision toward OTP.

Current status:

- Core synthesis logic is implemented.
- It can combine manually supplied agent verdicts.
- It is not yet fully wired to automatically consume results from the other services through Redis Streams.

### Decision/OTP Service

Folder:

```text
backend/services/decision-otp-service/
```

Purpose:

- Converts the final fraud score into an action.
- Handles OTP logic for medium-risk transactions.

Current implementation:

- Endpoint for decision evaluation exists.
- Uses thresholds:
  - score `< 0.30` means `PASS`
  - score `>= 0.30` and `< 0.70` means `OTP`
  - score `>= 0.70` means `BLOCK`
- Has mock/in-memory OTP logic.

Current limitation:

- OTP is not integrated with real SMS/email providers yet.
- OTP data is not stored durably in PostgreSQL yet.

## 4. Neo4j Work Completed

Neo4j is meant to store graph relationships between accounts/transactions so the system can detect fraud rings and suspicious money movement patterns.

### What Neo4j Is For

Neo4j should help with:

- Finding accounts close to known fraud rings.
- Detecting circular transaction flows.
- Detecting connected account groups.
- Supporting graph-based risk signals for the Geo Agent.
- Marking known fraud communities such as `COMM-042`.

### Files Related To Neo4j

Important files:

- `backend/docker-compose.yml`
- `backend/docker/neo4j/constraints.cypher`
- `backend/scripts/load_neo4j/loader.py`
- `backend/scripts/load_neo4j/README.md`
- `backend/shared/config/settings.py`

### Docker Compose Neo4j

`docker-compose.yml` defines a Neo4j service:

- Image: `neo4j:5`
- Container name: `fraud_neo4j`
- Browser port: `7474`
- Bolt port: `7687`
- Data volume: `neo4j_data`
- Import folder mounted from `backend/docker/neo4j`

### Neo4j Constraints

`docker/neo4j/constraints.cypher` currently defines uniqueness constraints for:

- `User.user_id`
- `Device.device_id`
- `Merchant.merchant_id`
- `Transaction.transaction_id`

### Neo4j Loader

The loader script is:

```text
backend/scripts/load_neo4j/loader.py
```

It reads:

- `datasets_processed/account_graph_nodes.csv`
- `datasets_processed/account_graph_edges.csv`
- `datasets/comm042_ring_members.json`

It creates:

- `Account` nodes
- `TRANSFER` relationships between accounts
- `is_fraud_seed` markings
- `community_id = "COMM-042"` for known ring members

It also:

- Verifies connection to Neo4j.
- Creates an `Account.id` uniqueness constraint.
- Supports `--clear` to delete existing Account graph data before loading.
- Uses `MERGE`, so it can be safely re-run without duplicating matching nodes/edges.
- Prints summary stats after loading.

Run command from `backend/`:

```bash
python -m scripts.load_neo4j.loader
```

With clearing:

```bash
python -m scripts.load_neo4j.loader --clear --yes
```

### Current Neo4j Status

Completed:

- Neo4j is included in Docker Compose.
- Neo4j dependency is included in Python project dependencies.
- Neo4j settings exist.
- Constraint file exists.
- Real loader script exists.
- Loader can create account nodes and transfer relationships.
- Loader can mark `COMM-042` fraud ring accounts.

Not completed:

- Geo Agent does not directly query Neo4j yet.
- API Gateway does not use Neo4j yet.
- No live graph lookup is connected to the real-time transaction flow yet.
- There is a small mismatch: `constraints.cypher` defines `User`, `Device`, `Merchant`, and `Transaction`, while the loader currently creates `Account` and `TRANSFER` graph structures.

## 5. PostgreSQL Work And Intended Usage

PostgreSQL is meant to be the durable relational database for the fraud detection system.

### What PostgreSQL Should Be Used For

In the full system, PostgreSQL should store:

- Transactions received by the API Gateway.
- Final fraud decisions.
- Agent risk results.
- SHAP/explanation records.
- OTP attempts and OTP status.
- Audit logs for compliance and debugging.
- Possibly user/device/merchant profile records.
- Historical labeled data for future model training.

### Files Related To PostgreSQL

Important files:

- `backend/docker-compose.yml`
- `backend/shared/config/settings.py`
- `backend/pyproject.toml`

### Docker Compose PostgreSQL

`docker-compose.yml` defines a PostgreSQL service:

- Image: `postgres:15`
- Container name: `fraud_postgres`
- Port: `5432`
- Database default: `fraud_db`
- User default: `postgres`
- Password default: `password`
- Volume: `postgres_data`

### Current PostgreSQL Status

Completed:

- PostgreSQL service is added to Docker Compose.
- PostgreSQL settings exist in `shared/config/settings.py`.
- `psycopg2-binary` dependency exists in `pyproject.toml`.
- Other services declare dependency on Postgres in Docker Compose.

Not completed:

- `docker/postgres/init.sql` is referenced by Docker Compose but does not currently exist.
- No service currently writes transactions or decisions to PostgreSQL.
- No database repository/DAO layer exists yet.
- Decision/OTP audit logging is described in the README but not implemented yet.
- OTP records are not persisted yet.

So PostgreSQL is planned and configured at the infrastructure level, but the actual schema and service integration still need to be built.

## 6. Redis Work And Intended Usage

Redis is used for stream/event infrastructure and should also support real-time counters.

### What Redis Should Be Used For

In the full system, Redis should be used for:

- Redis Streams between services.
- API Gateway broadcasting transaction events.
- Agents publishing their results.
- Velocity counters such as recent transaction counts.
- Possible short-lived OTP/session state.

### Current Redis Status

Completed:

- Redis service is in Docker Compose.
- Redis config exists in `docker/redis/redis.conf`.
- Redis settings exist in `shared/config/settings.py`.
- Redis Stream names exist in `shared/constants/channels.py`.
- Stream helper functions exist in `shared/utils/redis_pubsub.py`.

Not completed:

- Services do not yet run background workers consuming Redis Streams.
- API Gateway does not publish transaction events to Redis Streams yet.
- Velocity Agent does not calculate counters from Redis yet.
- Full service-to-service fanout is not wired yet.

## 7. Machine Learning Work Completed

The ML pipeline is under:

```text
backend/ml/
```

Completed:

- Feature engineering folder exists.
- Training scripts exist for:
  - XGBoost
  - Isolation Forest
  - LSTM
  - meta learner
- MLflow config exists.
- `train.py` exists as a one-shot training entrypoint.
- Behavior Agent can load trained artifacts when present.
- Behavior Agent can fall back to heuristic mode when artifacts are missing.
- SHAP-style explanation utilities exist.

Important note:

- The current dataset mentioned in `ml/README.md` is small, so model metrics may not be reliable yet.
- The code structure is ready for larger data, but real model validation still depends on better/full datasets.

## 8. Docker And Local Running

The backend is designed to run using Docker Compose.

Command:

```bash
cd backend
docker compose up --build
```

Services and ports:

- API Gateway: `8000`
- Velocity Agent: `8001`
- Geo Agent: `8002`
- Behavior Agent: `8003`
- Synthesis Agent: `8004`
- Decision/OTP Service: `8005`
- PostgreSQL: `5432`
- Redis: `6379`
- Neo4j Browser: `7474`
- Neo4j Bolt: `7687`

Each service has a health endpoint:

```text
/health
```

Run tests:

```bash
cd backend
uv run pytest
```

Run training:

```bash
cd backend
python train.py
```

## 9. What Is Completed So Far

Roughly completed:

- Backend folder structure.
- FastAPI service skeletons.
- Health endpoints for all services.
- Velocity Agent risk endpoint.
- Geo Agent risk endpoint.
- Behavior Agent risk endpoint.
- Behavior Agent model loading/fallback logic.
- SHAP-style explanation output.
- Synthesis Agent scoring and weight blending logic.
- Decision/OTP Service threshold logic.
- Mock/in-memory OTP logic.
- Shared schemas and constants.
- Redis Stream helper functions.
- Docker Compose infrastructure for services, Redis, Neo4j, and PostgreSQL.
- Neo4j graph loader.
- ML feature/training pipeline structure.
- Basic pytest service tests.

## 10. What Is Still Remaining

Main remaining tasks:

- Build real API Gateway transaction ingestion endpoint.
- Make API Gateway publish transaction events to Redis Streams.
- Add background workers so agents consume stream events automatically.
- Make agents publish risk results back to Redis Streams.
- Wire the full runtime pipeline end-to-end:
  - API Gateway
  - Velocity Agent
  - Geo Agent
  - Behavior Agent
  - Synthesis Agent
  - Decision/OTP Service
- Make Velocity Agent compute counters from Redis instead of receiving counts directly.
- Make Geo Agent query Neo4j directly for fraud ring proximity.
- Create the missing PostgreSQL init schema.
- Persist transactions, agent scores, final decisions, SHAP explanations, and OTP logs in PostgreSQL.
- Replace mock OTP with real SMS/email integration.
- Add stronger end-to-end tests.
- Validate model performance with better/full datasets.
- Clean up the mismatch between Neo4j constraint labels and the loader's `Account` graph model.

## 11. Current Completion Estimate

Current backend status is best described as a working prototype foundation, not a fully wired production backend.

Approximate completion:

```text
45% to 55% complete
```

Why:

- The service structure, core agent endpoints, synthesis logic, decision thresholds, ML pipeline shape, Docker setup, tests, and Neo4j loader are already present.
- However, the real end-to-end flow is not fully connected yet.
- Redis Streams are defined but not fully used by services.
- PostgreSQL is configured but not actually used by services yet.
- Neo4j data loading exists, but real-time Geo Agent graph querying is not wired yet.

In simple words: the brain parts are mostly shaped, but the full nervous system connecting all services together still needs work.

## 12. One-Line Summary For Presentation

This backend currently has a multi-agent fraud detection prototype with separate Velocity, Geo, Behavior, Synthesis, and Decision/OTP services, plus ML training and Neo4j graph loading support. The next major work is connecting the services end-to-end through Redis Streams and saving all transaction/audit results into PostgreSQL.
