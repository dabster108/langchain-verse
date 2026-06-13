# ML Training Pipeline

Weekly retraining and challenger–champion promotion workflow for the fraud detection models.

## Layout

```
ml/
├── training/          # Training scripts (XGBoost, Isolation Forest, LSTM, meta-learner)
├── mlflow/            # Experiment tracking + model registry config
└── models/            # Serialized artifacts (gitignored, Docker volume mount)
```

## Weekly retraining workflow

1. **Extract** — load labelled data from `backend/datasets/` (and production Postgres in prod).
2. **Train** — run each script; metrics are logged to MLflow under experiment `fraud-detection`.
3. **Evaluate** — compare challenger PR-AUC / AUROC against the current champion (see `eval/offline_validation.py`).
4. **Promote** — if challenger beats champion by ≥ `PROMOTION_MIN_DELTA` (default 0.01 PR-AUC), alias it as `champion` in the MLflow registry and copy artifacts to `ml/models/`.
5. **Deploy** — `behavior-agent` hot-reloads from `services/behavior-agent/models/` (volume-mounted from `ml/models/`).

## Run training locally

From the `backend/` directory:

```bash
python -m ml.training.run_all_training
```

Individual scripts:

```bash
python -m ml.training.train_xgboost
python -m ml.training.train_isolation_forest
python -m ml.training.train_lstm
python -m ml.training.train_meta_learner
```

Input: `datasets_processed/feature_table.csv` (from `ml.features.run_pipeline`).

## MLflow

- Tracking URI: `file:./mlruns/` (see `ml/mlflow/README.md`)
- View runs: `mlflow ui --backend-store-uri mlruns` (from `backend/`)

## Champion / challenger aliases

| Alias       | Role                                              |
|-------------|---------------------------------------------------|
| `champion`  | Production model served by `behavior-agent`       |
| `challenger`| Candidate from latest weekly retrain              |

Promotion requires the challenger to exceed the champion on PR-AUC by at least 0.01 on the held-out eval set (`fraud_labels_eval_HIDDEN.csv` via `eval/offline_validation.py`).
