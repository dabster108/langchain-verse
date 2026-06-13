# ML Pipeline

Offline feature engineering, model training, and experiment tracking.

## Quick commands (from `backend/`)

```bash
uv sync --all-groups          # one env for services + ML
python -m ml.features.run_pipeline
python -m ml.training.run_all_training
mlflow ui --backend-store-uri mlruns
```

## Layout

| Path | Purpose |
|------|---------|
| `features/` | Clean raw CSVs → `datasets_processed/feature_table.csv` |
| `training/` | Train XGBoost, Isolation Forest, LSTM, meta-learner |
| `models/` | Serialized artifacts (gitignored) — mounted into `behavior-agent` |
| `mlflow/config.py` | Tracking URI (`backend/mlruns/`) |

## Small-dataset caveat

~1,000 rows / ~18 fraud cases. Metrics will be unstable; LSTM sequences are mostly length-1. Code is production-shaped — re-validate on full data.

## Artifacts

| File | Consumer |
|------|----------|
| `feature_columns.json` | Tree models + inference alignment |
| `xgboost_model.pkl` | Behavior Agent |
| `isolation_forest_model.pkl` | Behavior Agent |
| `lstm_model.pt` | Behavior Agent (when enough history) |
| `meta_learner_model.pkl` | Synthesis Agent |

## MLflow experiments

- `behavior_agent_xgboost`
- `behavior_agent_isolation_forest`
- `behavior_agent_lstm`
- `synthesis_meta_learner`

View at http://127.0.0.1:5000 after `mlflow ui --backend-store-uri mlruns`.
