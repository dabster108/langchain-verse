# Offline Model Training Pipeline

Trains Behavior Agent models (and the synthesis meta-learner) from `datasets_processed/feature_table.csv` and writes artifacts to `ml/models/`.

## Small-dataset caveat

The current feature table has **~1,000 rows** and **~1.8% fraud (~18 positives)**. Every script includes a comment block noting that:

- Supervised models will overfit and metrics will be unstable.
- The LSTM sequence model is structurally correct but **not meaningful** when most accounts have a single transaction.
- Meta-learner agent scores are **mocked** until real Velocity/Geo/Behavior agents produce held-out outputs.

The code is written for production scale — re-run and re-validate once a larger labelled dataset is available.

## Scripts

| Script | Input | Output | MLflow experiment |
|--------|-------|--------|-------------------|
| `train_xgboost.py` | `feature_table.csv` | `xgboost_model.pkl`, `feature_columns.json` | `behavior_agent_xgboost` |
| `train_isolation_forest.py` | same feature columns | `isolation_forest_model.pkl` | `behavior_agent_isolation_forest` |
| `train_lstm.py` | per-account sequences (5 numeric features, window=64) | `lstm_model.pt` | `behavior_agent_lstm` |
| `train_meta_learner.py` | mocked agent (r,c) tuples + `transaction_type` | `meta_learner_model.pkl` | `synthesis_meta_learner` |
| `run_all_training.py` | orchestrates all four | summary table | — |
| `data_utils.py` | shared loaders / column selection | — | — |

### Feature columns (tree models)

`train_xgboost.py` drops ID/metadata columns (`txn_id`, `account_id`, timestamps, free-text fields, label metadata) and keeps numeric/boolean engineered features. The final column list is saved to:

```
ml/models/feature_columns.json
```

`train_isolation_forest.py` reuses that file when present.

### LSTM sequences

Built from `feature_table.csv` sorted by `account_id` + `timestamp`:

- Features: `amount_npr`, `hour_of_day`, `is_night`, `amount_ratio`, `vel_z_score_amount`
- Window length: **64** (paper spec), zero-padded with a mask
- Label: `is_fraud` of the **last** transaction in each account sequence

## Run everything

From the `backend/` directory (requires `xgboost`, `scikit-learn`, `torch`, `mlflow`, `joblib`, `pandas`, `numpy`):

```bash
python -m ml.training.run_all_training
```

Or run individual scripts:

```bash
python -m ml.training.train_xgboost
python -m ml.training.train_isolation_forest
python -m ml.training.train_lstm
python -m ml.training.train_meta_learner
```

## Artifacts (`ml/models/` — gitignored)

| File | Used by |
|------|---------|
| `feature_columns.json` | All tree models + live inference column alignment |
| `xgboost_model.pkl` | Behavior Agent (primary classifier) |
| `isolation_forest_model.pkl` | Behavior Agent (anomaly fallback) |
| `lstm_model.pt` | Behavior Agent (sequence model, when enough history exists) |
| `meta_learner_model.pkl` | Synthesis Agent (stacking layer) |

The live pipeline (Behavior Agent service) will load these artifacts at startup from `services/behavior-agent/models/` (Docker volume bind-mounted from `ml/models/`). Until that wiring is complete, artifacts remain on disk after offline training.

## MLflow UI

See [ml/mlflow/README.md](../mlflow/README.md).
