# MLflow Tracking

Offline training scripts log parameters, metrics, and model artifacts to a local MLflow store.

## Tracking URI

Default: `file:backend/mlruns/` (configured in `ml/mlflow/config.py`).

## View experiments

From the `backend/` directory:

```bash
mlflow ui --backend-store-uri mlruns
```

Then open http://127.0.0.1:5000 in your browser.

## Experiments logged by the training pipeline

| Experiment | Script |
|------------|--------|
| `behavior_agent_xgboost` | `ml/training/train_xgboost.py` |
| `behavior_agent_isolation_forest` | `ml/training/train_isolation_forest.py` |
| `behavior_agent_lstm` | `ml/training/train_lstm.py` |
| `synthesis_meta_learner` | `ml/training/train_meta_learner.py` |

Each run records hyperparameters, evaluation metrics, and a serialized model artifact.
