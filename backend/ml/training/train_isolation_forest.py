"""
Train the Behavior Agent Isolation Forest anomaly detector.

SMALL-DATASET CAVEAT
--------------------
~1,000 rows / ~18 fraud cases is insufficient for reliable anomaly detection.
Training is unsupervised on all rows; evaluation against ``is_fraud`` is
reference-only. Re-validate when a larger dataset is available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.mlflow.config import MLFLOW_TRACKING_URI, MODELS_OUTPUT_DIR
from ml.training.data_utils import (
    DEFAULT_FEATURE_TABLE,
    derive_feature_columns,
    load_feature_columns,
    load_feature_table,
    prepare_xy,
)

EXPERIMENT_NAME = "behavior_agent_isolation_forest"
MODEL_FILENAME = "isolation_forest_model.pkl"


def train_isolation_forest(
    feature_table_path: Path | None = None,
    output_dir: Path | None = None,
    *,
    random_state: int = 42,
) -> dict[str, float | str]:
    """Train Isolation Forest on the full feature matrix (unsupervised)."""
    out_dir = output_dir or MODELS_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_table(feature_table_path)
    feature_columns = load_feature_columns() or derive_feature_columns(df)
    X, y, _ = prepare_xy(df, feature_columns)

    contamination = float(y.mean())
    contamination = min(max(contamination, 0.001), 0.5)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="isolation_forest"):
        # Unsupervised fit on ALL rows — labels are not used for training.
        model.fit(X)

        anomaly_flags = model.predict(X)
        y_pred = (anomaly_flags == -1).astype(int)
        scores = -model.decision_function(X)

        auroc = float(roc_auc_score(y, scores)) if y.nunique() > 1 else 0.0
        precision = float(precision_score(y, y_pred, zero_division=0))
        recall = float(recall_score(y, y_pred, zero_division=0))
        f1 = float(f1_score(y, y_pred, zero_division=0))
        cm = confusion_matrix(y, y_pred)

        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_metric("auroc", auroc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(model, artifact_path="model")

        model_path = out_dir / MODEL_FILENAME
        joblib.dump(model, model_path)

        print("=== Isolation Forest evaluation (full dataset, reference only) ===")
        print(f"Contamination (empirical fraud rate): {contamination:.4f}")
        print(f"AUROC:     {auroc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"Confusion matrix:\n{cm}")
        print(f"Saved model → {model_path}")

        return {
            "model": "isolation_forest",
            "metric_name": "auroc",
            "metric_value": auroc,
            "path": str(model_path),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Behavior Agent Isolation Forest")
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=MODELS_OUTPUT_DIR)
    args = parser.parse_args()
    train_isolation_forest(args.feature_table, args.output_dir)


if __name__ == "__main__":
    main()
