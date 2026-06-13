"""
Train the Behavior Agent XGBoost classifier on feature_table.csv.

SMALL-DATASET CAVEAT
--------------------
This pipeline uses ~1,000 rows with ~1.8% fraud (~18 positives). That is far too
small for robust supervised learning; expect high variance and overfitting. The
code structure is production-ready — re-run and validate once a larger labelled
dataset is available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from ml.mlflow.config import MLFLOW_TRACKING_URI, MODELS_OUTPUT_DIR
from ml.training.data_utils import (
    DEFAULT_FEATURE_TABLE,
    load_feature_table,
    prepare_xy,
    save_feature_columns,
)

EXPERIMENT_NAME = "behavior_agent_xgboost"
MODEL_FILENAME = "xgboost_model.pkl"


def train_xgboost(
    feature_table_path: Path | None = None,
    output_dir: Path | None = None,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float | str]:
    """Train, evaluate, persist, and MLflow-log an XGBoost fraud classifier."""
    out_dir = output_dir or MODELS_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_table(feature_table_path)
    X, y, feature_columns = prepare_xy(df)
    save_feature_columns(feature_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    n_positive = int(y_train.sum())
    n_negative = int(len(y_train) - n_positive)
    scale_pos_weight = n_negative / max(n_positive, 1)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost"):
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= 0.5).astype(int)

        auroc = float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else 0.0
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred)

        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_metric("auroc", auroc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(model, artifact_path="model")

        model_path = out_dir / MODEL_FILENAME
        joblib.dump(model, model_path)

        print("=== XGBoost evaluation (test set) ===")
        print(f"AUROC:     {auroc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"Confusion matrix:\n{cm}")
        print(f"Saved model → {model_path}")

        return {
            "model": "xgboost",
            "metric_name": "auroc",
            "metric_value": auroc,
            "path": str(model_path),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Behavior Agent XGBoost model")
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=MODELS_OUTPUT_DIR)
    args = parser.parse_args()
    train_xgboost(args.feature_table, args.output_dir)


if __name__ == "__main__":
    main()
