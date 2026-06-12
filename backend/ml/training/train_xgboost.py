"""Train an XGBoost fraud classifier and log to MLflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from ml.mlflow.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODELS_OUTPUT_DIR


def load_training_frame(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    labels = pd.read_csv(data_dir / "fraud_labels_train.csv")
    txns = pd.read_csv(data_dir / "transactions_raw.csv")
    merged = txns.merge(labels, on="txn_id", how="inner")
    feature_cols = [c for c in merged.columns if c not in ("txn_id", "is_fraud", "timestamp")]
    X = merged[feature_cols].select_dtypes(include="number").fillna(0)
    y = merged["is_fraud"].astype(int)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost fraud classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("backend/datasets"))
    parser.add_argument("--output-dir", type=Path, default=MODELS_OUTPUT_DIR)
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_training_frame(args.data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=1.2,
        eval_metric="logloss",
        random_state=43 ,
    )

    with mlflow.start_run(run_name="xgboost"):
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)

        mlflow.log_param("model_type", "xgboost")
        mlflow.log_metric("auroc", auroc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.sklearn.log_model(model, artifact_path="model")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.output_dir / "xgboost.joblib")
        print(f"XGBoost saved → {args.output_dir / 'xgboost.joblib'}  AUROC={auroc:.4f}  PR-AUC={pr_auc:.4f}")


if __name__ == "__main__":
    main()
