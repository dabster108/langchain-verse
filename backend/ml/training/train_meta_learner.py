"""Train a meta-learner (stacking) over agent-level predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ml.mlflow.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODELS_OUTPUT_DIR


def load_meta_features(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    labels = pd.read_csv(data_dir / "fraud_labels_train.csv")
    baseline = pd.read_csv(data_dir / "rule_engine_baseline_predictions.csv")
    merged = baseline.merge(labels, on="txn_id", how="inner")
    feature_cols = [c for c in merged.columns if c not in ("txn_id", "is_fraud", "baseline_decision", "rule_triggered")]
    X = merged[feature_cols].select_dtypes(include="number").fillna(0).values
    y = merged["is_fraud"].astype(int).values
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train meta-learner stacking model")
    parser.add_argument("--data-dir", type=Path, default=Path("backend/datasets"))
    parser.add_argument("--output-dir", type=Path, default=MODELS_OUTPUT_DIR)
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_meta_features(args.data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")

    with mlflow.start_run(run_name="meta_learner"):
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)

        mlflow.log_param("model_type", "meta_learner")
        mlflow.log_metric("auroc", auroc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.sklearn.log_model(model, artifact_path="model")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.output_dir / "meta_learner.joblib")
        meta = {"feature_dim": X.shape[1], "auroc": auroc, "pr_auc": pr_auc}
        (args.output_dir / "meta_learner_metrics.json").write_text(json.dumps(meta, indent=2))
        print(
            f"Meta-learner saved → {args.output_dir / 'meta_learner.joblib'}  "
            f"AUROC={auroc:.4f}  PR-AUC={pr_auc:.4f}"
        )


if __name__ == "__main__":
    main()
