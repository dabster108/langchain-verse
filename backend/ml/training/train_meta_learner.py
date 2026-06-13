"""
Train the Synthesis Agent Random Forest meta-learner.

NOTE: Agent risk/confidence scores are MOCKED here because the Velocity, Geo,
and Behavior agents are not yet producing held-out evaluation outputs. Replace
``generate_mock_agent_scores()`` with real agent tuples once those services exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ml.mlflow.config import MLFLOW_TRACKING_URI, MODELS_OUTPUT_DIR
from ml.training.data_utils import DEFAULT_FEATURE_TABLE, load_feature_table

EXPERIMENT_NAME = "synthesis_meta_learner"
MODEL_FILENAME = "meta_learner_model.pkl"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _noisy_score(signal: np.ndarray, rng: np.random.Generator, noise: float = 0.35) -> np.ndarray:
    logits = signal + rng.normal(0.0, noise, size=len(signal))
    return np.clip(_sigmoid(logits), 0.0, 1.0)


def generate_mock_agent_scores(df: pd.DataFrame, *, random_state: int = 42) -> pd.DataFrame:
    """Produce synthetic (risk, confidence) tuples correlated with domain features."""
    rng = np.random.default_rng(random_state)
    out = pd.DataFrame(index=df.index)

    vel_z = df.get("vel_z_score_amount", pd.Series(0.0, index=df.index)).fillna(0)
    vel_1h = df.get("vel_txn_count_1h", pd.Series(0.0, index=df.index)).fillna(0)
    velocity_signal = 0.6 * vel_z + 0.4 * (vel_1h / max(vel_1h.max(), 1.0))

    geo_travel = df.get("geo_impossible_travel", pd.Series(False, index=df.index)).astype(int)
    geo_vpn = df.get("geo_is_vpn", pd.Series(False, index=df.index)).astype(int)
    geo_signal = 1.2 * geo_travel + 0.8 * geo_vpn + df.get("geo_prev_txn_km", pd.Series(0.0, index=df.index)).fillna(0) / 500.0

    behavior_signal = (
        1.5 * df.get("is_fraud_merchant", pd.Series(False, index=df.index)).astype(int)
        + 1.0 * df.get("is_structuring_amount", pd.Series(False, index=df.index)).astype(int)
        + 0.5 * df.get("amount_ratio", pd.Series(0.0, index=df.index)).fillna(0)
    )

    out["r_velocity"] = _noisy_score(velocity_signal.to_numpy(), rng, noise=0.40)
    out["r_geo"] = _noisy_score(geo_signal.to_numpy(), rng, noise=0.45)
    out["r_behavior"] = _noisy_score(behavior_signal.to_numpy(), rng, noise=0.35)

    out["c_velocity"] = np.clip(0.55 + 0.35 * out["r_velocity"] + rng.normal(0, 0.05, len(out)), 0, 1)
    out["c_geo"] = np.clip(0.50 + 0.40 * out["r_geo"] + rng.normal(0, 0.05, len(out)), 0, 1)
    out["c_behavior"] = np.clip(0.60 + 0.30 * out["r_behavior"] + rng.normal(0, 0.05, len(out)), 0, 1)
    out["transaction_type"] = df["type_encoded"].fillna(0).astype(int)
    return out


def train_meta_learner(
    feature_table_path: Path | None = None,
    output_dir: Path | None = None,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float | str]:
    """Train Random Forest meta-learner on mocked agent score tuples."""
    out_dir = output_dir or MODELS_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_table(feature_table_path)
    agent_scores = generate_mock_agent_scores(df, random_state=random_state)
    y = df["is_fraud"].astype(int)

    feature_cols = [
        "r_velocity",
        "r_geo",
        "r_behavior",
        "c_velocity",
        "c_geo",
        "c_behavior",
        "transaction_type",
    ]
    X = agent_scores[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="meta_learner"):
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= 0.5).astype(int)

        auroc = float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else 0.0
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred)

        mlflow.log_param("mock_agent_scores", True)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_metric("auroc", auroc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(model, artifact_path="model")

        model_path = out_dir / MODEL_FILENAME
        joblib.dump({"model": model, "feature_columns": feature_cols}, model_path)

        print("=== Meta-learner evaluation (mock agent scores, test set) ===")
        print(f"AUROC:     {auroc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"Confusion matrix:\n{cm}")
        print(f"Saved model → {model_path}")

        return {
            "model": "meta_learner",
            "metric_name": "auroc",
            "metric_value": auroc,
            "path": str(model_path),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train synthesis meta-learner")
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=MODELS_OUTPUT_DIR)
    args = parser.parse_args()
    train_meta_learner(args.feature_table, args.output_dir)


if __name__ == "__main__":
    main()
