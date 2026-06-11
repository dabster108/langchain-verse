"""Train a lightweight LSTM sequence model (stub with MLflow logging)."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ml.mlflow.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODELS_OUTPUT_DIR


class FraudLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return torch.sigmoid(self.head(out[:, -1, :]))


def load_sequences(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    labels = pd.read_csv(data_dir / "fraud_labels_train.csv")
    velocity = pd.read_csv(data_dir / "velocity_snapshots.csv")
    merged = velocity.merge(labels, on="txn_id", how="inner")
    feature_cols = [c for c in merged.columns if c not in ("txn_id", "is_fraud")]
    X = merged[feature_cols].select_dtypes(include="number").fillna(0).values
    y = merged["is_fraud"].astype(int).values
    # Reshape to (N, seq_len=1, features) for LSTM input.
    return X[:, np.newaxis, :], y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM sequence fraud model")
    parser.add_argument("--data-dir", type=Path, default=Path("backend/datasets"))
    parser.add_argument("--output-dir", type=Path, default=MODELS_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_sequences(args.data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    device = torch.device("cpu")
    model = FraudLSTM(input_dim=X.shape[2]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    with mlflow.start_run(run_name="lstm"):
        for epoch in range(args.epochs):
            model.train()
            optimiser.zero_grad()
            preds = model(X_train_t)
            loss = loss_fn(preds, y_train_t)
            loss.backward()
            optimiser.step()
            mlflow.log_metric("train_loss", float(loss.item()), step=epoch)

        model.eval()
        with torch.no_grad():
            proba = model(X_test_t).numpy().reshape(-1)
        auroc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)

        mlflow.log_param("model_type", "lstm")
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_metric("auroc", auroc)
        mlflow.log_metric("pr_auc", pr_auc)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.output_dir / "lstm.pt")
        print(f"LSTM saved → {args.output_dir / 'lstm.pt'}  AUROC={auroc:.4f}  PR-AUC={pr_auc:.4f}")


if __name__ == "__main__":
    main()
