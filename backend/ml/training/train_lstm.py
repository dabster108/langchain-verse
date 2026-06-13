"""
Train a per-account LSTM sequence model for the Behavior Agent.

NOTE: With only ~1,000 transactions across ~1,000 accounts, most accounts have
only one transaction — sequence modeling is NOT meaningful at this scale. This
script is structured for when per-account history (50+ transactions/account) is
available. Training will run but results are not meaningful now.

Label choice: each padded window is labelled with ``is_fraud`` of the LAST
transaction in that account's chronological sequence (predict fraud at the
current event given prior context). Per-timestep labels would require many more
positive sequences than we have here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.mlflow.config import MLFLOW_TRACKING_URI, MODELS_OUTPUT_DIR
from ml.training.data_utils import DEFAULT_FEATURE_TABLE, LSTM_SEQUENCE_FEATURES, load_feature_table

EXPERIMENT_NAME = "behavior_agent_lstm"
MODEL_FILENAME = "lstm_model.pt"
SEQUENCE_WINDOW = 64


@dataclass
class SequenceBatch:
    """Padded sequence tensors and labels."""

    sequences: np.ndarray
    labels: np.ndarray
    masks: np.ndarray


class FraudLSTM(nn.Module):
    """Simple stacked LSTM for binary fraud classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


def build_sequences(
    df: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...] = LSTM_SEQUENCE_FEATURES,
    window: int = SEQUENCE_WINDOW,
) -> SequenceBatch:
    """Build padded per-account transaction sequences.

    Sequences are sorted by ``account_id`` + ``timestamp``. Each account
    contributes one sample labelled with the fraud flag of its most recent
    transaction.
    """
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    work = work.sort_values(["account_id", "timestamp"])

    sequences: list[np.ndarray] = []
    labels: list[int] = []
    masks: list[np.ndarray] = []

    for _, group in work.groupby("account_id", sort=False):
        feats = group[list(feature_cols)].fillna(0).astype(float).values
        label = int(group["is_fraud"].iloc[-1])

        if len(feats) >= window:
            seq = feats[-window:]
            mask = np.ones(window, dtype=np.float32)
        else:
            pad_len = window - len(feats)
            seq = np.vstack([np.zeros((pad_len, len(feature_cols)), dtype=float), feats])
            mask = np.array([0.0] * pad_len + [1.0] * len(feats), dtype=np.float32)

        sequences.append(seq)
        labels.append(label)
        masks.append(mask)

    return SequenceBatch(
        sequences=np.asarray(sequences, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        masks=np.asarray(masks, dtype=np.float32),
    )


def train_lstm(
    feature_table_path: Path | None = None,
    output_dir: Path | None = None,
    *,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    random_state: int = 42,
) -> dict[str, float | str]:
    """Train the LSTM sequence model and persist the state dict."""
    out_dir = output_dir or MODELS_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(random_state)
    df = load_feature_table(feature_table_path)
    batch = build_sequences(df)

    X = torch.tensor(batch.sequences, dtype=torch.float32)
    y = torch.tensor(batch.labels, dtype=torch.float32).unsqueeze(1)

    pos_weight = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1.0)])
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    model = FraudLSTM(input_dim=len(LSTM_SEQUENCE_FEATURES))
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    final_loss = 0.0
    with mlflow.start_run(run_name="lstm"):
        mlflow.log_param("sequence_window", SEQUENCE_WINDOW)
        mlflow.log_param("sequence_features", list(LSTM_SEQUENCE_FEATURES))
        mlflow.log_param("n_sequences", len(X))
        mlflow.log_param("epochs", epochs)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                optimiser.zero_grad()
                logits = model(x_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimiser.step()
                epoch_loss += float(loss.item()) * len(x_batch)
            final_loss = epoch_loss / len(X)
            print(f"Epoch {epoch + 1}/{epochs}  loss={final_loss:.4f}")
            mlflow.log_metric("train_loss", final_loss, step=epoch)

        model_path = out_dir / MODEL_FILENAME
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": len(LSTM_SEQUENCE_FEATURES),
                "sequence_window": SEQUENCE_WINDOW,
                "feature_columns": list(LSTM_SEQUENCE_FEATURES),
            },
            model_path,
        )
        print(f"Saved model → {model_path}")

        return {
            "model": "lstm",
            "metric_name": "final_loss",
            "metric_value": final_loss,
            "path": str(model_path),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Behavior Agent LSTM sequence model")
    parser.add_argument("--feature-table", type=Path, default=DEFAULT_FEATURE_TABLE)
    parser.add_argument("--output-dir", type=Path, default=MODELS_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    train_lstm(args.feature_table, args.output_dir, epochs=args.epochs)


if __name__ == "__main__":
    main()
