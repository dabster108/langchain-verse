"""Run all offline Behavior Agent + synthesis training scripts in sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.mlflow.config import MODELS_OUTPUT_DIR
from ml.training.train_isolation_forest import train_isolation_forest
from ml.training.train_lstm import train_lstm
from ml.training.train_meta_learner import train_meta_learner
from ml.training.train_xgboost import train_xgboost


def run_all_training(
    feature_table_path: Path | None = None,
    output_dir: Path | None = None,
) -> list[dict[str, float | str]]:
    """Execute xgboost → isolation_forest → lstm → meta_learner."""
    out = output_dir or MODELS_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Offline model training pipeline")
    print("=" * 60)

    results: list[dict[str, float | str]] = []

    print("\n[1/4] Training XGBoost …")
    results.append(train_xgboost(feature_table_path, out))

    print("\n[2/4] Training Isolation Forest …")
    results.append(train_isolation_forest(feature_table_path, out))

    print("\n[3/4] Training LSTM …")
    results.append(train_lstm(feature_table_path, out))

    print("\n[4/4] Training meta-learner …")
    results.append(train_meta_learner(feature_table_path, out))

    print("\n" + "=" * 60)
    print("Training summary")
    print("=" * 60)
    print(f"{'Model':<20} {'Metric':<12} {'Value':<10} Path")
    print("-" * 60)
    for row in results:
        print(
            f"{row['model']:<20} {row['metric_name']:<12} "
            f"{float(row['metric_value']):<10.4f} {row['path']}"
        )
    print("=" * 60)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all offline training scripts")
    parser.add_argument(
        "--feature-table",
        type=Path,
        default=None,
        help="Path to feature_table.csv (default: datasets_processed/feature_table.csv)",
    )
    parser.add_argument("--output-dir", type=Path, default=MODELS_OUTPUT_DIR)
    args = parser.parse_args()
    run_all_training(args.feature_table, args.output_dir)


if __name__ == "__main__":
    main()
