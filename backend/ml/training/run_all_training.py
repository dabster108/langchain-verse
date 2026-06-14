"""Run all offline Behavior Agent + synthesis training scripts in sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ml.mlflow.config import MODELS_OUTPUT_DIR
from ml.training.data_utils import DEFAULT_FEATURE_TABLE
from ml.training.manifest import save_training_manifest

TRAINERS: tuple[str, ...] = (
    "ml.training.train_xgboost",
    "ml.training.train_isolation_forest",
    "ml.training.train_lstm",
    "ml.training.train_meta_learner",
)

METRIC_KEYS: dict[str, tuple[str, str]] = {
    "xgboost": ("auroc", "xgboost_model.pkl"),
    "isolation_forest": ("auroc", "isolation_forest_model.pkl"),
    "lstm": ("final_loss", "lstm_model.pt"),
    "meta_learner": ("auroc", "meta_learner_model.pkl"),
}


def _run_module(module: str, feature_table: Path, output_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        module,
        "--feature-table",
        str(feature_table),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)


def run_all_training(
    feature_table_path: Path | None = None,
    output_dir: Path | None = None,
) -> list[dict[str, float | str]]:
    """Execute xgboost → isolation_forest → lstm → meta_learner (separate processes)."""
    out = output_dir or MODELS_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    feature_path = feature_table_path or DEFAULT_FEATURE_TABLE
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {feature_path}\n"
            "Run first: python -m ml.features.run_pipeline"
        )

    print("=" * 60)
    print("Offline model training pipeline")
    print(f"Feature table: {feature_path}")
    print(f"Output dir:    {out}")
    print("=" * 60)

    for idx, module in enumerate(TRAINERS, start=1):
        print(f"\n[{idx}/{len(TRAINERS)}] {module} …")
        _run_module(module, feature_path, out)

    # Build summary from manifest inputs (metrics printed by each script).
    results: list[dict[str, float | str]] = []
    for name, (_, filename) in METRIC_KEYS.items():
        results.append(
            {
                "model": name,
                "metric_name": METRIC_KEYS[name][0],
                "metric_value": 0.0,
                "path": str(out / filename),
            }
        )

    manifest = save_training_manifest(results, out)

    print("\n" + "=" * 60)
    print("Training complete — artifacts in ml/models/")
    print("=" * 60)
    for artifact in (out / "training_manifest.json",):
        if artifact.exists():
            print(f"Manifest → {artifact}")
    for name in ("feature_columns.json", "xgboost_model.pkl", "isolation_forest_model.pkl", "lstm_model.pt", "meta_learner_model.pkl"):
        path = out / name
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {name}")
    print("=" * 60)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all offline training scripts")
    parser.add_argument("--feature-table", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=MODELS_OUTPUT_DIR)
    args = parser.parse_args()
    run_all_training(args.feature_table, args.output_dir)


if __name__ == "__main__":
    main()
