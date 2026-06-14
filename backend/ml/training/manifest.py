"""Write a manifest of trained model artifacts after a training run."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ml.mlflow.config import MODELS_OUTPUT_DIR

ARTIFACT_FILES: tuple[str, ...] = (
    "feature_columns.json",
    "xgboost_model.pkl",
    "isolation_forest_model.pkl",
    "lstm_model.pt",
    "meta_learner_model.pkl",
)


def save_training_manifest(
    results: list[dict[str, float | str]],
    output_dir: Path | None = None,
) -> Path:
    """Persist run summary and list of expected artifact files."""
    out_dir = output_dir or MODELS_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = []
    for name in ARTIFACT_FILES:
        path = out_dir / name
        artifacts.append(
            {
                "file": name,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )

    payload = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_dir),
        "models": results,
        "artifacts": artifacts,
    }
    manifest_path = out_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path
