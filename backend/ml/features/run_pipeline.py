"""Entrypoint: clean raw datasets and write processed outputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from ml.features.build_features import build_feature_table
from ml.features.clean_transactions import clean_geo_events, clean_otp_logs, clean_transactions

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = BACKEND_ROOT / "datasets"
DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "datasets_processed"

GRAPH_FILES: tuple[str, ...] = ("account_graph_nodes.csv", "account_graph_edges.csv")


def run_pipeline(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Execute the full cleaning + feature-engineering pipeline."""
    src = data_dir or DEFAULT_DATA_DIR
    out = output_dir or DEFAULT_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading raw data from {src}")
    transactions = pd.read_csv(src / "transactions_raw.csv")
    customers = pd.read_csv(src / "customer_profiles.csv")
    geo = pd.read_csv(src / "geo_events.csv")
    velocity = pd.read_csv(src / "velocity_snapshots.csv")
    labels = pd.read_csv(src / "fraud_labels_train.csv")
    otp = pd.read_csv(src / "otp_logs.csv")
    baseline = pd.read_csv(src / "rule_engine_baseline_predictions.csv")

    print("[2/4] Cleaning individual tables")
    cleaned_txn = clean_transactions(transactions)
    cleaned_geo = clean_geo_events(geo)
    cleaned_otp = clean_otp_logs(otp)

    _save_csv(cleaned_txn, out / "transactions_raw_cleaned.csv")
    _save_csv(customers, out / "customer_profiles.csv")
    _save_csv(cleaned_geo, out / "geo_events.csv")
    _save_csv(velocity, out / "velocity_snapshots.csv")
    _save_csv(labels, out / "fraud_labels_train.csv")
    _save_csv(cleaned_otp, out / "otp_logs.csv")
    _save_csv(baseline, out / "rule_engine_baseline_predictions.csv")

    # Device fingerprints: copy JSON (structure preserved for Neo4j / device lookups).
    shutil.copy2(src / "device_fingerprints.json", out / "device_fingerprints.json")

    # Graph data: copied unchanged (not part of the flat feature table).
    for name in GRAPH_FILES:
        shutil.copy2(src / name, out / name)
        print(f"  Copied graph file (unchanged): {name}")

    print("[3/4] Building feature table")
    feature_table, txn_type_encoding = build_feature_table(data_dir=src)

    print("[4/4] Writing outputs")
    _save_csv(feature_table, out / "feature_table.csv")
    encoding_path = out / "txn_type_encoding.json"
    encoding_path.write_text(json.dumps(txn_type_encoding, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  Saved txn_type_encoding.json ({len(txn_type_encoding)} types)")
    print(f"Done — processed files written to {out}")


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"  Saved {path.name} ({len(df):,} rows)")


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
