"""Assemble the model-ready feature table from raw backend datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.features.clean_transactions import clean_geo_events, clean_otp_logs, clean_transactions

BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = BACKEND_ROOT / "datasets"

FRAUD_MERCHANT_IDS: frozenset[str] = frozenset({"MERCH-8812", "MERCH-9041", "MERCH-7712"})
STRUCTURING_THRESHOLDS: tuple[int, ...] = (9_999, 49_999, 99_999)
STRUCTURING_TOLERANCE: float = 600.0

# Columns pulled from joined tables (documented in README).
GEO_FEATURE_COLS: tuple[str, ...] = (
    "latitude",
    "longitude",
    "is_vpn",
    "is_tor",
    "is_datacenter",
    "velocity_flag",
    "km_from_home_district",
    "prev_txn_km",
    "prev_txn_time_delta_min",
    "impossible_travel",
)

VELOCITY_FEATURE_COLS: tuple[str, ...] = (
    "txn_count_1m",
    "txn_count_5m",
    "txn_count_15m",
    "txn_count_1h",
    "txn_count_24h",
    "txn_count_7d",
    "z_score_amount",
    "dormancy_break",
    "night_flag",
    "new_counterparty_flag",
)

CUSTOMER_FEATURE_COLS: tuple[str, ...] = (
    "risk_tier",
    "kyc_tier",
    "avg_monthly_txn_value_npr",
    "avg_monthly_txn_count",
    "is_dormant",
    "churn_risk_score",
)

DEVICE_FEATURE_COLS: tuple[str, ...] = (
    "is_rooted_or_jailbroken",
    "vpn_detected",
    "tor_exit_node",
    "biometric_enrolled",
    "num_accounts_seen_on_device",
    "is_shared_device",
)


def build_feature_table(data_dir: Path | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load, clean, join, and engineer features for fraud model training.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, int]]
        Feature table with ``is_fraud`` as the final column, and the
        ``txn_type`` integer-encoding mapping.
    """
    root = data_dir or DEFAULT_DATA_DIR

    transactions = pd.read_csv(root / "transactions_raw.csv")
    customers = pd.read_csv(root / "customer_profiles.csv")
    geo = pd.read_csv(root / "geo_events.csv")
    velocity = pd.read_csv(root / "velocity_snapshots.csv")
    labels = pd.read_csv(root / "fraud_labels_train.csv")
    otp = pd.read_csv(root / "otp_logs.csv")
    baseline = pd.read_csv(root / "rule_engine_baseline_predictions.csv")
    devices = _load_device_fingerprints(root / "device_fingerprints.json")

    cleaned_txn = clean_transactions(transactions)

    features, txn_type_encoding = _engineer_transaction_features(cleaned_txn, customers)
    features = _join_geo(features, clean_geo_events(geo))
    features = _join_velocity(features, velocity)
    features = _join_devices(features, devices)
    features = _join_otp(features, clean_otp_logs(otp))
    features = _join_baseline(features, baseline)
    features = _join_labels(features, labels)
    features = _one_hot_encode(features)

    # Label column last.
    if "is_fraud" in features.columns:
        label = features.pop("is_fraud")
        features["is_fraud"] = label

    _print_summary(features)
    return features, txn_type_encoding


def _load_device_fingerprints(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as fh:
        records = json.load(fh)
    df = pd.DataFrame(records)
    df["has_risk_signals"] = df["risk_signals"].apply(lambda x: bool(x) if isinstance(x, list) else False)
    df["risk_signal_count"] = df["risk_signals"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    return df


def _engineer_transaction_features(
    df: pd.DataFrame,
    customers: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Temporal, amount, and merchant features from cleaned transactions."""
    out = df.copy()

    # Temporal features (NPT parsed as-is; see clean_transactions TODO).
    out["hour_of_day"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6])
    out["is_night"] = (out["hour_of_day"] >= 22) | (out["hour_of_day"] < 5)

    # Transaction-type integer encoding (mapping returned separately by caller).
    type_mapping = {t: i for i, t in enumerate(sorted(out["txn_type"].dropna().unique()))}
    out["type_encoded"] = out["txn_type"].map(type_mapping).astype("Int64")

    # Customer-relative amount ratio and profile attributes.
    cust = customers[["account_id", *CUSTOMER_FEATURE_COLS]].copy()
    cust = cust.rename(
        columns={c: f"cust_{c}" if c != "account_id" else c for c in cust.columns if c != "account_id"}
    )
    out = out.merge(cust, on="account_id", how="left")
    denom = out["cust_avg_monthly_txn_value_npr"].replace(0, pd.NA)
    out["amount_ratio"] = (out["amount_npr"] / denom).fillna(0.0)

    # Structuring-pattern amounts (just below reporting thresholds).
    out["is_structuring_amount"] = out["amount_npr"].apply(_is_structuring_amount)

    # Known high-risk merchant counterparties.
    out["is_fraud_merchant"] = out["counterparty_id"].isin(FRAUD_MERCHANT_IDS)

    return out, type_mapping


def _is_structuring_amount(amount: float) -> bool:
    return any(abs(amount - threshold) <= STRUCTURING_TOLERANCE for threshold in STRUCTURING_THRESHOLDS)


def _join_geo(df: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    cols = ["txn_id", *GEO_FEATURE_COLS, "is_malformed_ip"]
    geo_sub = geo[cols].rename(columns={c: f"geo_{c}" for c in GEO_FEATURE_COLS + ("is_malformed_ip",)})
    return df.merge(geo_sub, on="txn_id", how="left")


def _join_velocity(df: pd.DataFrame, velocity: pd.DataFrame) -> pd.DataFrame:
    cols = ["txn_id", *VELOCITY_FEATURE_COLS]
    vel_sub = velocity[cols].rename(columns={c: f"vel_{c}" for c in VELOCITY_FEATURE_COLS})
    return df.merge(vel_sub, on="txn_id", how="left")


def _join_devices(df: pd.DataFrame, devices: pd.DataFrame) -> pd.DataFrame:
    dev_cols = ["device_id", *DEVICE_FEATURE_COLS, "has_risk_signals", "risk_signal_count"]
    dev_sub = devices[dev_cols].rename(
        columns={c: f"dev_{c}" for c in DEVICE_FEATURE_COLS + ("has_risk_signals", "risk_signal_count")}
    )
    return df.merge(dev_sub, on="device_id", how="left")


def _join_otp(df: pd.DataFrame, otp: pd.DataFrame) -> pd.DataFrame:
    otp_sub = otp.groupby("txn_id", as_index=False).first()
    otp_sub["has_otp_log"] = True
    keep = ["txn_id", "has_otp_log", "trigger_reason", "final_decision", "sim_swap_suspected"]
    otp_sub = otp_sub[keep].rename(
        columns={
            "trigger_reason": "otp_trigger_reason",
            "final_decision": "otp_final_decision",
            "sim_swap_suspected": "otp_sim_swap_suspected",
        }
    )
    merged = df.merge(otp_sub, on="txn_id", how="left")
    merged["has_otp_log"] = merged["has_otp_log"].fillna(False).astype(bool)
    return merged


def _join_baseline(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    base_sub = baseline.rename(
        columns={
            "baseline_decision": "rule_baseline_decision",
            "rule_triggered": "rule_triggered",
            "confidence": "rule_confidence",
        }
    )
    return df.merge(base_sub, on="txn_id", how="left")


def _join_labels(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    label_cols = ["txn_id", "is_fraud", "fraud_type", "fraud_confidence"]
    return df.merge(labels[label_cols], on="txn_id", how="left")


def _one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("currency", "channel", "auth_method", "response_code"):
        dummies = pd.get_dummies(out[col], prefix=col, dtype=bool)
        out = pd.concat([out.drop(columns=[col]), dummies], axis=1)
    return out


def _print_summary(df: pd.DataFrame) -> None:
    print("\n=== Feature table summary ===")
    print(f"Rows: {len(df):,}")
    if "is_fraud" in df.columns:
        fraud_rate = df["is_fraud"].astype(bool).mean()
        print(f"Fraud rate: {fraud_rate:.4f} ({df['is_fraud'].astype(bool).sum()} positives)")
    print("\nNull counts (non-zero only):")
    nulls = df.isna().sum()
    for col, count in nulls[nulls > 0].items():
        print(f"  {col}: {count}")
    if nulls.sum() == 0:
        print("  (none)")
    print("\nDtypes:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")

