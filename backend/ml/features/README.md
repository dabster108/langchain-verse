# Feature Engineering Pipeline

Cleans raw CSV/JSON datasets in `backend/datasets/` and writes processed outputs to `backend/datasets_processed/`.

## Scripts

| File | Purpose |
|------|---------|
| `clean_transactions.py` | `clean_transactions(df)` — transaction-level data-quality rules |
| `build_features.py` | `build_feature_table()` — joins all tables and engineers model features |
| `run_pipeline.py` | CLI entrypoint — cleans individual tables + builds `feature_table.csv` |

## Run

From the `backend/` directory:

```bash
python -m ml.features.run_pipeline
```

## Cleaning rules (`clean_transactions`)

| Issue | Rule |
|-------|------|
| `device_id` ~30% null | Keep null; add `has_device_id` |
| `merchant_category_code` ~1.5% null | Fill with `"UNKNOWN"` |
| `terminal_id` / `session_id` null by channel | Keep null; add `has_terminal_id`, `has_session_id` |
| `fx_rate` null for NPR (~96%) | Keep null; add `has_fx_rate` |
| `notes` ~78% null | Drop column; add `has_notes` |
| Malformed IPs (`127.0.0.1`, `0.0.0.0`, `10.0.0.1`) | `is_malformed_ip=True` |
| `timestamp` NPT format | Parse to `datetime64`; ATM UTC edge cases not auto-fixed (see TODO in code) |
| `amount_npr` mixed decimals | Round to 2 dp |
| Duplicate bursts | `is_possible_duplicate` when same `account_id` + `amount_npr` within ±5 s (flag only, never drop) |

`clean_geo_events` and `clean_otp_logs` apply the same IP flag / datetime parsing to their respective tables.

## Engineered features

### From transactions (after cleaning)

| Feature | Description |
|---------|-------------|
| `hour_of_day` | Hour 0–23 (NPT, parsed as-is) |
| `day_of_week` | Monday=0 … Sunday=6 |
| `is_weekend` | Saturday or Sunday |
| `is_night` | 22:00–05:00 NPT |
| `type_encoded` | Integer encoding of `txn_type` (mapping in `txn_type_encoding.json`) |
| `amount_ratio` | `amount_npr / cust_avg_monthly_txn_value_npr` |
| `is_structuring_amount` | Amount within ±600 of 9 999 / 49 999 / 99 999 NPR |
| `is_fraud_merchant` | `counterparty_id` ∈ {MERCH-8812, MERCH-9041, MERCH-7712} |
| `currency_*` | One-hot columns for `currency` |
| `channel_*` | One-hot columns for `channel` |
| `auth_method_*` | One-hot columns for `auth_method` |
| `response_code_*` | One-hot columns for `response_code` |

### Joined from `customer_profiles` (prefix `cust_`)

`cust_risk_tier`, `cust_kyc_tier`, `cust_avg_monthly_txn_value_npr`, `cust_avg_monthly_txn_count`, `cust_is_dormant`, `cust_churn_risk_score`

### Joined from `geo_events` (prefix `geo_`)

`geo_latitude`, `geo_longitude`, `geo_is_vpn`, `geo_is_tor`, `geo_is_datacenter`, `geo_velocity_flag`, `geo_km_from_home_district`, `geo_prev_txn_km`, `geo_prev_txn_time_delta_min`, `geo_impossible_travel`, `geo_is_malformed_ip`

### Joined from `velocity_snapshots` (prefix `vel_`)

`vel_txn_count_1m` … `vel_txn_count_7d`, `vel_z_score_amount`, `vel_dormancy_break`, `vel_night_flag`, `vel_new_counterparty_flag`

### Joined from `device_fingerprints` (prefix `dev_`)

`dev_is_rooted_or_jailbroken`, `dev_vpn_detected`, `dev_tor_exit_node`, `dev_biometric_enrolled`, `dev_num_accounts_seen_on_device`, `dev_is_shared_device`, `dev_has_risk_signals`, `dev_risk_signal_count`

### Joined from `otp_logs` (sparse ~2.2%)

`has_otp_log`, `otp_trigger_reason`, `otp_final_decision`, `otp_sim_swap_suspected`

### Joined from `rule_engine_baseline_predictions`

`rule_baseline_decision`, `rule_triggered`, `rule_confidence`

### Label (last column)

`is_fraud` (+ `fraud_type`, `fraud_confidence` for analysis)

## Output files (`datasets_processed/`)

| File | Contents |
|------|----------|
| `feature_table.csv` | Full merged feature matrix for training |
| `transactions_raw_cleaned.csv` | Cleaned transactions |
| `customer_profiles.csv` | Pass-through |
| `geo_events.csv` | Cleaned geo events |
| `velocity_snapshots.csv` | Pass-through |
| `fraud_labels_train.csv` | Pass-through |
| `otp_logs.csv` | Cleaned OTP logs |
| `rule_engine_baseline_predictions.csv` | Pass-through |
| `device_fingerprints.json` | Copied unchanged |
| `account_graph_nodes.csv` | Copied unchanged (graph — not in flat table) |
| `account_graph_edges.csv` | Copied unchanged (graph — not in flat table) |
| `txn_type_encoding.json` | `txn_type` → integer mapping |
