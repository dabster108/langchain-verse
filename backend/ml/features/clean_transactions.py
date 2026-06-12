"""Cleaning rules for the raw transactions table."""

from __future__ import annotations

import pandas as pd

# Malformed / non-geolocatable IPs called out in the data dictionary (~0.2% of rows).
MALFORMED_IPS: frozenset[str] = frozenset({"127.0.0.1", "0.0.0.0", "10.0.0.1"})

# Nepal-time timestamps use `YYYY-MM-DD HH:MM:SS.mmm` (UTC+5:45).
_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply documented data-quality rules to ``transactions_raw``.

    Each step below cites the issue it addresses in the project data dictionary.

    Parameters
    ----------
    df:
        Raw transactions DataFrame (must not modify the source file on disk).

    Returns
    -------
    pd.DataFrame
        Cleaned copy with derived boolean flags and normalised fields.
    """
    out = df.copy()

    # --- device_id: ~30% null for WEB/branch — keep null, expose presence flag ---
    out["has_device_id"] = out["device_id"].notna()

    # --- merchant_category_code: ~1.5% null — impute sentinel category ---
    out["merchant_category_code"] = out["merchant_category_code"].fillna("UNKNOWN")

    # --- terminal_id / session_id: channel-dependent nulls are expected — no imputation ---
    out["has_terminal_id"] = out["terminal_id"].notna()
    out["has_session_id"] = out["session_id"].notna()

    # --- fx_rate: null for ~96% NPR rows — keep null, add presence flag ---
    out["has_fx_rate"] = out["fx_rate"].notna()

    # --- notes: ~78% null — too sparse for baseline features ---
    out["has_notes"] = out["notes"].notna()
    out = out.drop(columns=["notes"])

    # --- ip_address: flag malformed loopback/private addresses ---
    out["is_malformed_ip"] = out["ip_address"].astype(str).isin(MALFORMED_IPS)

    # --- timestamp: parse Nepal-time strings; ATM UTC edge cases not auto-corrected ---
    # TODO: ~0.9% of ATM records may be logged in UTC instead of NPT — manual review
    #       or domain-specific correction required; we parse values as-is here.
    out["timestamp"] = pd.to_datetime(out["timestamp"], format=_TIMESTAMP_FORMAT, errors="coerce")

    # --- amount_npr: normalise mixed 2 vs 4 decimal precision to 2 dp ---
    out["amount_npr"] = out["amount_npr"].round(2)

    # --- duplicate detection: flag (never drop) near-duplicate bursts ---
    out["is_possible_duplicate"] = _flag_possible_duplicates(out)

    return out


def _flag_possible_duplicates(df: pd.DataFrame) -> pd.Series:
    """Mark rows sharing account_id + amount_npr within ±5 seconds."""
    flags = pd.Series(False, index=df.index)
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        return flags

    work = df.loc[df["timestamp"].notna(), ["account_id", "amount_npr", "timestamp"]].copy()
    work = work.sort_values("timestamp")

    for (_account, _amount), group in work.groupby(["account_id", "amount_npr"], sort=False):
        if len(group) < 2:
            continue
        times = group["timestamp"].astype("int64") // 10**9  # epoch seconds
        indices = group.index.to_list()
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if abs(int(times.iloc[i]) - int(times.iloc[j])) <= 5:
                    flags.loc[indices[i]] = True
                    flags.loc[indices[j]] = True
    return flags


def clean_geo_events(df: pd.DataFrame) -> pd.DataFrame:
    """Light cleaning for geo_events (malformed IP flag + timestamp parse)."""
    out = df.copy()
    out["is_malformed_ip"] = out["ip_address"].astype(str).isin(MALFORMED_IPS)
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], format=_TIMESTAMP_FORMAT, errors="coerce")
    return out


def clean_otp_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Parse datetime columns in sparse OTP logs."""
    out = df.copy()
    datetime_cols = [c for c in out.columns if c.endswith("_at")]
    for col in datetime_cols:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out
