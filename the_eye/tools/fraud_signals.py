"""
Fraud signal detectors for The Eye.

Every detector reads from enriched_transactions.csv (produced by run_feature_engineering)
when it exists, falling back to the raw transactions.csv otherwise.  Using precomputed
features makes each detector faster and more accurate.

Detectors
---------
detect_location_anomalies      – GPS distance between user and transaction city > 100 km
detect_withdrawal_anomalies    – cash withdrawal in a city never visited per GPS history
detect_amount_anomalies        – transaction amount > 2× monthly salary
detect_temporal_anomalies      – night-window (00–05) or rapid-fire (< 5 min)
detect_phishing_victims        – sender received phishing / suspicious-domain comms
detect_new_recipient_anomalies – first-ever transfer to a new IBAN, amount > 1× salary  [NEW]
detect_iban_country_anomalies  – sender and recipient IBAN countries differ             [NEW]
detect_velocity_burst          – ≥ 2 transactions by the same sender in 60 minutes      [NEW]
"""

import json
import math

import pandas as pd
from pathlib import Path

from the_eye import config


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _load_enriched() -> pd.DataFrame:
    """Load enriched_transactions.csv if it exists, else raw transactions.csv."""
    if config.WORKDIR_PATH:
        enriched = Path(config.WORKDIR_PATH) / "enriched_transactions.csv"
        if enriched.exists():
            return pd.read_csv(enriched)
    return pd.read_csv(Path(config.DATASET_PATH) / "transactions.csv")


def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6_371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Detector 1 ────────────────────────────────────────────────────────────────

def detect_location_anomalies() -> str:
    """Detect in-person payments where the sender's GPS biotag was more than 100 km
    away from the transaction city at the time of the transaction.

    Uses the precomputed gps_distance_to_tx_km column from enriched_transactions.csv
    when available.  Falls back to haversine-vs-residence when not.

    Returns:
        JSON array. Each element:
        - "transaction_id": UUID of the flagged transaction
        - "reason": distance in km between GPS ping and transaction city

        Example:
        [{"transaction_id": "43be5588-...", "reason": "GPS mismatch: 3806km between user location and transaction city"}]
        Returns [] if no anomalies found.

    When to call: always call first — GPS mismatch > 500 km is near-definitive fraud evidence.
    """
    df = _load_enriched()
    suspicious = []

    if "gps_distance_to_tx_km" in df.columns:
        in_person = df[df["transaction_type"] == "in-person payment"].copy()
        flagged   = in_person[in_person["gps_distance_to_tx_km"] > 100].dropna(
            subset=["gps_distance_to_tx_km"]
        )
        for _, row in flagged.iterrows():
            suspicious.append({
                "transaction_id": row["transaction_id"],
                "reason": (
                    f"GPS mismatch: {row['gps_distance_to_tx_km']:.0f}km between "
                    f"user GPS location and transaction city"
                ),
            })
    else:
        # Fallback: compare GPS ping to residence (original logic)
        import json as _json
        base = Path(config.DATASET_PATH)
        with open(base / "users.json")     as f: users     = _json.load(f)
        with open(base / "locations.json") as f: locations = _json.load(f)

        user_by_iban  = {u.get("iban", ""): u for u in users}
        loc_by_biotag: dict[str, list] = {}
        for loc in locations:
            loc_by_biotag.setdefault(loc.get("biotag", ""), []).append(loc)

        in_person = df[df["transaction_type"] == "in-person payment"]
        for _, row in in_person.iterrows():
            sender_iban = row.get("sender_iban", "")
            user        = user_by_iban.get(str(sender_iban) if pd.notna(sender_iban) else "")
            if not user:
                continue
            user_locs = loc_by_biotag.get(str(row.get("sender_id", "")), [])
            if not user_locs:
                continue
            res = user.get("residence", {})
            if not res.get("lat") or not res.get("lng"):
                continue
            closest = min(
                user_locs,
                key=lambda l: abs(pd.Timestamp(l["timestamp"]) - pd.Timestamp(row["timestamp"]))
                if "timestamp" in l else float("inf"),
                default=None,
            )
            if closest and "lat" in closest and "lng" in closest:
                dist = _haversine(
                    float(res["lat"]), float(res["lng"]),
                    float(closest["lat"]), float(closest["lng"]),
                )
                if dist > 100:
                    suspicious.append({
                        "transaction_id": row["transaction_id"],
                        "reason": f"GPS mismatch: {dist:.0f}km between transaction and user location",
                    })

    return json.dumps(suspicious, indent=2)


# ── Detector 2 ────────────────────────────────────────────────────────────────

def detect_withdrawal_anomalies() -> str:
    """Detect cash withdrawals in a city the user has never visited per their GPS history.

    Uses the precomputed is_new_city_tx column from enriched_transactions.csv when available.
    Falls back to home-city string matching when not.

    Returns:
        JSON array. Each element:
        - "transaction_id": UUID of the flagged transaction
        - "reason": withdrawal city vs home city

        Returns [] if all withdrawals match a known location.

    When to call: always call — a withdrawal in an unknown city is a high-confidence signal.
    """
    df = _load_enriched()
    suspicious = []
    withdrawals = df[df["transaction_type"] == "withdrawal"]

    if "is_new_city_tx" in df.columns:
        flagged = withdrawals[withdrawals["is_new_city_tx"] == True]
        for _, row in flagged.iterrows():
            tx_loc = str(row.get("location", "")) if pd.notna(row.get("location")) else "unknown"
            suspicious.append({
                "transaction_id": row["transaction_id"],
                "reason": f"Withdrawal in '{tx_loc}' — city not in sender's GPS location history",
            })
    else:
        # Fallback: home-city string matching (original logic)
        import json as _json
        with open(Path(config.DATASET_PATH) / "users.json") as f:
            users = _json.load(f)
        user_by_iban = {u.get("iban", ""): u for u in users}

        for _, row in withdrawals.iterrows():
            sender_iban = str(row.get("sender_iban", ""))
            user        = user_by_iban.get(sender_iban)
            if not user:
                for u in users:
                    if str(row.get("sender_id", "")) in str(u):
                        user = u
                        break
            if not user:
                continue
            home_city = user.get("residence", {}).get("city", "")
            location  = str(row.get("location", ""))
            if home_city and location and home_city.lower() not in location.lower():
                suspicious.append({
                    "transaction_id": row["transaction_id"],
                    "reason": (
                        f"Withdrawal in '{location.split(' - ')[0].strip()}' "
                        f"while user residence is '{home_city}'"
                    ),
                })

    return json.dumps(suspicious, indent=2)


# ── Detector 3 ────────────────────────────────────────────────────────────────

def detect_amount_anomalies() -> str:
    """Detect transactions whose amount exceeds 2× the sender's monthly salary.

    Uses the precomputed amount_vs_salary_ratio column from enriched_transactions.csv
    when available.  Falls back to live salary lookup when not.

    Returns:
        JSON array. Each element:
        - "transaction_id": UUID of the flagged transaction
        - "reason": amount, monthly salary, and multiplier

        Returns [] if no anomalies found.

    When to call: always call — combine with temporal and GPS signals for compound detection.
    """
    df = _load_enriched()
    suspicious = []

    if "amount_vs_salary_ratio" in df.columns:
        flagged = df[df["amount_vs_salary_ratio"] > 2.0].dropna(subset=["amount_vs_salary_ratio"])
        for _, row in flagged.iterrows():
            ratio = row["amount_vs_salary_ratio"]
            ms    = row.get("monthly_salary") or (row["amount"] / ratio)
            suspicious.append({
                "transaction_id": row["transaction_id"],
                "reason": (
                    f"Amount {row['amount']:.2f} is {ratio:.1f}× monthly salary "
                    f"({ms:.2f})"
                ),
            })
    else:
        # Fallback: live salary lookup (original logic)
        import json as _json
        with open(Path(config.DATASET_PATH) / "users.json") as f:
            users = _json.load(f)
        salary_map = {u.get("iban", ""): u.get("salary", 0) for u in users}

        for _, row in df.iterrows():
            sender_iban    = row.get("sender_iban", "")
            annual_salary  = salary_map.get(str(sender_iban) if pd.notna(sender_iban) else "", 0)
            if annual_salary and annual_salary > 0:
                monthly_salary = annual_salary / 12
                ratio          = row["amount"] / monthly_salary
                if ratio > 2.0:
                    suspicious.append({
                        "transaction_id": row["transaction_id"],
                        "reason": (
                            f"Amount {row['amount']:.2f} is {ratio:.1f}× monthly salary "
                            f"({monthly_salary:.2f})"
                        ),
                    })

    return json.dumps(suspicious, indent=2)


# ── Detector 4 ────────────────────────────────────────────────────────────────

def detect_temporal_anomalies() -> str:
    """Detect transactions at unusual hours (00:00–05:59) or in rapid succession (< 5 min).

    Uses the precomputed is_unusual_hour and time_since_last_tx_seconds columns
    from enriched_transactions.csv when available.

    Returns:
        JSON array. Each element:
        - "transaction_id": UUID of the flagged transaction
        - "reason": hour for night-window; elapsed seconds for rapid-fire

        Returns [] if no anomalies found.

    When to call: always call — night-window alone is medium confidence; combined with
    phishing, GPS mismatch, or withdrawal anomaly it becomes high confidence.
    """
    df = _load_enriched()
    suspicious = []

    if "is_unusual_hour" in df.columns:
        for _, row in df[df["is_unusual_hour"] == True].iterrows():
            hour = int(row.get("tx_hour", pd.Timestamp(row["timestamp"]).hour))
            suspicious.append({
                "transaction_id": row["transaction_id"],
                "reason": f"Transaction at unusual hour ({hour:02d}:00)",
            })
        rapid = df[df["time_since_last_tx_seconds"] < 300].dropna(
            subset=["time_since_last_tx_seconds"]
        )
        for _, row in rapid.iterrows():
            suspicious.append({
                "transaction_id": row["transaction_id"],
                "reason": (
                    f"Rapid transaction: {row['time_since_last_tx_seconds']:.0f}s "
                    f"after previous by same sender"
                ),
            })
    else:
        # Fallback: compute on the fly (original logic)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"]      = df["timestamp"].dt.hour
        for _, row in df[df["hour"].between(0, 5)].iterrows():
            suspicious.append({
                "transaction_id": row["transaction_id"],
                "reason": f"Transaction at unusual hour ({row['hour']:02d}:00)",
            })
        df_sorted               = df.sort_values(["sender_id", "timestamp"])
        df_sorted["prev_time"]  = df_sorted.groupby("sender_id")["timestamp"].shift(1)
        df_sorted["secs_since"] = (
            df_sorted["timestamp"] - df_sorted["prev_time"]
        ).dt.total_seconds()
        for _, row in df_sorted[df_sorted["secs_since"] < 300].iterrows():
            suspicious.append({
                "transaction_id": row["transaction_id"],
                "reason": f"Rapid transaction: {row['secs_since']:.0f}s after previous by same sender",
            })

    return json.dumps(suspicious, indent=2)


# ── Detector 5 ────────────────────────────────────────────────────────────────

def detect_phishing_victims() -> str:
    """Identify transactions from senders whose communications contain phishing content.

    Uses precomputed phishing_in_comms and suspicious_domain_in_comms columns from
    enriched_transactions.csv.  Returns transaction IDs (not raw snippets), so
    pattern_agent can correlate directly without LLM guesswork.

    Falls back to raw keyword scanning of sms.json / mails.json when the enriched
    CSV is not available (returns snippets in that case).

    Returns:
        JSON array. Each element:
        - "transaction_id": UUID of the flagged transaction
        - "reason": which phishing signals were found in the sender's comms

        Returns [] if no phishing content detected.

    When to call: always call — a phishing signal combined with any other anomaly
    raises confidence to "high".
    """
    df = _load_enriched()
    suspicious = []

    if "phishing_in_comms" in df.columns:
        mask = (
            (df["phishing_in_comms"] == True)
            | (df.get("suspicious_domain_in_comms", pd.Series(False, index=df.index)) == True)
        )
        for _, row in df[mask].iterrows():
            signals = []
            if row.get("phishing_in_comms"):
                signals.append("phishing_keywords")
            if row.get("suspicious_domain_in_comms"):
                signals.append("suspicious_domain")
            suspicious.append({
                "transaction_id": row["transaction_id"],
                "reason": f"Sender has phishing signals in communications: {', '.join(signals)}",
            })
    else:
        # Fallback: raw keyword scanning (original logic — returns snippets)
        import json as _json
        base = Path(config.DATASET_PATH)
        with open(base / "sms.json")   as f: sms_list  = _json.load(f)
        with open(base / "mails.json") as f: mail_list = _json.load(f)
        phishing_keywords = [
            "click here", "verify your account", "suspended", "urgent action",
            "confirm your", "password", "login immediately", "clicca qui", "verifica",
            "account blocked", "security alert",
        ]
        for item in sms_list + mail_list:
            text    = str(item).lower()
            matched = [kw for kw in phishing_keywords if kw in text]
            if matched:
                suspicious.append({"keywords": matched, "snippet": str(item)[:300]})

    return json.dumps(suspicious[:50], indent=2)


# ── Detector 6 (NEW) ──────────────────────────────────────────────────────────

def detect_new_recipient_anomalies() -> str:
    """Detect high-value transfers to a recipient the sender has never transacted with before.

    Uses the precomputed is_new_recipient and amount_vs_salary_ratio columns.
    Flags transfers where is_new_recipient is True AND amount_vs_salary_ratio > 1.0.
    A large transfer to an unknown recipient is a strong account-takeover signal.

    Returns:
        JSON array. Each element:
        - "transaction_id": UUID of the flagged transaction
        - "reason": recipient IBAN and salary-ratio context

        Returns [] if no anomalies found.

    When to call: always call — combine with phishing and temporal signals.
    """
    df = _load_enriched()
    suspicious = []

    if "is_new_recipient" not in df.columns:
        return json.dumps([], indent=2)

    mask = df["is_new_recipient"] == True
    if "amount_vs_salary_ratio" in df.columns:
        mask = mask & (df["amount_vs_salary_ratio"] > 1.0)

    for _, row in df[mask].iterrows():
        recipient = (
            str(row.get("recipient_iban", ""))
            if pd.notna(row.get("recipient_iban"))
            else str(row.get("recipient_id", "unknown"))
        )
        ratio_str = (
            f" ({row['amount_vs_salary_ratio']:.1f}× salary)"
            if pd.notna(row.get("amount_vs_salary_ratio"))
            else ""
        )
        suspicious.append({
            "transaction_id": row["transaction_id"],
            "reason": (
                f"First-ever transfer to new recipient {recipient}; "
                f"amount {row['amount']:.2f}{ratio_str}"
            ),
        })

    return json.dumps(suspicious, indent=2)


# ── Detector 7 (NEW) ──────────────────────────────────────────────────────────

def detect_iban_country_anomalies() -> str:
    """Detect transactions where the sender and recipient IBAN countries differ.

    Uses the precomputed iban_country_mismatch column.  Cross-border IBAN transfers
    are not inherently fraudulent, but combined with other signals they are significant.

    Returns:
        JSON array. Each element:
        - "transaction_id": UUID of the flagged transaction
        - "reason": sender and recipient IBAN countries

        Returns [] if no mismatches found.

    When to call: always call — use as a supporting signal, not a standalone flag.
    """
    df = _load_enriched()
    suspicious = []

    if "iban_country_mismatch" not in df.columns:
        return json.dumps([], indent=2)

    for _, row in df[df["iban_country_mismatch"] == True].iterrows():
        s_country = row.get("sender_iban_country", "?")
        r_country = row.get("recipient_iban_country", "?")
        suspicious.append({
            "transaction_id": row["transaction_id"],
            "reason": (
                f"IBAN country mismatch: sender country {s_country} → "
                f"recipient country {r_country}"
            ),
        })

    return json.dumps(suspicious, indent=2)


# ── Detector 8 (NEW) ──────────────────────────────────────────────────────────

def detect_velocity_burst() -> str:
    """Detect senders who made 2 or more transactions within a 60-minute window.

    Uses the precomputed velocity_burst_count column.  A burst of transactions in a
    short window suggests automated fraud scripts or simultaneous device compromise.

    Returns:
        JSON array. Each element:
        - "transaction_id": UUID of the flagged transaction
        - "reason": number of prior transactions in the 60-minute window

        Returns [] if no velocity bursts found.

    When to call: always call — velocity burst alone is medium confidence; combined
    with phishing or GPS mismatch it becomes high confidence.
    """
    df = _load_enriched()
    suspicious = []

    if "velocity_burst_count" not in df.columns:
        return json.dumps([], indent=2)

    for _, row in df[df["velocity_burst_count"] >= 2].iterrows():
        suspicious.append({
            "transaction_id": row["transaction_id"],
            "reason": (
                f"Velocity burst: {int(row['velocity_burst_count'])} prior transactions "
                f"by same sender in the preceding 60 minutes"
            ),
        })

    return json.dumps(suspicious, indent=2)
