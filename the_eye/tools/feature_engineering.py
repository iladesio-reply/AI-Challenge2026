"""
Feature engineering tool for The Eye fraud detection system.

Loads all five dataset sources (transactions, users, GPS locations, SMS, email),
computes 14 behavioural / geospatial / communication features per transaction,
and saves enriched_transactions.csv to WORKDIR_PATH.

Computed feature columns
------------------------
amount_vs_salary_ratio      – tx.amount / (annual_salary / 12)
time_since_last_tx_seconds  – seconds since the sender's previous transaction
is_unusual_hour             – True when tx hour is 00:00–05:59
is_new_recipient            – True on first-ever transfer to this recipient IBAN/ID
is_new_recipient_country    – True when the recipient IBAN country is new for this sender
iban_country_mismatch       – True when sender IBAN country ≠ recipient IBAN country
velocity_burst_count        – # transactions by same sender in the 60 min before this tx
gps_distance_to_tx_km       – haversine km between closest GPS ping and tx location city
                              (in-person + withdrawal only; None otherwise)
is_new_city_tx              – True when tx location city absent from sender GPS history
                              (in-person + withdrawal only)
has_impossible_travel       – True when sender GPS history contains a transition
                              implying speed > 1 500 km/h (faster than any aircraft)
phishing_in_comms           – True when phishing keywords found in sender's comms
suspicious_domain_in_comms  – True when lookalike domains found in sender's comms
urgency_keywords_count      – # urgency keywords in sender's comms
payment_link_in_comms       – True when payment / invoice links found in sender's emails
"""

import json
import math
import re

import pandas as pd
from pathlib import Path

from the_eye import config

# ── Keyword / pattern constants ────────────────────────────────────────────────

_PHISHING_KW = [
    "click here", "verify your account", "suspended", "urgent action",
    "confirm your", "password", "login immediately", "clicca qui", "verifica",
    "account blocked", "security alert", "verify now", "urgent",
]

_SUSPICIOUS_DOMAIN_RE = [
    r"paypa[l1][^a-z]",
    r"mirr0r",
    r"micros0ft",
    r"amaz0n",
    r"g00gle",
    r"[a-z]+-secure\.[a-z]{2,4}",
    r"[a-z]+-alert\.[a-z]{2,4}",
    r"[a-z]+1[a-z]*\.(net|com|org)",   # digit substitution in domain
]

_URGENCY_KW = [
    "urgent", "immediate", "act now", "suspended", "verify now",
    "account blocked", "security alert", "within 24 hours", "asap",
]

_PAYMENT_LINK_RE = [
    r"https?://\S*pay\S*",      # URL containing "pay" — covers fake payment portals
    r"https?://\S*/pay",        # URL path ending in /pay
    r"click.{1,30}pay",         # "click here to pay"
    r"pay.{1,30}link",          # "payment link below"
    # NOTE: bare "invoice" / "payment due" / "fattura" removed — too common in
    # legitimate utility and billing emails; they would flag every transaction for
    # users who receive any kind of bill, bloating detect_urgency_signals output.
]

# ── Internal helpers ───────────────────────────────────────────────────────────

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


def _iban_country(iban_val) -> str | None:
    v = str(iban_val).strip() if pd.notna(iban_val) else ""
    return v[:2].upper() if len(v) >= 2 and v[:2].isalpha() else None


def _city_from_location(location_str: str) -> str:
    """Extract normalised city name from a location field like 'Turin - Piazza Castello'."""
    if not location_str:
        return ""
    return location_str.split(" - ")[0].strip().lower()


# ── Main tool ─────────────────────────────────────────────────────────────────

def run_feature_engineering() -> str:
    """Compute 14 fraud-detection features per transaction and save enriched_transactions.csv.

    Reads the raw dataset from config.DATASET_PATH and writes the enriched CSV to
    config.WORKDIR_PATH/enriched_transactions.csv.  All downstream fraud detectors
    automatically prefer the enriched CSV when it exists.

    Returns:
        A plain-text summary of the computed signal counts, used by preprocessing_agent
        as its final response.

    When to call: call this exactly once, at the start of every pipeline run,
    before any fraud detector tools are invoked.
    """
    base    = Path(config.DATASET_PATH)
    workdir = Path(config.WORKDIR_PATH)

    # ── Load raw data ──────────────────────────────────────────────────────────
    df = pd.read_csv(base / "transactions.csv")

    with open(base / "users.json")     as f: users     = json.load(f)
    with open(base / "locations.json") as f: locations = json.load(f)
    with open(base / "sms.json")       as f: sms_list  = json.load(f)
    with open(base / "mails.json")     as f: mail_list = json.load(f)

    # ── Build lookup tables ────────────────────────────────────────────────────
    user_by_iban: dict[str, dict] = {u["iban"]: u for u in users}

    # city name → (lat, lng) derived from the GPS dataset itself — no external API needed
    city_coords: dict[str, tuple[float, float]] = {}
    for loc in locations:
        city = str(loc.get("city", "")).strip()
        if city and "lat" in loc and "lng" in loc:
            city_coords[city.lower()] = (float(loc["lat"]), float(loc["lng"]))

    # GPS history per biotag (= sender_id), sorted chronologically
    loc_by_biotag: dict[str, list[dict]] = {}
    for loc in locations:
        biotag = str(loc.get("biotag", ""))
        if not biotag:
            continue
        try:
            ts = pd.Timestamp(loc["timestamp"])
        except Exception:
            continue
        loc_by_biotag.setdefault(biotag, []).append({
            "ts":   ts,
            "lat":  float(loc["lat"]),
            "lng":  float(loc["lng"]),
            "city": str(loc.get("city", "")).strip().lower(),
        })
    for biotag in loc_by_biotag:
        loc_by_biotag[biotag].sort(key=lambda p: p["ts"])

    # Set of GPS cities visited per user
    user_cities_visited: dict[str, set[str]] = {
        biotag: {p["city"] for p in pings if p["city"]}
        for biotag, pings in loc_by_biotag.items()
    }

    # ── Sort transactions (required for sequential features) ──────────────────
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["sender_id", "timestamp"]).reset_index(drop=True)

    # sender_id → sender_iban (first occurrence wins)
    sender_to_iban: dict[str, str] = {}
    for _, row in df.iterrows():
        sid   = str(row["sender_id"])
        siban = str(row.get("sender_iban", "")) if pd.notna(row.get("sender_iban")) else ""
        if siban and sid not in sender_to_iban:
            sender_to_iban[sid] = siban

    # ── Feature 1: amount_vs_salary_ratio ────────────────────────────────────
    monthly_salaries: list[float | None] = []
    for _, row in df.iterrows():
        siban = str(row.get("sender_iban", "")) if pd.notna(row.get("sender_iban")) else ""
        user  = user_by_iban.get(siban)
        sal   = user.get("salary", 0) if user else 0
        monthly_salaries.append((sal / 12) if sal and sal > 0 else None)

    df["monthly_salary"]        = monthly_salaries
    df["amount_vs_salary_ratio"] = df.apply(
        lambda r: round(r["amount"] / r["monthly_salary"], 3) if r["monthly_salary"] else None,
        axis=1,
    )

    # ── Feature 2: time_since_last_tx_seconds ────────────────────────────────
    df["prev_ts"] = df.groupby("sender_id")["timestamp"].shift(1)
    df["time_since_last_tx_seconds"] = (df["timestamp"] - df["prev_ts"]).dt.total_seconds()

    # ── Feature 3: is_unusual_hour ───────────────────────────────────────────
    df["tx_hour"]       = df["timestamp"].dt.hour
    df["is_unusual_hour"] = df["tx_hour"].between(0, 5)

    # ── Feature 4: is_new_recipient ──────────────────────────────────────────
    seen_recip: dict[str, set] = {}
    new_recip_flags: list[bool] = []
    for _, row in df.iterrows():
        sid   = str(row["sender_id"])
        riban = str(row.get("recipient_iban", "")) if pd.notna(row.get("recipient_iban")) else ""
        rid   = str(row.get("recipient_id",   "")) if pd.notna(row.get("recipient_id"))   else ""
        key   = riban or rid
        seen  = seen_recip.setdefault(sid, set())
        new_recip_flags.append(bool(key) and key not in seen)
        seen.add(key)
    df["is_new_recipient"] = new_recip_flags

    # ── Feature 5 & 6: IBAN countries ────────────────────────────────────────
    df["sender_iban_country"]    = df["sender_iban"].apply(_iban_country)
    df["recipient_iban_country"] = df["recipient_iban"].apply(_iban_country)

    seen_countries: dict[str, set] = {}
    new_country_flags: list[bool] = []
    for _, row in df.iterrows():
        sid     = str(row["sender_id"])
        country = row["recipient_iban_country"]
        seen    = seen_countries.setdefault(sid, set())
        if country:
            new_country_flags.append(country not in seen)
            seen.add(country)
        else:
            new_country_flags.append(False)
    df["is_new_recipient_country"] = new_country_flags

    df["iban_country_mismatch"] = (
        df["sender_iban_country"].notna()
        & df["recipient_iban_country"].notna()
        & (df["sender_iban_country"] != df["recipient_iban_country"])
    )

    # ── Feature 7: velocity_burst_count ──────────────────────────────────────
    velocity_counts: list[int] = []
    for _, row in df.iterrows():
        window_start = row["timestamp"] - pd.Timedelta(minutes=60)
        prior = df[
            (df["sender_id"]  == row["sender_id"])
            & (df["timestamp"] >  window_start)
            & (df["timestamp"] <  row["timestamp"])
        ]
        velocity_counts.append(len(prior))
    df["velocity_burst_count"] = velocity_counts

    # ── Feature 8: gps_distance_to_tx_km ─────────────────────────────────────
    # For in-person payments and withdrawals: haversine between the closest GPS
    # ping (by time) and the transaction's city coordinates.
    # City coordinates are sourced from the GPS dataset itself.
    gps_distances: list[float | None] = []
    for _, row in df.iterrows():
        if row["transaction_type"] not in ("in-person payment", "withdrawal"):
            gps_distances.append(None)
            continue
        biotag = str(row["sender_id"])
        pings  = loc_by_biotag.get(biotag, [])
        if not pings:
            gps_distances.append(None)
            continue
        tx_time = row["timestamp"]
        closest = min(pings, key=lambda p: abs(p["ts"] - tx_time))

        tx_loc  = str(row.get("location", "")) if pd.notna(row.get("location")) else ""
        tx_city = _city_from_location(tx_loc)
        if tx_city and tx_city in city_coords:
            tx_lat, tx_lng = city_coords[tx_city]
            dist = _haversine(closest["lat"], closest["lng"], tx_lat, tx_lng)
            gps_distances.append(round(dist, 1))
        else:
            gps_distances.append(None)
    df["gps_distance_to_tx_km"] = gps_distances

    # ── Feature 9: is_new_city_tx ────────────────────────────────────────────
    new_city_flags: list[bool] = []
    for _, row in df.iterrows():
        if row["transaction_type"] not in ("in-person payment", "withdrawal"):
            new_city_flags.append(False)
            continue
        biotag  = str(row["sender_id"])
        visited = user_cities_visited.get(biotag, set())
        tx_loc  = str(row.get("location", "")) if pd.notna(row.get("location")) else ""
        tx_city = _city_from_location(tx_loc)
        new_city_flags.append(bool(tx_city) and tx_city not in visited)
    df["is_new_city_tx"] = new_city_flags

    # ── Feature 10: has_impossible_travel ────────────────────────────────────
    # Flag users whose GPS history contains a transition with speed > 1 500 km/h
    # (no commercial aircraft exceeds this), implying the biotag was cloned / spoofed.
    impossible_users: set[str] = set()
    for biotag, pings in loc_by_biotag.items():
        for i in range(1, len(pings)):
            dt_h = (pings[i]["ts"] - pings[i - 1]["ts"]).total_seconds() / 3600
            if 0 < dt_h < 2:          # only flag transitions shorter than 2 hours
                dist_km = _haversine(
                    pings[i - 1]["lat"], pings[i - 1]["lng"],
                    pings[i]["lat"],     pings[i]["lng"],
                )
                if dist_km / dt_h > 1_500:
                    impossible_users.add(biotag)
                    break
    df["has_impossible_travel"] = df["sender_id"].isin(impossible_users)

    # ── Communication features (11–14) ────────────────────────────────────────
    # Match communications to senders via first/last name search.
    # We build one concatenated lower-case text corpus per sender.
    sender_comms_text: dict[str, str] = {}
    for sender_id, sender_iban_val in sender_to_iban.items():
        user        = user_by_iban.get(sender_iban_val, {})
        name_tokens = [
            t.lower() for t in [user.get("first_name", ""), user.get("last_name", "")]
            if t and len(t) > 2
        ]
        if not name_tokens:
            sender_comms_text[sender_id] = ""
            continue
        texts = []
        for s in sms_list:
            text = str(s.get("sms", ""))
            if any(tok in text.lower() for tok in name_tokens):
                texts.append(text)
        for m in mail_list:
            text = str(m.get("mail", ""))
            if any(tok in text.lower() for tok in name_tokens):
                texts.append(text)
        sender_comms_text[sender_id] = " ".join(texts).lower()

    def _has_phishing(text: str) -> bool:
        return any(kw in text for kw in _PHISHING_KW)

    def _has_suspicious_domain(text: str) -> bool:
        return any(re.search(p, text) for p in _SUSPICIOUS_DOMAIN_RE)

    def _count_urgency(text: str) -> int:
        return sum(1 for kw in _URGENCY_KW if kw in text)

    def _has_payment_link(text: str) -> bool:
        return any(re.search(p, text) for p in _PAYMENT_LINK_RE)

    # Feature 11
    df["phishing_in_comms"] = df["sender_id"].map(
        lambda sid: _has_phishing(sender_comms_text.get(sid, ""))
    )
    # Feature 12
    df["suspicious_domain_in_comms"] = df["sender_id"].map(
        lambda sid: _has_suspicious_domain(sender_comms_text.get(sid, ""))
    )
    # Feature 13
    df["urgency_keywords_count"] = df["sender_id"].map(
        lambda sid: _count_urgency(sender_comms_text.get(sid, ""))
    )
    # Feature 14
    df["payment_link_in_comms"] = df["sender_id"].map(
        lambda sid: _has_payment_link(sender_comms_text.get(sid, ""))
    )

    # ── Save enriched CSV ─────────────────────────────────────────────────────
    enriched_path = workdir / "enriched_transactions.csv"
    df.to_csv(enriched_path, index=False)

    # ── Build summary ─────────────────────────────────────────────────────────
    n = len(df)
    stats = {
        "phishing_in_comms":        int(df["phishing_in_comms"].sum()),
        "suspicious_domain":        int(df["suspicious_domain_in_comms"].sum()),
        "unusual_hour (00-05)":     int(df["is_unusual_hour"].sum()),
        "new_recipient":            int(df["is_new_recipient"].sum()),
        "new_recipient_country":    int(df["is_new_recipient_country"].sum()),
        "iban_country_mismatch":    int(df["iban_country_mismatch"].sum()),
        "velocity_burst (≥2/60m)":  int((df["velocity_burst_count"] >= 2).sum()),
        "gps_dist >100km":          int((df["gps_distance_to_tx_km"] > 100).dropna().sum()),
        "new_city_tx":              int(df["is_new_city_tx"].sum()),
        "impossible_travel_users":  int(df["has_impossible_travel"].sum()),
        "high_amount (>2× salary)": int((df["amount_vs_salary_ratio"] > 2).dropna().sum()),
        "payment_link_in_comms":    int(df["payment_link_in_comms"].sum()),
    }

    lines = [
        f"Preprocessing complete. Enriched dataset ({n} txns) → {enriched_path}",
        "",
        "Signal counts across all transactions:",
    ] + [f"  {k}: {v}" for k, v in stats.items()]

    return "\n".join(lines)
