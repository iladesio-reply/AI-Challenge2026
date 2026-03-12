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
    """Detect in-person payments where the sender's GPS biotag was > 100 km from the
    transaction city at the time of the transaction.

    This detector catches card-present fraud: a physical card (or cloned card) used at a
    point-of-sale while the legitimate account holder's GPS biotag was recorded at a
    different location.  The Mirror Hacker uses cloned payment devices at distant ATMs
    and shops while the victim's biotag shows them at home.

    Algorithm:
        Reads the precomputed gps_distance_to_tx_km column from enriched_transactions.csv.
        Selects rows where transaction_type == "in-person payment" AND
        gps_distance_to_tx_km > 100.
        Fallback (no enriched CSV): computes haversine distance between the sender's closest
        GPS ping and their registered residence in users.json — less accurate because it
        compares to home address rather than to the actual transaction city.

    Key precomputed column: gps_distance_to_tx_km
        Computed by run_feature_engineering as the haversine distance (km) between:
        - The sender's GPS biotag ping with the timestamp closest to the transaction time.
        - The centroid lat/lng of the transaction location city (from locations.json).
        Only populated for in-person payments and withdrawals; None for all other types.

    Signal produced: "gps_mismatch"

    Confidence rules (applied by pattern_agent / reflection_agent):
        - gps_distance_to_tx_km > 500 km → "high" alone (physically impossible same-day travel)
        - gps_distance_to_tx_km 100–500 km → "medium" alone;
          upgrades to "high" when combined with any other signal
          (e.g. temporal_anomaly, phishing_exposure, withdrawal_anomaly)

    Returns:
        JSON array string.  Each element contains:
        - "transaction_id" (str): UUID of the flagged in-person payment.
        - "reason" (str): human-readable distance description, e.g.
          "GPS mismatch: 3806km between user GPS location and transaction city"
        Returns "[]" if no in-person payment exceeds the 100 km threshold.

    Example output:
        [
          {
            "transaction_id": "43be5588-2cfb-47c1-a8aa-aeb8d2f38aff",
            "reason": "GPS mismatch: 3806km between user GPS location and transaction city"
          }
        ]

    Legitimate transactions NOT flagged:
        - Transfers, e-commerce, card-not-present, direct debits
          (transaction_type ≠ "in-person payment" are excluded entirely)
        - In-person payments within 100 km of the nearest GPS ping

    When to call:
        Always — call this first in the detection sequence.  GPS mismatch > 500 km is
        near-definitive fraud evidence.  Its output should be merged with
        detect_temporal_anomalies and detect_phishing_victims results for compound scoring.
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
    """Detect cash withdrawals made in a city the sender has never visited per their GPS history.

    ATM withdrawals in unfamiliar cities are a strong fraud signal: the Mirror Hacker uses
    cloned cards or compromised credentials to drain accounts at ATMs far from the victim's
    regular territory.  This detector cross-references the withdrawal location against the
    sender's complete GPS biotag history — if the city has never appeared in the history,
    the withdrawal is anomalous.

    Algorithm:
        Reads the precomputed is_new_city_tx column from enriched_transactions.csv.
        Selects rows where transaction_type == "withdrawal" AND is_new_city_tx == True.
        Fallback (no enriched CSV): compares the withdrawal location string against the
        sender's home city from users.json — less reliable because it only checks one city
        instead of the full GPS travel history.

    Key precomputed column: is_new_city_tx
        Computed by run_feature_engineering: True when the transaction location city (parsed
        from the "location" field before the " - " separator) does not appear in the set of
        all city names recorded in the sender's GPS biotag history.
        Only populated for in-person payments and withdrawals; False otherwise.

    Signal produced: "withdrawal_anomaly"

    Confidence rules (applied by pattern_agent / reflection_agent):
        - withdrawal_anomaly alone → "medium" (consistent with travel, but suspicious)
        - withdrawal_anomaly + gps_mismatch or temporal_anomaly → "high"
        - withdrawal_anomaly + phishing_exposure → "high" (likely account takeover)

    Returns:
        JSON array string.  Each element contains:
        - "transaction_id" (str): UUID of the flagged withdrawal.
        - "reason" (str): location string and explanation, e.g.
          "Withdrawal in 'Lyon - ATM Central' — city not in sender's GPS location history"
        Returns "[]" if all withdrawals occur in cities present in the sender's GPS history.

    Example output:
        [
          {
            "transaction_id": "7ac3e100-...",
            "reason": "Withdrawal in 'Lyon - ATM Central' — city not in sender's GPS location history"
          }
        ]

    Legitimate transactions NOT flagged:
        - Withdrawals in cities the sender has previously visited (per GPS data)
        - Transfers, e-commerce, in-person payments (only "withdrawal" type is evaluated)

    When to call:
        Always — an ATM withdrawal in a never-visited city is a reliable fraud indicator.
        Combine with detect_location_anomalies and detect_temporal_anomalies results
        to confirm high-confidence compound signals.
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
    """Detect transactions whose amount exceeds 2× the sender's estimated monthly salary.

    Fraudsters often drain accounts through a single large transfer or a rapid sequence
    of high-value transactions.  By comparing each transaction amount to the sender's
    monthly salary (annual salary ÷ 12), this detector flags economically implausible
    payments that the account holder is unlikely to make legitimately.  The Mirror Hacker
    varies amounts to stay just above salary thresholds, so both 2× and 10× thresholds
    are tracked for graduated confidence.

    Algorithm:
        Reads the precomputed amount_vs_salary_ratio column from enriched_transactions.csv.
        Flags all rows where amount_vs_salary_ratio > 2.0 (and the ratio is not None).
        Fallback (no enriched CSV): looks up the sender's annual salary from users.json
        by sender_iban, computes monthly salary = annual / 12, then applies the same
        2× threshold.

    Key precomputed column: amount_vs_salary_ratio
        = transaction.amount / (user.salary / 12).
        None when the sender's IBAN is not matched in users.json.
        Also used by detect_new_recipient_anomalies for the 1× threshold there.

    Signal produced: "amount_anomaly"

    Confidence rules (applied by pattern_agent / reflection_agent):
        - amount_vs_salary_ratio > 10× → "high" alone (extreme outlier)
        - amount_vs_salary_ratio 3–10× → "medium" alone;
          upgrades to "high" with any corroborating signal
        - amount_vs_salary_ratio 2–3× → "low" unless combined with another signal
        - reflection_agent will force "high" if details mention "> 10×"

    Returns:
        JSON array string.  Each element contains:
        - "transaction_id" (str): UUID of the flagged transaction.
        - "reason" (str): amount, multiplier, and monthly salary reference, e.g.
          "Amount 4500.00 is 3.2× monthly salary (1406.25)"
        Returns "[]" if no transaction exceeds 2× monthly salary.

    Example output:
        [
          {
            "transaction_id": "ae4125db-5912-45e7-b13e-a3a33609ddf1",
            "reason": "Amount 4500.00 is 3.2× monthly salary (1406.25)"
          }
        ]

    Legitimate transactions NOT flagged (decision_agent filters these after detection):
        - Transfers with description "Salary payment" from a sender whose ID starts with "EMP"
          (MirrorPay employer payroll — large amounts are expected)
        - Transfers with description "Rent payment" to known property management entities
        - Low-amount direct debits consistent with utility or insurance billing cycles

    When to call:
        Always — combine its output with detect_temporal_anomalies and
        detect_phishing_victims to escalate medium-confidence flags to high.
        decision_agent will drop legitimate salary/rent payments automatically.
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
    """Detect transactions at unusual hours (00:00–05:59) or in rapid succession (< 5 min apart).

    This detector covers two distinct temporal fraud patterns used by the Mirror Hacker:
    1. Night-window exploitation: initiating fraudulent transfers at 00–05 when victims
       are asleep and real-time notification apps are silenced.  Victims typically only
       discover the fraud the next morning, giving the hacker hours to move funds onward.
    2. Rapid-fire bursts: multiple transactions within minutes, characteristic of automated
       fraud scripts that exhaust an account balance before the system triggers a block.
       Rapid-fire and velocity_burst often overlap — both should be reported.

    Algorithm:
        Reads is_unusual_hour and time_since_last_tx_seconds from enriched_transactions.csv.
        - Night-window: selects rows where is_unusual_hour == True (hour 0–5 inclusive).
        - Rapid-fire: selects rows where time_since_last_tx_seconds < 300 (i.e. < 5 minutes
          after the same sender's immediately preceding transaction).
        A single transaction may appear in both results if it meets both criteria.
        Fallback (no enriched CSV): computes hour from timestamp, and prev_time via
        groupby shift — same logic but slower, applied on-the-fly.

    Key precomputed columns:
        is_unusual_hour             (bool)   – True for tx hour in 00–05.
        time_since_last_tx_seconds  (float)  – Seconds since same sender's previous tx;
                                               None for a sender's first transaction.
        tx_hour                     (int)    – Hour component of timestamp (0–23).

    Signal produced: "temporal_anomaly"

    Confidence rules (applied by pattern_agent / reflection_agent):
        - temporal_anomaly alone (night-window) → "medium"
        - temporal_anomaly alone (rapid-fire)   → "medium"
        - temporal_anomaly + any other signal   → "high"
        - temporal_anomaly + phishing_exposure  → "high" (social-engineering + timing)
        - temporal_anomaly + gps_mismatch       → "high" (impossible presence)

    Returns:
        JSON array string.  Each element contains:
        - "transaction_id" (str): UUID of the flagged transaction.
        - "reason" (str): describes which criterion triggered, e.g.
          "Transaction at unusual hour (03:00)"  or
          "Rapid transaction: 48s after previous by same sender"
        A transaction may appear twice if both criteria apply — pattern_agent will
        deduplicate by merging signals.
        Returns "[]" if no temporal anomalies found.

    Example output:
        [
          {
            "transaction_id": "40ee0d5f-53d3-493b-a888-ac59f77321f9",
            "reason": "Transaction at unusual hour (03:00)"
          },
          {
            "transaction_id": "7bcd2301-...",
            "reason": "Rapid transaction: 48s after previous by same sender"
          }
        ]

    Legitimate transactions NOT flagged:
        - Scheduled direct debits and standing orders (transaction_type == "direct debit")
          that happen to run at night are common for utility payments — check description
          before escalating these; decision_agent handles final filtering.

    When to call:
        Always — night-window is a medium-confidence standalone signal and a high-confidence
        amplifier.  Its output should be merged with all other detector results before
        pattern_agent assigns final confidence levels.
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

    Social engineering is the Mirror Hacker's primary account-takeover vector: victims
    receive convincing SMS or email messages impersonating MirrorPay security, urging
    them to "verify" credentials or "confirm" a payment.  After the victim complies,
    the hacker initiates fraudulent transfers from the now-compromised account.
    This detector flags every transaction by a sender who received such communications,
    regardless of whether the victim explicitly clicked — exposure alone is a risk signal.

    Algorithm:
        Reads phishing_in_comms and suspicious_domain_in_comms from enriched_transactions.csv.
        Flags rows where phishing_in_comms == True OR suspicious_domain_in_comms == True.
        Both columns are computed by run_feature_engineering by matching sender first/last
        name tokens (> 2 chars) against the full text of all SMS and email messages, then
        checking for keyword patterns and lookalike domain regexes.
        Fallback (no enriched CSV): scans sms.json and mails.json directly for phishing
        keywords, but returns raw message snippets rather than transaction IDs — pattern_agent
        cannot correlate these back to specific transactions without the enriched CSV.

    Key precomputed columns:
        phishing_in_comms           (bool) – True when sender's communications contain one or
                                             more of: "click here", "verify your account",
                                             "suspended", "urgent action", "confirm your",
                                             "password", "login immediately", "clicca qui",
                                             "verifica", "account blocked", "security alert",
                                             "verify now", "urgent".
        suspicious_domain_in_comms  (bool) – True when communications contain lookalike domains
                                             such as "paypa1.com", "mirr0r", "micros0ft",
                                             "amaz0n", "g00gle", "*-secure.*", "*-alert.*",
                                             or any domain with digit substitution.

    Signal produced: "phishing_exposure"

    Confidence rules (applied by pattern_agent / reflection_agent):
        - phishing_exposure alone → "medium"
        - phishing_exposure + urgency_signal → "high" (compound social engineering —
          forced upgrade by reflection_agent regardless of other signals)
        - phishing_exposure + new_recipient   → "high" (victim directed to a mule account)
        - phishing_exposure + any other signal → "high"

    Returns:
        JSON array string.  Each element contains:
        - "transaction_id" (str): UUID of a transaction by a phishing-exposed sender.
        - "reason" (str): which signals triggered, e.g.
          "Sender has phishing signals in communications: phishing_keywords, suspicious_domain"
        All transactions by an exposed sender are returned (not just one), because the
        hacker may execute multiple fraudulent transactions from the same compromised account.
        Returns "[]" if no sender has phishing content in their communications.

    Example output:
        [
          {
            "transaction_id": "40ee0d5f-53d3-493b-a888-ac59f77321f9",
            "reason": "Sender has phishing signals in communications: phishing_keywords"
          }
        ]

    When to call:
        Always — phishing_exposure is a medium-confidence standalone signal and a
        high-confidence amplifier with any corroborating anomaly.  Merge its output with
        detect_urgency_signals and detect_new_recipient_anomalies to build compound scores.
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

    First-time transfers to a new IBAN are a hallmark of account-takeover fraud: the
    attacker adds a mule account as a new payee and immediately initiates a large transfer.
    Victims are often unaware until they check their balance.  This detector applies a
    minimum amount threshold (> 1× monthly salary) to suppress false positives from small
    trial transfers or low-value first payments between friends.

    Algorithm:
        Reads is_new_recipient and amount_vs_salary_ratio from enriched_transactions.csv.
        Flags rows where is_new_recipient == True AND amount_vs_salary_ratio > 1.0.
        If amount_vs_salary_ratio is absent (salary unknown), flags all new-recipient rows.
        Falls back to returning an empty result if is_new_recipient column is missing.

    Key precomputed columns:
        is_new_recipient       (bool)  – True for the first-ever transaction from this sender
                                         to this recipient_iban / recipient_id, computed in
                                         strict chronological order.
        amount_vs_salary_ratio (float) – tx.amount / monthly_salary; used to apply the 1×
                                         minimum threshold.

    Signal produced: "new_recipient"

    Confidence rules (applied by pattern_agent / reflection_agent):
        - new_recipient alone, amount 1–2× salary   → "low"
        - new_recipient alone, amount 2–10× salary  → "medium"
        - new_recipient alone, amount > 10× salary  → "high"
        - new_recipient + phishing_exposure          → "high" (mule + social engineering)
        - new_recipient + iban_country_mismatch     → "high" (cross-border mule account)
        - new_recipient + temporal_anomaly          → "high"

    Returns:
        JSON array string.  Each element contains:
        - "transaction_id" (str): UUID of the flagged transfer.
        - "reason" (str): recipient identifier, amount, and salary ratio, e.g.
          "First-ever transfer to new recipient FR67H10...; amount 3200.00 (2.3× salary)"
        Returns "[]" if no qualifying new-recipient transfers exist.

    Example output:
        [
          {
            "transaction_id": "2de71a00-...",
            "reason": "First-ever transfer to new recipient FR67H1015963463086100916903; amount 3200.00 (2.3× salary)"
          }
        ]

    Legitimate transactions NOT flagged (due to the 1× salary threshold):
        - Small first-time transfers to friends or merchants (< 1× monthly salary)
        - The amount threshold does not apply when salary is unknown — all new recipients
          are reported in that case; decision_agent applies final legitimacy filtering.

    When to call:
        Always — pair this with detect_phishing_victims and detect_iban_country_anomalies
        for high-confidence compound account-takeover signals.
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


# ── Detector 9 ────────────────────────────────────────────────────────────────

def detect_impossible_travel() -> str:
    """Detect transactions from senders whose GPS history implies travel speed > 1 500 km/h.

    In Reply Mirror (2087), every citizen's biotag continuously emits GPS pings.  If two
    consecutive pings for the same biotag imply a speed greater than 1 500 km/h (faster
    than any commercial aircraft, and approaching Mach 1.5), the physical biotag cannot
    have moved that quickly — the device was either cloned or its GPS data was spoofed.
    This is one of the Mirror Hacker's most sophisticated tactics: creating a duplicate
    biotag fingerprint so the victim's location history shows plausible movement while a
    cloned device initiates fraudulent transactions elsewhere.

    Algorithm:
        Reads has_impossible_travel from enriched_transactions.csv.
        Flags all transactions by senders whose biotag history contains at least one
        ping-to-ping transition with speed > 1 500 km/h within a 2-hour window.
        The 2-hour cap avoids false positives from GPS data gaps spanning days.
        Falls back to returning an empty result if has_impossible_travel column is missing.

    Key precomputed column: has_impossible_travel
        Computed by run_feature_engineering: True for a sender if ANY consecutive GPS ping
        pair (sorted by timestamp) yields:
            haversine_distance_km / elapsed_hours > 1 500
        and the elapsed time is between 0 and 2 hours.
        All transactions by that sender are flagged — not just the one nearest the anomaly.

    Signal produced: "impossible_travel"

    Confidence rules (applied by pattern_agent / reflection_agent):
        - impossible_travel ALONE → always "high" (definitive biotag-clone evidence)
          This is a special override: reflection_agent forces "high" regardless of other
          signals or the initial confidence assigned by pattern_agent.
        - Combined with any other signal → "high" (no upgrade needed; already forced)

    Returns:
        JSON array string.  Each element contains:
        - "transaction_id" (str): UUID of a transaction by an impossible-travel sender.
        - "reason" (str): fixed description of the detected anomaly:
          "Sender GPS history contains impossible travel (speed > 1 500 km/h) — likely biotag clone or device takeover"
        Returns "[]" if no sender's GPS history contains impossible-speed transitions.

    Example output:
        [
          {
            "transaction_id": "9ff0a200-...",
            "reason": "Sender GPS history contains impossible travel (speed > 1 500 km/h) — likely biotag clone or device takeover"
          }
        ]

    When to call:
        Always — impossible_travel is the only signal that guarantees "high" confidence
        on its own.  Call it alongside all other detectors; the merge step in pattern_agent
        will combine it with any other signals found for the same transaction.
    """
    df = _load_enriched()
    suspicious = []

    if "has_impossible_travel" not in df.columns:
        return json.dumps([], indent=2)

    flagged = df[df["has_impossible_travel"] == True]
    for _, row in flagged.iterrows():
        suspicious.append({
            "transaction_id": row["transaction_id"],
            "reason": (
                "Sender GPS history contains impossible travel (speed > 1 500 km/h) — "
                "likely biotag clone or device takeover"
            ),
        })

    return json.dumps(suspicious, indent=2)


# ── Detector 10 ───────────────────────────────────────────────────────────────

def detect_urgency_signals() -> str:
    """Detect transactions from senders whose communications show social-engineering manipulation.

    A refinement of phishing detection, this detector specifically targets urgency-and-payment-
    link manipulation: the Mirror Hacker sends messages designed to create panic ("your account
    will be suspended in 24 hours") and then provides a payment link or portal to collect
    credentials or authorize fraudulent transfers.  While phishing_exposure captures broad
    credential-phishing patterns, urgency_signal focuses on the payment-action phase of an
    attack — when a victim is being coerced into completing a financial transaction.

    Algorithm:
        Reads urgency_keywords_count and payment_link_in_comms from enriched_transactions.csv.
        Flags rows where urgency_keywords_count ≥ 2 OR payment_link_in_comms == True.
        The threshold of ≥ 2 urgency keywords avoids flagging every message that includes
        the word "urgent" once; a pattern of multiple urgency markers indicates a crafted
        social-engineering message rather than a routine notification.
        Falls back to returning an empty result if both columns are missing.

    Key precomputed columns:
        urgency_keywords_count  (int)  – Count of urgency keywords found in sender's comms:
                                         "urgent", "immediate", "act now", "suspended",
                                         "verify now", "account blocked", "security alert",
                                         "within 24 hours", "asap".
        payment_link_in_comms   (bool) – True when sender's comms contain URL patterns
                                         matching fake payment portals: "https://…pay…",
                                         path "/pay", "click … pay", "pay … link".
                                         NOTE: bare "invoice", "payment due", "fattura"
                                         are excluded — they match legitimate billing emails.

    Signal produced: "urgency_signal"

    Confidence rules (applied by pattern_agent / reflection_agent):
        - urgency_signal alone → "medium"
        - urgency_signal + phishing_exposure → "high" (forced upgrade — compound social
          engineering special override in both pattern_agent and reflection_agent)
        - urgency_signal + new_recipient     → "high"
        - urgency_signal + any other signal  → "high"

    Returns:
        JSON array string.  Each element contains:
        - "transaction_id" (str): UUID of a transaction by a manipulation-targeted sender.
        - "reason" (str): which signals triggered, e.g.
          "Social engineering signals in sender comms: urgency_keywords (3), payment_link"
        All transactions by an affected sender are returned.
        Returns "[]" if no sender meets the urgency or payment-link thresholds.

    Example output:
        [
          {
            "transaction_id": "4a92ab00-...",
            "reason": "Social engineering signals in sender comms: urgency_keywords (3)"
          },
          {
            "transaction_id": "5b13cd11-...",
            "reason": "Social engineering signals in sender comms: urgency_keywords (2), payment_link"
          }
        ]

    When to call:
        Always — pair with detect_phishing_victims output: the combination of
        phishing_exposure + urgency_signal is a forced "high" confidence upgrade
        regardless of how many signals exist.  Merge results before pattern_agent
        assigns confidence levels.
    """
    df = _load_enriched()
    suspicious = []

    if "urgency_keywords_count" not in df.columns and "payment_link_in_comms" not in df.columns:
        return json.dumps([], indent=2)

    mask = pd.Series(False, index=df.index)
    if "urgency_keywords_count" in df.columns:
        mask = mask | (df["urgency_keywords_count"] >= 2)
    if "payment_link_in_comms" in df.columns:
        mask = mask | (df["payment_link_in_comms"] == True)

    for _, row in df[mask].iterrows():
        parts = []
        urgency_count = int(row.get("urgency_keywords_count", 0) or 0)
        if urgency_count >= 2:
            parts.append(f"urgency_keywords ({urgency_count})")
        if row.get("payment_link_in_comms", False):
            parts.append("payment_link")
        suspicious.append({
            "transaction_id": row["transaction_id"],
            "reason": f"Social engineering signals in sender comms: {', '.join(parts)}",
        })

    return json.dumps(suspicious, indent=2)


# ── Detector 7 (NEW) ──────────────────────────────────────────────────────────

def detect_iban_country_anomalies() -> str:
    """Detect transactions where the sender and recipient IBAN countries differ.

    Cross-border IBAN transfers are legitimate in everyday banking (e.g. sending rent
    to a foreign landlord), but they become highly significant when combined with other
    fraud signals.  The Mirror Hacker typically routes stolen funds to mule accounts in
    different countries to complicate recovery — a new foreign recipient added during a
    phishing attack is a strong indicator.  This detector surfaces the raw mismatch so
    pattern_agent can use it as a corroborating signal rather than a standalone flag.

    Algorithm:
        Reads the precomputed iban_country_mismatch column from enriched_transactions.csv.
        Flags all rows where iban_country_mismatch == True.
        No fallback logic — returns an empty result if the column is absent.

    Key precomputed columns:
        sender_iban_country    (str) – First 2 letters of sender_iban (ISO country code).
        recipient_iban_country (str) – First 2 letters of recipient_iban (ISO country code).
        iban_country_mismatch  (bool) – True when both country codes exist and differ.
        Both are None if the IBAN value is missing or malformed.

    Signal produced: "iban_country_mismatch"

    Confidence rules (applied by pattern_agent / reflection_agent):
        - iban_country_mismatch ALONE → "low" (very common in legitimate cross-border banking)
        - iban_country_mismatch + new_recipient           → "high" (foreign mule account)
        - iban_country_mismatch + gps_mismatch            → "high"
        - iban_country_mismatch + temporal_anomaly        → "medium"
        - iban_country_mismatch + 2 or more other signals → "high"

    Returns:
        JSON array string.  Each element contains:
        - "transaction_id" (str): UUID of the flagged transaction.
        - "reason" (str): sender and recipient IBAN country codes, e.g.
          "IBAN country mismatch: sender country IT → recipient country US"
        Returns "[]" if all transactions share the same sender/recipient IBAN country.

    Example output:
        [
          {
            "transaction_id": "ghi12345-...",
            "reason": "IBAN country mismatch: sender country IT → recipient country US"
          }
        ]

    Legitimate transactions commonly flagged but correctly filtered:
        - International salary payments, foreign subscriptions, cross-border rent
          (these are flagged here but dropped by decision_agent when legitimacy signals
          such as "Salary payment" or "Rent payment" descriptions are present)

    When to call:
        Always — use as a corroborating signal in the merge step.  Alone it produces only
        "low" confidence; combine with detect_new_recipient_anomalies or
        detect_phishing_victims to escalate to "high".
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
    """Detect senders who made 2 or more transactions within a 60-minute rolling window.

    Automated fraud scripts and account-compromise tools often exhaust an account by
    executing many transactions in rapid succession before the bank's fraud engine triggers
    a block.  This detector counts how many prior transactions the same sender made in the
    60 minutes before each transaction — a count of ≥ 2 means the current transaction is
    at least the third in an hour, which is unusual for a typical consumer account.

    Algorithm:
        Reads the precomputed velocity_burst_count column from enriched_transactions.csv.
        Flags all rows where velocity_burst_count ≥ 2.
        velocity_burst_count for a given transaction T is the count of all other
        transactions by the same sender with timestamp in (T − 60 min, T) — exclusive.
        Falls back to returning an empty result if the column is absent.

    Key precomputed column: velocity_burst_count
        (int) – Number of preceding transactions by the same sender in the 60-minute window
        immediately before this transaction.  A count of 0 means no prior transactions in
        the window; a count of ≥ 2 is the flag threshold.

    Signal produced: "velocity_burst"

    Confidence rules (applied by pattern_agent / reflection_agent):
        - velocity_burst ALONE        → "low" (high-frequency but may be legitimate)
        - velocity_burst + phishing_exposure or gps_mismatch → "high"
        - velocity_burst + temporal_anomaly               → "medium"
        - velocity_burst + 2 or more other signals        → "high"

    Returns:
        JSON array string.  Each element contains:
        - "transaction_id" (str): UUID of the transaction that is part of a burst.
        - "reason" (str): burst count description, e.g.
          "Velocity burst: 3 prior transactions by same sender in the preceding 60 minutes"
        All transactions in a burst window are returned individually (e.g. if a sender
        makes 4 transactions in an hour, all 4 may appear with counts 0, 1, 2, 3 —
        only those with count ≥ 2 are included).
        Returns "[]" if no sender has ≥ 2 preceding transactions in any 60-minute window.

    Example output:
        [
          {
            "transaction_id": "8de90b22-...",
            "reason": "Velocity burst: 2 prior transactions by same sender in the preceding 60 minutes"
          }
        ]

    Legitimate transactions NOT flagged:
        - Salary runs from EMP senders (multiple transfers in a short time are expected
          for employer payroll systems; decision_agent filters these on the EMP prefix)
        - The first two transactions in a burst window (count 0 and 1) are not returned

    When to call:
        Always — velocity_burst is weak alone but provides meaningful corroboration.
        Check detect_temporal_anomalies results for overlap with rapid-fire detection
        (< 5 min between consecutive transactions), which is a complementary signal.
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
