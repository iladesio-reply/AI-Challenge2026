import json
import math
import pandas as pd
from pathlib import Path
from the_eye import config


def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)
    ) * math.sin(dlng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def detect_location_anomalies() -> str:
    """Detect in-person payment fraud by comparing transaction location against the sender's real-time GPS biotag.

    For each in-person payment, finds the closest GPS biotag ping (by timestamp) for the sender
    and computes the haversine distance between that ping and the user's registered residence.
    Flags transactions where the distance exceeds 100km — physically implying the card was
    used in a location the owner could not have reached.

    Returns:
        JSON array of suspicious transactions. Each element contains:
        - "transaction_id": UUID of the flagged transaction
        - "reason": human-readable explanation with distance in km

        Example:
        [
          {
            "transaction_id": "43be5588-2cfb-47c1-a8aa-aeb8d2f38aff",
            "reason": "GPS mismatch: 3806km between transaction and user location"
          }
        ]
        Returns an empty array [] if no anomalies are found.

    When to call: always call this as the first detector. It produces the strongest
    fraud signal — a GPS mismatch >500km is near-definitive evidence of card theft or cloning.
    """
    base = Path(config.DATASET_PATH)
    df = pd.read_csv(base / "transactions.csv")

    with open(base / "users.json") as f:
        users = json.load(f)
    with open(base / "locations.json") as f:
        locations = json.load(f)

    user_by_iban = {u.get("iban", ""): u for u in users}
    loc_by_biotag: dict[str, list] = {}
    for loc in locations:
        loc_by_biotag.setdefault(loc.get("biotag", ""), []).append(loc)

    suspicious = []
    in_person = df[df["transaction_type"] == "in-person payment"]

    for _, row in in_person.iterrows():
        sender = row.get("sender_id", "")
        sender_iban = row.get("sender_iban", "")
        user = user_by_iban.get(sender_iban) or user_by_iban.get(sender)
        if not user:
            continue
        user_locs = loc_by_biotag.get(sender, [])
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
            dist = _haversine(float(res["lat"]), float(res["lng"]), float(closest["lat"]), float(closest["lng"]))
            if dist > 100:
                suspicious.append({
                    "transaction_id": row["transaction_id"],
                    "reason": f"GPS mismatch: {dist:.0f}km between transaction and user location",
                })

    return json.dumps(suspicious, indent=2)


def detect_withdrawal_anomalies() -> str:
    """Detect cash withdrawal fraud by comparing the withdrawal city against the user's home city.

    For each withdrawal transaction, extracts the city from the location field and checks whether
    it matches the user's registered residence city. A cash withdrawal in a foreign city indicates
    the card or credentials were used without the owner's physical presence.

    This detector covers all 42 withdrawal transactions in the dataset — a major source of fraud
    signals that is entirely separate from the GPS biotag check.

    Returns:
        JSON array of suspicious transactions. Each element contains:
        - "transaction_id": UUID of the flagged transaction
        - "reason": explanation with home city and withdrawal city

        Example:
        [
          {
            "transaction_id": "81c13ad6-e1b7-4aa7-b962-ec816d984e79",
            "reason": "Withdrawal in Chicago while user residence is Rennes"
          }
        ]
        Returns an empty array [] if all withdrawals match the user's home city.

    When to call: always call this. Withdrawal mismatches are high-confidence signals —
    no legitimate reason exists for a user to withdraw cash in a city thousands of km from home
    without a corresponding GPS ping near that location.
    """
    base = Path(config.DATASET_PATH)
    df = pd.read_csv(base / "transactions.csv")

    with open(base / "users.json") as f:
        users = json.load(f)

    user_by_iban = {u.get("iban", ""): u for u in users}
    user_by_id = {u.get("iban", "").replace("IBAN", ""): u for u in users}

    suspicious = []
    withdrawals = df[df["transaction_type"] == "withdrawal"]

    for _, row in withdrawals.iterrows():
        sender_iban = str(row.get("sender_iban", ""))
        sender_id = str(row.get("sender_id", ""))
        user = user_by_iban.get(sender_iban)

        if not user:
            for u in users:
                if sender_id in str(u):
                    user = u
                    break

        if not user:
            continue

        home_city = user.get("residence", {}).get("city", "")
        location = str(row.get("location", ""))

        if home_city and location and home_city.lower() not in location.lower():
            suspicious.append({
                "transaction_id": row["transaction_id"],
                "reason": f"Withdrawal in '{location.split(' - ')[0].strip()}' while user residence is '{home_city}'",
            })

    return json.dumps(suspicious, indent=2)


def detect_amount_anomalies() -> str:
    """Detect transactions with unusually large amounts relative to the sender's monthly salary.

    Computes the ratio of transaction amount to the sender's monthly salary (annual salary / 12).
    Flags transactions where amount > 2x monthly salary, which is atypical for legitimate
    personal spending and may indicate unauthorized fund transfers or account takeover.

    Note: only applies to senders whose IBAN appears in users.json. Employer transfers
    (sender_id starting with EMP) are not matched and are naturally excluded.

    Returns:
        JSON array of suspicious transactions. Each element contains:
        - "transaction_id": UUID of the flagged transaction
        - "reason": explanation with absolute amount, monthly salary, and multiplier

        Example:
        [
          {
            "transaction_id": "142ddf94-ca31-4985-a2bc-fab64bf31432",
            "reason": "Amount 8400.00 is 2.4x monthly salary (3500.00)"
          }
        ]
        Returns an empty array [] if no anomalies are found.

    When to call: always call this. Combine results with temporal and GPS signals —
    a high-amount transaction at night or from an unusual location is a strong compound signal.
    """
    base = Path(config.DATASET_PATH)
    df = pd.read_csv(base / "transactions.csv")

    with open(base / "users.json") as f:
        users = json.load(f)

    salary_map = {u.get("iban", ""): u.get("salary", 0) for u in users}

    suspicious = []
    for _, row in df.iterrows():
        sender_iban = row.get("sender_iban", "")
        annual_salary = salary_map.get(sender_iban, 0)
        if annual_salary and annual_salary > 0:
            monthly_salary = annual_salary / 12
            ratio = row["amount"] / monthly_salary
            if ratio > 2.0:
                suspicious.append({
                    "transaction_id": row["transaction_id"],
                    "reason": f"Amount {row['amount']:.2f} is {ratio:.1f}x monthly salary ({monthly_salary:.2f})",
                })

    return json.dumps(suspicious, indent=2)


def detect_temporal_anomalies() -> str:
    """Detect transactions that occur at statistically unusual times or in suspiciously rapid succession.

    Two sub-detectors:
    1. Night-window (00:00–06:00): legitimate users rarely transact in this extended window.
       Mirror Hackers are known to shift activity to late-night hours to avoid real-time monitoring.
    2. Rapid-fire (<5 minutes): two transactions by the same sender within 5 minutes suggest
       automated fraud scripts or simultaneous device compromise.

    Returns:
        JSON array of suspicious transactions. Each element contains:
        - "transaction_id": UUID of the flagged transaction
        - "reason": explanation (hour for night-window; elapsed seconds for rapid-fire)

        Example:
        [
          {
            "transaction_id": "40ee0d5f-53d3-493b-a888-ac59f77321f9",
            "reason": "Transaction at unusual hour (03:00)"
          },
          {
            "transaction_id": "0c84db52-b7ff-4873-a6f2-dc3fe7e0bf3f",
            "reason": "Rapid transaction: 42s after previous by same sender"
          }
        ]
        Returns an empty array [] if no anomalies are found.

    When to call: always call this. Night-window alone is medium confidence; combined with
    a phishing signal, GPS mismatch, or withdrawal anomaly it becomes high confidence.
    """
    base = Path(config.DATASET_PATH)
    df = pd.read_csv(base / "transactions.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    suspicious = []

    for _, row in df[df["hour"].between(0, 6)].iterrows():
        suspicious.append({
            "transaction_id": row["transaction_id"],
            "reason": f"Transaction at unusual hour ({row['hour']:02d}:00)",
        })

    df_sorted = df.sort_values(["sender_id", "timestamp"])
    df_sorted["prev_time"] = df_sorted.groupby("sender_id")["timestamp"].shift(1)
    df_sorted["seconds_since_last"] = (
        df_sorted["timestamp"] - df_sorted["prev_time"]
    ).dt.total_seconds()

    for _, row in df_sorted[df_sorted["seconds_since_last"] < 300].iterrows():
        suspicious.append({
            "transaction_id": row["transaction_id"],
            "reason": f"Rapid transaction: {row['seconds_since_last']:.0f}s after previous by same sender",
        })

    return json.dumps(suspicious, indent=2)


def detect_phishing_victims() -> str:
    """Scan SMS and email communications for phishing content that may have compromised user credentials.

    Searches all SMS threads (sms.json) and email messages (mails.json) for known phishing
    keywords and social engineering patterns. A user who received a phishing message is at elevated
    risk of account takeover — their subsequent transactions should be treated with higher suspicion,
    especially if combined with temporal or amount anomalies.

    Keywords scanned: "click here", "verify your account", "suspended", "urgent action",
    "confirm your", "password", "login immediately", "clicca qui", "verifica",
    "account blocked", "security alert".

    Note: this tool returns communication snippets, not transaction IDs. Use the user identifiers
    in the snippets to cross-reference with transactions from the same sender.

    Returns:
        JSON array of suspicious communications (up to 30). Each element contains:
        - "keywords": list of matched phishing keywords found in the message
        - "snippet": first 300 characters of the suspicious message

        Example:
        [
          {
            "keywords": ["verify your account", "password"],
            "snippet": "From: security@mirrorpay-alert.com\\nSubject: Urgent: verify your account..."
          }
        ]
        Returns an empty array [] if no phishing content is detected.

    When to call: always call this. Cross-reference user names/IDs found in snippets
    against the senders of flagged transactions to identify account-takeover patterns.
    """
    base = Path(config.DATASET_PATH)

    with open(base / "sms.json") as f:
        sms_list = json.load(f)
    with open(base / "mails.json") as f:
        mail_list = json.load(f)

    phishing_keywords = [
        "click here", "verify your account", "suspended", "urgent action",
        "confirm your", "password", "login immediately", "clicca qui", "verifica",
        "account blocked", "security alert",
    ]

    suspicious = []
    for item in sms_list + mail_list:
        text = str(item).lower()
        matched = [kw for kw in phishing_keywords if kw in text]
        if matched:
            suspicious.append({"keywords": matched, "snippet": str(item)[:300]})

    return json.dumps(suspicious[:30], indent=2)
