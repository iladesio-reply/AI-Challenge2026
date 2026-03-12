import json
import pandas as pd
from pathlib import Path
from the_eye import config


def get_transaction_summary() -> str:
    """Return a statistical overview of the current level's transaction dataset.

    Reads transactions.csv from config.DATASET_PATH and computes aggregate statistics
    that are useful for understanding the scale and composition of the dataset before
    invoking fraud detectors.  This provides a fast sanity-check view: how many
    transactions exist, what types are present, what the amount distribution looks like,
    and how many unique senders and recipients are involved.

    No parameters required.

    Returns:
        Plain-text summary string with the following fields:
        - "Total transactions"  – row count of transactions.csv
        - "Date range"          – earliest and latest timestamp values
        - "Transaction types"   – value counts dict, e.g. {'transfer': 384,
                                  'in-person payment': 43, 'withdrawal': 12, ...}
        - "Payment methods"     – value counts dict for non-null payment_method values,
                                  e.g. {'debit card': 85, 'mobile phone': 12, ...}
        - "Amount stats"        – min, max, and mean of the amount column (EUR)
        - "Unique senders"      – cardinality of sender_id column
        - "Unique recipients"   – cardinality of recipient_id column

    Example return value:
        Transaction dataset summary:
        - Total transactions: 524
        - Date range: 2087-01-01T09:19:18 to 2087-12-28T14:33:00
        - Transaction types: {'transfer': 384, 'in-person payment': 43, 'withdrawal': 12}
        - Payment methods: {'debit card': 85, 'mobile phone': 12}
        - Amount stats: min=5.20, max=18729.76, mean=1203.44
        - Unique senders: 47
        - Unique recipients: 312

    When to call:
        Call once at the start of an investigation session to understand the scope of the
        dataset.  Do NOT call repeatedly in a loop — the summary is static for a given
        pipeline run.  Do NOT use this to retrieve specific transaction records; use
        get_transactions_json() for that.
    """
    base = Path(config.DATASET_PATH)
    df = pd.read_csv(base / "transactions.csv")
    return (
        f"Transaction dataset summary:\n"
        f"- Total transactions: {len(df)}\n"
        f"- Date range: {df['timestamp'].min()} to {df['timestamp'].max()}\n"
        f"- Transaction types: {df['transaction_type'].value_counts().to_dict()}\n"
        f"- Payment methods: {df['payment_method'].dropna().value_counts().to_dict()}\n"
        f"- Amount stats: min={df['amount'].min():.2f}, max={df['amount'].max():.2f}, mean={df['amount'].mean():.2f}\n"
        f"- Unique senders: {df['sender_id'].nunique()}\n"
        f"- Unique recipients: {df['recipient_id'].nunique()}\n"
    )


def get_transactions_json(limit: int = 50) -> str:
    """Return raw transaction records as a JSON array for manual inspection.

    Reads transactions.csv from config.DATASET_PATH and returns up to `limit` rows
    as a JSON array.  Useful for spot-checking specific transaction fields or manually
    investigating patterns that the automated detectors may not cover.  Records are
    returned in file order (not sorted by time or sender).

    Args:
        limit (int): Maximum number of records to return.  Default: 50.
                     Increase to 200–500 for broader sampling across the dataset.
                     Keep small (10–20) when you only need a few examples to inspect.
                     Very large values (> 1000) may hit context-window limits.

    Returns:
        JSON array string.  Each element is a transaction object with these fields:
        - "transaction_id"   (str)          – UUID of the transaction.
        - "sender_id"        (str)          – Sender's biotag / citizen ID (= GPS biotag key).
        - "recipient_id"     (str | null)   – Recipient's citizen ID (null for external IBANs).
        - "transaction_type" (str)          – One of: "transfer", "in-person payment",
                                              "withdrawal", "direct debit", "e-commerce".
        - "amount"           (float)        – Transaction amount in EUR.
        - "location"         (str | null)   – "City - Venue" for physical transactions; null
                                              for transfers and e-commerce.
        - "payment_method"   (str | null)   – e.g. "debit card", "mobile phone", "bank transfer".
        - "sender_iban"      (str)          – Sender's IBAN (first 2 chars = country code).
        - "recipient_iban"   (str | null)   – Recipient's IBAN; null if recipient is internal.
        - "balance_after"    (float)        – Sender's account balance after this transaction.
        - "description"      (str | null)   – Free-text transaction description; check for
                                              "Salary payment" (EMP sender) or "Rent payment"
                                              to identify legitimate transactions.
        - "timestamp"        (str)          – ISO 8601 datetime string.

    Example return value (one record):
        [
          {
            "transaction_id": "43be5588-2cfb-47c1-a8aa-aeb8d2f38aff",
            "sender_id": "CLLT-ZCHR-7FA-RUE-0",
            "recipient_id": null,
            "transaction_type": "in-person payment",
            "amount": 4500.00,
            "location": "Turin - Piazza Castello",
            "payment_method": "debit card",
            "sender_iban": "IT60L0100803268000000246810",
            "recipient_iban": null,
            "balance_after": 1200.00,
            "description": null,
            "timestamp": "2087-03-14T03:22:00"
          }
        ]

    When to call:
        Use for manual spot-checking when you need to read raw field values for specific
        transactions (e.g. to verify a description, check a balance, or inspect a location
        string).  Do NOT use this for bulk fraud detection — the automated detectors in
        fraud_signals.py work directly on the enriched CSV and are far more efficient.
        Prefer get_user_profile() when the goal is to look up a sender's salary or IBAN.
    """
    base = Path(config.DATASET_PATH)
    df = pd.read_csv(base / "transactions.csv")
    return df.head(limit).to_json(orient="records", indent=2)


def get_user_profile(user_id: str) -> str:
    """Return the full profile of a MirrorPay citizen by their sender/recipient ID or IBAN.

    Searches users.json for a user whose serialized record contains the given identifier
    string.  Returns the complete citizen profile, which includes salary (critical for
    contextualizing amount anomalies), residence coordinates (for manual GPS comparison),
    and a natural-language description that encodes the citizen's travel habits and
    phishing susceptibility as prose (in French, the language of the dataset).

    Args:
        user_id (str): Any substring to match against the user record.  Common values:
                       - IBAN string (e.g. "IT60L0100803268000000246810")
                       - sender_id / biotag (e.g. "CLLT-ZCHR-7FA-RUE-0")
                       - First or last name fragment (e.g. "Zacharie", "Collet")
                       The match is a substring search on the full JSON-serialized record,
                       so any unique fragment will work.

    Returns:
        JSON object string with the user's complete profile.  Key fields:
        - "first_name", "last_name"    (str) – Citizen's name (used for comms matching).
        - "iban"                       (str) – Citizen's IBAN (primary key in users.json).
        - "salary"                     (int) – Annual salary in EUR.  Monthly = salary / 12.
                                               Use this to contextualise amount anomalies:
                                               is the flagged amount really unusual for this
                                               specific person?
        - "job"                        (str) – Occupation (context for salary expectations).
        - "residence"  {lat, lng, city} (obj)– Home coordinates.  Not used for fraud detection
                                               (GPS ping vs transaction city is preferred),
                                               but useful for manual verification.
        - "description"                (str) – Narrative text (in French) describing personality,
                                               travel habits, and online behaviour.
                                               Look for phishing-susceptibility clues such as:
                                               "imprudent en ligne", "chance sur deux de cliquer
                                               sur un lien de phishing" (high risk), or
                                               "très prudent" (low risk).
        Returns a "User <id> not found." string if no record matches.

    Example return value (excerpt):
        {
          "first_name": "Zacharie",
          "last_name": "Collet",
          "iban": "IT60L0100803268000000246810",
          "salary": 42000,
          "job": "Ingénieur logiciel",
          "residence": {"lat": 45.0703, "lng": 7.6869, "city": "Turin"},
          "description": "...parfois imprudent en ligne, avec près d'une chance sur deux de cliquer sur un lien de phishing..."
        }

    When to call:
        - When you need the exact annual salary of a specific sender to calculate whether
          a flagged transaction's amount_vs_salary_ratio is accurate or plausible.
        - When you need the residence city for manual GPS sanity-check.
        - When the pattern_agent or decision_agent needs to confirm phishing susceptibility
          beyond the boolean phishing_in_comms feature.
        Do NOT call this in a loop over all senders — use the enriched CSV features instead.
    """
    base = Path(config.DATASET_PATH)
    with open(base / "users.json") as f:
        users = json.load(f)
    for user in users:
        if user_id in str(user):
            return json.dumps(user, indent=2)
    return f"User {user_id} not found."


def get_user_location_history(biotag: str) -> str:
    """Return the GPS location history for a user identified by their biotag.

    In Reply Mirror (2087), every citizen carries a biotag — a biometric device that
    continuously emits GPS pings to the MirrorPay network.  The history of these pings
    is used by fraud detectors to verify physical presence: if the user's biotag was
    recorded in Rome while an in-person payment was made in Tokyo, the transaction is
    almost certainly fraudulent.  This function provides the raw ping data for manual
    verification or deeper investigation beyond the precomputed features.

    Args:
        biotag (str): The biotag identifier.  This is the same value as the sender_id
                      field in transactions.csv (e.g. "CLLT-ZCHR-7FA-RUE-0").
                      Use the sender_id of the transaction you are investigating.

    Returns:
        JSON array string of up to 50 GPS pings, sorted in file order (not guaranteed
        chronological — sort by timestamp if ordering matters).
        Each ping object contains:
        - "biotag"     (str)   – The sender_id / biotag identifier.
        - "timestamp"  (str)   – ISO 8601 datetime of the GPS ping.
        - "lat"        (str)   – Latitude as a string (convert to float for calculations).
        - "lng"        (str)   – Longitude as a string (convert to float for calculations).
        - "city"       (str)   – City name at the recorded coordinates.
        Returns "No location history for biotag <biotag>." if no records exist.

    Example return value (two pings):
        [
          {
            "biotag": "CLLT-ZCHR-7FA-RUE-0",
            "timestamp": "2087-03-13T22:14:00",
            "lat": "45.0703",
            "lng": "7.6869",
            "city": "Turin"
          },
          {
            "biotag": "CLLT-ZCHR-7FA-RUE-0",
            "timestamp": "2087-03-14T03:05:00",
            "lat": "48.8566",
            "lng": "2.3522",
            "city": "Paris"
          }
        ]

    When to call:
        - When you need to manually verify a specific in-person payment flagged by
          detect_location_anomalies — look for pings near the transaction timestamp
          to confirm or refute the GPS mismatch.
        - When you need to check whether a specific city appears in a sender's history
          to corroborate or challenge a detect_withdrawal_anomalies result.
        - When investigating an impossible_travel flag — retrieve the pings to manually
          compute the speed between consecutive city transitions.
        Do NOT call this for every sender in the dataset — the has_impossible_travel
        and gps_distance_to_tx_km columns in the enriched CSV already capture the
        relevant precomputed signals for all senders at once.
    """
    base = Path(config.DATASET_PATH)
    with open(base / "locations.json") as f:
        locations = json.load(f)
    history = [loc for loc in locations if loc.get("biotag") == biotag]
    if not history:
        return f"No location history for biotag {biotag}."
    return json.dumps(history[:50], indent=2)


def get_communications(user_id: str) -> str:
    """Return SMS and email communications that mention a specific user.

    Searches sms.json and mails.json for messages whose serialized content contains the
    given user_id string.  This is the primary tool for retrieving the raw text of
    phishing attempts, social engineering messages, and suspicious payment-link emails
    that may have preceded a fraudulent transaction.  Use it to read the actual message
    content when the boolean features (phishing_in_comms, urgency_keywords_count, etc.)
    flag a sender but you need to understand the specific attack narrative.

    Args:
        user_id (str): Any substring to match against the communication records.
                       Effective values include:
                       - sender_id / biotag (e.g. "CLLT-ZCHR-7FA-RUE-0")
                       - First or last name (e.g. "Zacharie", "Collet")
                       - Email address fragment, IBAN substring, or any unique identifier.
                       The match is a substring search on the full JSON-serialized record,
                       so name fragments work reliably if unique (> 2 characters).

    Returns:
        JSON object string with two keys:
        - "sms"   (list[obj]) – Up to 10 SMS records matching the query.
                                Each object has at minimum a "sms" key with the message text.
        - "mails" (list[obj]) – Up to 10 email records matching the query.
                                Each object has at minimum a "mail" key with the full HTML/plain
                                text content of the email.
        Both lists are empty ([]) if no matching records are found.
        Returns at most 10 records per channel — if a sender received more than 10 messages,
        only the first 10 are returned.

    Example return value:
        {
          "sms": [
            {"sms": "Zacharie, votre compte MirrorPay a été suspendu. Vérifiez ici: https://mirr0r-secure.net/verify"}
          ],
          "mails": [
            {"mail": "<html>...Urgent: cliquez ici pour confirmer votre identité...</html>"}
          ]
        }

    What to look for in the returned content:
        - Phishing indicators: "vérifiez", "suspendu", "urgent", "confirmer", "password",
          "click here", "verify your account" (in any language).
        - Suspicious domains: "mirr0r", "paypa1.com", "*-secure.*", "*-alert.*",
          digit-substituted brand names.
        - Payment links: "https://…pay…", URLs containing "pay" or "/pay" in the path.
        - Urgency escalation: multiple urgency keywords ("urgent", "immediate", "asap",
          "within 24 hours") in the same message.

    When to call:
        - When you need to manually inspect the raw phishing content targeting a flagged
          sender — for example, to write a precise "details" sentence for the output JSON.
        - When you need to verify whether a payment link URL is genuinely suspicious or
          a legitimate billing portal.
        - When the phishing_in_comms or urgency_keywords_count features flagged a sender
          and you need the actual message text to confirm the attack context.
        Do NOT call this for every flagged sender — read the precomputed enriched CSV
        features (phishing_in_comms, urgency_keywords_count, payment_link_in_comms)
        for bulk analysis; use this only for targeted per-sender deep dives.
    """
    base = Path(config.DATASET_PATH)
    with open(base / "sms.json") as f:
        sms_list = json.load(f)
    with open(base / "mails.json") as f:
        mail_list = json.load(f)
    relevant_sms = [s for s in sms_list if user_id in str(s)]
    relevant_mails = [m for m in mail_list if user_id in str(m)]
    return json.dumps({"sms": relevant_sms[:10], "mails": relevant_mails[:10]}, indent=2)
