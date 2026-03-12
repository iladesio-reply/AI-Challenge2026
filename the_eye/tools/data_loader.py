import json
import pandas as pd
from pathlib import Path
from the_eye import config


def get_transaction_summary() -> str:
    """Return a statistical overview of the current level's transaction dataset.

    Reads transactions.csv and computes aggregate statistics useful for understanding
    the dataset before running fraud detectors: volume, date range, type distribution,
    payment method breakdown, and amount statistics.

    Returns:
        Plain-text summary string. Example:
        Transaction dataset summary:
        - Total transactions: 524
        - Date range: 2087-01-01T09:19:18 to 2087-12-28T14:33:00
        - Transaction types: {'transfer': 384, 'in-person payment': 43, ...}
        - Payment methods: {'debit card': 85, 'mobile phone': 12, ...}
        - Amount stats: min=5.20, max=18729.76, mean=1203.44
        - Unique senders: 47
        - Unique recipients: 312

    When to call: call this first to understand the dataset size and composition
    before delegating to pattern_agent.
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

    Reads transactions.csv and returns the first `limit` rows as JSON.
    Fields per record: transaction_id, sender_id, recipient_id, transaction_type,
    amount, location, payment_method, sender_iban, recipient_iban, balance_after,
    description, timestamp.

    Args:
        limit: Maximum number of records to return. Defaults to 50.
                Increase for broader sampling; keep low to stay within context limits.

    Returns:
        JSON array string of transaction records.

    When to call: use for spot-checking specific transaction patterns or when you need
    raw data to manually identify fraud signals not covered by the automated detectors.
    """
    base = Path(config.DATASET_PATH)
    df = pd.read_csv(base / "transactions.csv")
    return df.head(limit).to_json(orient="records", indent=2)


def get_user_profile(user_id: str) -> str:
    """Return the full profile of a MirrorPay citizen by their sender/recipient ID or IBAN.

    Searches users.json for a user whose data contains the given identifier (IBAN,
    sender_id, or name fragment). Returns the complete profile including name, birth year,
    salary, job, IBAN, residence coordinates, and a natural-language description that
    includes information about travel habits and phishing susceptibility.

    Args:
        user_id: Any identifier to search for — IBAN (e.g. "FR67H1015963463086100916903"),
                 sender_id (e.g. "CLLT-ZCHR-7FA-RUE-0"), or partial name.

    Returns:
        JSON object string with the user's full profile, or a "not found" message.

        Example field of interest in the description:
        "...parfois imprudent en ligne, avec près d'une chance sur deux de cliquer sur
         un lien de phishing..." → high phishing susceptibility.

    When to call: use to verify a flagged sender's salary (for amount anomaly context),
    residence location (for GPS comparison), or phishing susceptibility (for account
    takeover risk assessment).
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

    In Reply Mirror (2087), every citizen carries a biotag that continuously pings
    their GPS position. This history is used to verify whether a user was physically
    present at the location of an in-person payment.

    Args:
        biotag: The biotag identifier, which matches the sender_id field in transactions.csv
                (e.g. "CLLT-ZCHR-7FA-RUE-0").

    Returns:
        JSON array of up to 50 location pings, each containing:
        - "biotag": user identifier
        - "timestamp": ISO datetime of the ping
        - "lat", "lng": GPS coordinates (as strings)
        - "city": city name at that location

        Returns a "not found" message if no pings exist for the given biotag.

    When to call: use when investigating an in-person payment flagged by detect_location_anomalies,
    or to manually verify a user's whereabouts during a suspicious transaction window.
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

    Searches sms.json and mails.json for any message containing the given user_id string.
    Useful for identifying phishing attempts targeting a specific user, verifying whether
    a user interacted with a suspicious link, or finding social engineering context.

    Args:
        user_id: Any string to search for within communications — sender_id, name,
                 email address, or IBAN fragment (e.g. "Zacharie", "CLLT-ZCHR-7FA-RUE-0").

    Returns:
        JSON object with two fields:
        - "sms": list of up to 10 matching SMS records (each has a "sms" text field)
        - "mails": list of up to 10 matching email records (each has a "mail" HTML/text field)

        Example: {"sms": [...], "mails": [...]}

    When to call: use to enrich fraud analysis for a specific flagged sender — look for
    phishing emails, fake security alerts, or social engineering attempts that preceded
    the suspicious transaction.
    """
    base = Path(config.DATASET_PATH)
    with open(base / "sms.json") as f:
        sms_list = json.load(f)
    with open(base / "mails.json") as f:
        mail_list = json.load(f)
    relevant_sms = [s for s in sms_list if user_id in str(s)]
    relevant_mails = [m for m in mail_list if user_id in str(m)]
    return json.dumps({"sms": relevant_sms[:10], "mails": relevant_mails[:10]}, indent=2)
