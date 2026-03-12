import re
from pathlib import Path


def write_predictions(fraud_ids: list[str], output_path: str) -> str:
    """Write the final list of suspected fraudulent transaction IDs to the output file.

    Creates the output directory if needed, deduplicates IDs, and writes one UUID
    per line in ASCII encoding with a trailing newline — the format required by the
    Reply AI Agent Challenge 2026 submission spec.

    Args:
        fraud_ids: List of transaction UUID strings to flag as fraudulent.
                   Duplicates are removed; order is preserved.
        output_path: Destination file path (e.g. "output/predictions.txt").

    Returns:
        Confirmation string: "Written N fraud IDs to <path>"

    Raises:
        UnicodeEncodeError: if any ID contains non-ASCII characters (should not happen
                            with standard UUIDs).

    Submission validity constraints (enforced externally):
        - File must not be empty (at least 1 ID)
        - File must not contain ALL transaction IDs
        - Must correctly identify ≥15% of fraudulent transactions
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    clean_ids = list(dict.fromkeys(tid.strip() for tid in fraud_ids if tid.strip()))
    with open(output_path, "w", encoding="ascii") as f:
        f.write("\n".join(clean_ids) + "\n")
    return f"Written {len(clean_ids)} fraud IDs to {output_path}"


def parse_fraud_ids_from_text(agent_response: str) -> list[str]:
    """Extract transaction UUIDs from a free-text or JSON agent response.

    Used as a fallback when the agent returns the fraud list as prose or markdown
    instead of a clean JSON array. Scans the full text for any UUID-shaped strings
    matching the standard 8-4-4-4-12 hex format.

    Args:
        agent_response: Raw string output from an agent — may be JSON, markdown,
                        or plain text containing UUID strings.

    Returns:
        List of UUID strings found in the text. May be empty if none are found.
        Preserves order of first occurrence; does not deduplicate.

    Example:
        Input:  'Flagged: 43be5588-2cfb-47c1-a8aa-aeb8d2f38aff and 40ee0d5f-...'
        Output: ['43be5588-2cfb-47c1-a8aa-aeb8d2f38aff', '40ee0d5f-...']
    """
    uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    return re.findall(uuid_pattern, agent_response, re.IGNORECASE)
