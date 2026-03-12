import os
import time
import uuid
import requests
from langfuse import Langfuse

_client: Langfuse | None = None
_session_id: str | None = None


def wait_for_langfuse(
    max_wait_seconds: int = 300,
    poll_interval: int = 10,
) -> bool:
    """Ping the Langfuse host until it responds or timeout expires.

    Returns True if reachable, False if timeout exceeded.
    """
    host = os.environ.get("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
    health_url = f"{host}/api/public/health"
    deadline = time.time() + max_wait_seconds

    print(f"[Langfuse] Waiting for server at {host} ...")
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            resp = requests.get(health_url, timeout=5)
            if resp.status_code < 500:
                print(f"[Langfuse] Server reachable (HTTP {resp.status_code}) after {attempt} attempt(s).")
                return True
        except requests.RequestException:
            pass
        remaining = int(deadline - time.time())
        print(f"[Langfuse] Not ready yet. Retrying in {poll_interval}s (up to {remaining}s remaining)...")
        time.sleep(poll_interval)

    print(f"[Langfuse] Server unreachable after {max_wait_seconds}s. Continuing without Langfuse.")
    return False


def init_tracking(wait: bool = True) -> str:
    """Initialize Langfuse and create a session. Returns session_id for submission.

    Args:
        wait: If True, waits until the Langfuse server is reachable before initializing.
    """
    global _client, _session_id

    _session_id = str(uuid.uuid4())

    if wait:
        reachable = wait_for_langfuse()
    else:
        reachable = True

    if reachable:
        _client = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ.get("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
        )

    print(f"[Langfuse] Session ID: {_session_id}")
    return _session_id


def get_session_id() -> str | None:
    return _session_id


def trace_agent_call(name: str, input_data: dict, output_data: dict) -> None:
    """Log a single agent interaction to Langfuse."""
    if not _client or not _session_id:
        return
    _client.trace(name=name, session_id=_session_id, input=input_data, output=output_data)
    _safe_flush()


def flush() -> None:
    _safe_flush()


def _safe_flush() -> None:
    if not _client:
        return
    try:
        _client.flush()
    except Exception:
        pass
