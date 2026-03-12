"""
Entry point for running The Eye programmatically.
Usage: python main.py --dataset "data/Brave New World_train/public" --output output/level1.txt

A timestamped working folder is created under workdir/ for each run.
The dataset is cloned there and all outputs (enriched CSV, predictions) live in that folder.
"""
import asyncio
import argparse
import logging
import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Suppress OpenTelemetry OTLP export timeout noise — ADK sends traces to
# challenges.reply.com which can time out; this does not affect our output.
logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)
logging.getLogger("opentelemetry.sdk").setLevel(logging.CRITICAL)
from dotenv import load_dotenv

load_dotenv()

# LiteLLM routes to OpenRouter using these env vars
os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))
os.environ.setdefault("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

# Parse args early so config is set before agents are imported
parser = argparse.ArgumentParser(description="The Eye - Fraud Detection Agent")
parser.add_argument("--dataset", default=None)
parser.add_argument("--output", default=None)
args = parser.parse_args()

# Set initial paths from args / env
from the_eye import config
config.set_paths(args.dataset, args.output)

# --- Create timestamped working folder ---
# e.g.  workdir/Brave New World_train_20260312_143021/
_dataset_dir = Path(config.DATASET_PATH)
_level_name  = _dataset_dir.parent.name           # "Brave New World_train"
_timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
workdir      = Path("workdir") / f"{_level_name}_{_timestamp}"
workdir.mkdir(parents=True, exist_ok=True)

# Clone the dataset into the working folder so all artefacts stay together
_data_dst = workdir / "data"
shutil.copytree(str(_dataset_dir), str(_data_dst))

# Resolve output path: use explicit arg if given, otherwise put it inside workdir
_output_path = config.OUTPUT_PATH or str(workdir / "predictions.txt")

# Lock in the final paths for the whole run (must happen before importing agents)
config.set_paths(
    dataset_path=str(_data_dst),
    output_path=_output_path,
    workdir_path=str(workdir),
)

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from the_eye.agent import root_agent
from the_eye.tracking.langfuse_tracker import init_tracking, get_session_id, flush


async def run() -> None:
    langfuse_session_id = init_tracking(wait=True)

    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name="the_eye", session_service=session_service)

    user_id = "challenge_user"

    await session_service.create_session(
        app_name="the_eye", user_id=user_id, session_id=langfuse_session_id
    )

    print(f"\n[The Eye] Starting analysis...")
    print(f"[The Eye] Dataset : {config.DATASET_PATH}")
    print(f"[The Eye] Workdir : {config.WORKDIR_PATH}")
    print(f"[The Eye] Output  : {config.OUTPUT_PATH}")
    print(f"[Langfuse] Session ID: {langfuse_session_id}\n")

    message = types.Content(
        role="user",
        parts=[types.Part(text="Analyze the dataset and write fraud predictions to the output file.")],
    )

    final_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=langfuse_session_id,
        new_message=message,
    ):
        if event.is_final_response():
            final_text = "".join(p.text for p in event.content.parts if hasattr(p, "text"))

    # Parse fraud IDs from the orchestrator's final response and write output file
    from the_eye.tools.output_writer import write_predictions, parse_fraud_ids_from_text
    import json as _json
    try:
        items = _json.loads(final_text)
        fraud_ids = [item["transaction_id"] for item in items if "transaction_id" in item]
    except Exception:
        fraud_ids = parse_fraud_ids_from_text(final_text)
        items = []

    # Hard cap: output must not exceed 30% of dataset size.
    # If over the cap, keep only "high" confidence entries.
    total_txns = sum(1 for _ in open(Path(config.DATASET_PATH) / "transactions.csv")) - 1
    cap = max(5, int(total_txns * 0.50))
    if len(fraud_ids) > cap and items:
        high_ids = [
            item["transaction_id"] for item in items
            if "transaction_id" in item and item.get("confidence") == "high"
        ]
        if high_ids:
            fraud_ids = high_ids
            print(f"[The Eye] Cap applied: {len(items)} → {len(fraud_ids)} (high-confidence only)")

    result = write_predictions(fraud_ids, config.OUTPUT_PATH)
    print(f"\n[The Eye] {result}")
    print(f"[The Eye] Run artefacts in: {workdir}")

    flush()
    print(f"\n[Submit this Session ID to Langfuse]: {get_session_id()}")


if __name__ == "__main__":
    asyncio.run(run())
