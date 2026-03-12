"""
Entry point for running The Eye programmatically.
Usage: python main.py --dataset "data/Brave New World_train/public" --output output/level1.txt
"""
import asyncio
import argparse
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
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

# Set global config BEFORE importing agents (tools read from config at call time)
from the_eye import config
config.set_paths(args.dataset, args.output)

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
    print(f"[The Eye] Dataset: {config.DATASET_PATH}")
    print(f"[The Eye] Output:  {config.OUTPUT_PATH}")
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

    # Write predictions from final response (deterministic — does not rely on LLM tool call)
    from the_eye.tools.output_writer import write_predictions, parse_fraud_ids_from_text
    import json as _json
    try:
        items = _json.loads(final_text)
        fraud_ids = [item["transaction_id"] for item in items if "transaction_id" in item]
    except Exception:
        fraud_ids = parse_fraud_ids_from_text(final_text)

    result = write_predictions(fraud_ids, config.OUTPUT_PATH)
    print(f"\n[The Eye] {result}")

    flush()
    print(f"\n[Submit this Session ID to Langfuse]: {get_session_id()}")


if __name__ == "__main__":
    asyncio.run(run())
