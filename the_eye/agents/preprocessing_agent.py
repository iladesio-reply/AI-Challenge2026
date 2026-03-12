"""
Preprocessing Agent — Step 1 of the SequentialAgent pipeline.

Calls run_feature_engineering() to produce enriched_transactions.csv in WORKDIR_PATH,
then reports the feature-count summary.  All downstream fraud detectors automatically
use the enriched CSV once it exists.
"""
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from the_eye.tools.feature_engineering import run_feature_engineering

_INSTRUCTION = """
<OBJECTIVE_AND_PERSONA>
You are the Preprocessing Agent for The Eye, MirrorPay's fraud detection system (Reply Mirror 2087).
Your sole objective is to run the feature engineering pipeline and report its output.
</OBJECTIVE_AND_PERSONA>

<INSTRUCTIONS>
1. Call run_feature_engineering() — this computes 14 fraud-detection features for every
   transaction in the dataset and saves enriched_transactions.csv to the working directory.
2. Return the tool output verbatim as your final response. Do not add commentary.
</INSTRUCTIONS>

<CONSTRAINTS>
- Call run_feature_engineering() exactly once.
- Return the tool output verbatim.
- Do not call any other tools.
- Do not ask the user for input.
</CONSTRAINTS>
"""

preprocessing_agent = Agent(
    name="preprocessing_agent",
    model=LiteLlm(model="openai/gpt-4o-mini"),
    description=(
        "Runs the feature engineering pipeline: loads transactions, users, GPS locations, "
        "SMS and email data; computes 14 behavioural and geospatial features per transaction; "
        "saves enriched_transactions.csv to the working directory. "
        "Must execute before any fraud detector tools are called."
    ),
    instruction=_INSTRUCTION,
    tools=[run_feature_engineering],
)
