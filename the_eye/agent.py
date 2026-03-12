"""
Root agent for Google ADK — SequentialAgent pipeline:

  Step 1: preprocessing_agent  →  feature engineering (enriched_transactions.csv)
  Step 2: orchestrator_agent   →  fraud detection (pattern_agent → decision_agent)

ADK looks for `root_agent` in this module when running `adk run` or `adk web`.
"""
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm

from the_eye.agents.preprocessing_agent import preprocessing_agent
from the_eye.agents.data_agent import data_agent
from the_eye.agents.pattern_agent import pattern_agent
from the_eye.agents.decision_agent import decision_agent

# ── Step 2: fraud detection orchestrator ──────────────────────────────────────

_ORCHESTRATOR_INSTRUCTION = """
<OBJECTIVE_AND_PERSONA>
You are The Eye, MirrorPay's fraud detection coordinator in Reply Mirror (2087).
The preprocessing step has already run and produced enriched_transactions.csv with
14 precomputed features in the working directory.  Your objective is to orchestrate
the fraud detection pipeline and return the final confirmed fraud list.
</OBJECTIVE_AND_PERSONA>

<INSTRUCTIONS>
Follow these steps in order without asking the user for input:
1. Delegate to pattern_agent with the message:
   "Run all fraud detectors on the enriched dataset and return the consolidated JSON list."
2. Take the complete JSON output from pattern_agent and delegate to decision_agent with:
   "Filter this suspicious transaction list and return only confirmed fraud as a JSON array: <paste full pattern_agent output here>"
3. Return the complete JSON array from decision_agent as your final response, with no modifications.
</INSTRUCTIONS>

<CONTEXT>
Agent responsibilities:
- pattern_agent: runs all fraud detectors (location anomaly, withdrawal anomaly, amount
  anomaly, temporal anomaly, phishing victims, new-recipient anomaly, IBAN-country anomaly,
  velocity burst).  Each detector reads from enriched_transactions.csv and uses precomputed
  features (gps_distance_to_tx_km, amount_vs_salary_ratio, phishing_in_comms, etc.).
  Returns a JSON list with signals and confidence.
- decision_agent: filters that list by applying fraud/legitimate rules.
  Returns a JSON array of confirmed fraud transactions.
- data_agent: provides on-demand access to raw dataset files. Delegate here only if an agent
  needs to inspect a specific user profile, location history, or communication beyond
  what the enriched CSV provides.

The final JSON array returned by decision_agent will be parsed by the system to write
the output file.  It must be valid JSON and must contain transaction_id fields.
</CONTEXT>

<CONSTRAINTS>
Dos:
- Pass the complete, unmodified JSON from pattern_agent to decision_agent.
- Return the complete, unmodified JSON from decision_agent as your final response.
- Execute all steps automatically, without waiting for user confirmation between steps.

Don'ts:
- Do not truncate, summarize, or reformat the JSON at any point in the pipeline.
- Do not add prose, headers, or markdown fences around the final JSON response.
- Do not ask the user for clarification or additional input.
</CONSTRAINTS>

<OUTPUT_FORMAT>
Your final response must be exactly the JSON array returned by decision_agent.
No text before or after the array.

Example of a valid final response:
[
  {"transaction_id": "43be5588-2cfb-47c1-a8aa-aeb8d2f38aff", "signals": ["gps_mismatch"], "confidence": "high"},
  {"transaction_id": "40ee0d5f-53d3-493b-a888-ac59f77321f9", "signals": ["temporal_anomaly", "phishing_exposure"], "confidence": "high"}
]
</OUTPUT_FORMAT>

<RECAP>
Delegate to pattern_agent → pass its full output to decision_agent → return
decision_agent's JSON array unchanged as your final response.
</RECAP>
"""

orchestrator_agent = Agent(
    name="the_eye_orchestrator",
    model=LiteLlm(model="openai/gpt-4o-mini"),
    description=(
        "Fraud detection coordinator. Orchestrates the pipeline: pattern_agent detects "
        "signals using precomputed enriched features, decision_agent filters false positives. "
        "Returns the final confirmed fraud transaction list as a JSON array."
    ),
    instruction=_ORCHESTRATOR_INSTRUCTION,
    tools=[],
    sub_agents=[data_agent, pattern_agent, decision_agent],
)

# ── Root agent: deterministic sequential pipeline ─────────────────────────────

root_agent = SequentialAgent(
    name="the_eye",
    description=(
        "The Eye: full fraud detection pipeline for MirrorPay. "
        "Step 1 — preprocessing_agent computes 14 features and saves enriched_transactions.csv. "
        "Step 2 — orchestrator_agent runs all signal detectors and returns confirmed fraud IDs."
    ),
    sub_agents=[preprocessing_agent, orchestrator_agent],
)
