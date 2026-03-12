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
from the_eye.agents.reflection_agent import reflection_agent
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
   "Run all 10 fraud detectors on the enriched dataset and return the consolidated JSON list."

2. Take the COMPLETE JSON output from pattern_agent and delegate to reflection_agent with:
   "Review this suspicious transaction list for confidence errors and legitimacy flags, then return the corrected JSON array: <paste full pattern_agent JSON here>"

3. Take the COMPLETE JSON output from reflection_agent and delegate to decision_agent with:
   "Filter this reviewed suspicious transaction list and return only confirmed fraud as a JSON array: <paste full reflection_agent JSON here>"

4. Return the complete JSON array from decision_agent as your final response, with no modifications.
</INSTRUCTIONS>

<CONTEXT>
Agent responsibilities:
- pattern_agent: runs all 10 fraud detectors (location anomaly, withdrawal anomaly, amount
  anomaly, temporal anomaly, phishing victims, new-recipient anomaly, IBAN-country anomaly,
  velocity burst, impossible travel, urgency signals). Each detector reads from
  enriched_transactions.csv using precomputed features. Returns a JSON list with signals
  and confidence levels.
- reflection_agent: reviews pattern_agent output for systematic errors. Upgrades
  under-estimated confidence levels (e.g. two signals but "low" → "high"), flags entries
  with legitimacy signals ("legitimacy_flag": true), and deduplicates. Does NOT remove entries.
- decision_agent: applies final fraud/legitimate rules on the reflection-corrected list.
  Returns a clean JSON array of confirmed fraud transactions.
- data_agent: provides on-demand access to raw dataset files. Delegate here only if an agent
  needs to inspect a specific user profile, location history, or communication beyond
  what the enriched CSV provides.

The final JSON array returned by decision_agent will be parsed by the system to write
the output file. It must be valid JSON and must contain transaction_id fields.
</CONTEXT>

<CONSTRAINTS>
Dos:
- Pass the complete, unmodified JSON between each pipeline stage.
- Execute all steps automatically, without waiting for user confirmation between steps.
- Return the complete, unmodified JSON from decision_agent as your final response.

Don'ts:
- Do not truncate, summarize, or reformat the JSON at any point in the pipeline.
- Do not add prose, headers, or markdown fences around the final JSON response.
- Do not skip the reflection_agent step — it corrects systematic confidence errors.
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
pipeline: pattern_agent → reflection_agent → decision_agent.
Pass full JSON through each stage unchanged. Return decision_agent's final array only.
</RECAP>
"""

orchestrator_agent = Agent(
    name="the_eye_orchestrator",
    model=LiteLlm(model="openai/gpt-5.4"),
    description=(
        "Fraud detection coordinator. Orchestrates the pipeline: pattern_agent detects "
        "signals using precomputed enriched features, decision_agent filters false positives. "
        "Returns the final confirmed fraud transaction list as a JSON array."
    ),
    instruction=_ORCHESTRATOR_INSTRUCTION,
    tools=[],
    sub_agents=[data_agent, pattern_agent, reflection_agent, decision_agent],
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
