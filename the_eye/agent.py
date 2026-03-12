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
You have four sub-agents: pattern_agent, reflection_agent, decision_agent, data_agent.
</OBJECTIVE_AND_PERSONA>

<INSTRUCTIONS>
Follow these steps in order without asking the user for input:

Step 1 — Fraud signal sweep (delegate to pattern_agent):
  Send pattern_agent exactly this message:
  "Run all 10 fraud detectors on the enriched dataset and return the consolidated JSON list."
  Wait for pattern_agent to return a JSON array.

Step 2 — Confidence review and legitimacy flagging (delegate to reflection_agent):
  Take the COMPLETE JSON array from pattern_agent and send reflection_agent:
  "Review this suspicious transaction list for confidence errors and legitimacy flags,
   then return the corrected JSON array: <paste full pattern_agent JSON here>"
  Do NOT truncate or summarise the JSON. Paste it in full.
  Wait for reflection_agent to return the corrected JSON array.

Step 3 — Final fraud/legitimate filtering (delegate to decision_agent):
  Take the COMPLETE JSON array from reflection_agent and send decision_agent:
  "Filter this reviewed suspicious transaction list and return only confirmed fraud
   as a JSON array: <paste full reflection_agent JSON here>"
  Do NOT truncate or summarise the JSON. Paste it in full.
  Wait for decision_agent to return the final JSON array.

Step 4 — Return final result:
  Return the complete JSON array from decision_agent as your final response, unchanged.
</INSTRUCTIONS>

<SUB_AGENT_REFERENCE>
pattern_agent
    Purpose   : Runs all 10 fraud signal detectors on enriched_transactions.csv using
                precomputed features (no raw CSV parsing needed), then merges results by
                transaction_id, assigns confidence levels, and returns a deduplicated JSON list.
    When to delegate: Always — this is Step 1 of every pipeline run.
    What to send: "Run all 10 fraud detectors on the enriched dataset and return the
                  consolidated JSON list."
    What to expect: A JSON array where each element has:
                    - "transaction_id" (str): UUID
                    - "signals" (list[str]): signal names that fired, e.g. ["gps_mismatch", "temporal_anomaly"]
                    - "details" (str): one sentence of evidence
                    - "confidence" (str): "high", "medium", or "low"
    Valid signal names: "gps_mismatch", "withdrawal_anomaly", "amount_anomaly",
                        "temporal_anomaly", "phishing_exposure", "new_recipient",
                        "iban_country_mismatch", "velocity_burst", "impossible_travel", "urgency_signal"

reflection_agent
    Purpose   : Reviews pattern_agent's JSON list for three systematic issues:
                (1) under-estimated confidence — upgrades based on signal count and special rules;
                (2) legitimacy flagging — adds "legitimacy_flag": true for obvious false positives;
                (3) deduplication — merges duplicate transaction_id entries.
                Never removes entries; never downgrades confidence.
    When to delegate: Always — immediately after receiving pattern_agent's output (Step 2).
                      Skipping this step risks sending miscalibrated confidence levels to decision_agent.
    What to send: The full pattern_agent JSON array (unmodified) in the message body.
    What to expect: The same JSON array with corrected "confidence" values and optional
                    "legitimacy_flag": true fields added.  Entry count may decrease only
                    if duplicate transaction_ids were merged.

decision_agent
    Purpose   : Applies final fraud/legitimate decision rules to the reflection-corrected list.
                Keeps all "high" confidence entries, drops "medium" entries with legitimacy signals,
                and drops "low" entries without corroboration.
                Returns only confirmed fraud transactions as a clean JSON array.
    When to delegate: Always — immediately after receiving reflection_agent's output (Step 3).
    What to send: The full reflection_agent JSON array (unmodified) in the message body.
    What to expect: A JSON array (may be shorter than input) where each element has:
                    - "transaction_id" (str): UUID
                    - "signals" (list[str]): preserved from pattern_agent
                    - "confidence" (str): preserved from reflection_agent
                    This array is the final pipeline output — return it verbatim.

data_agent
    Purpose   : On-demand access to raw dataset files: transaction statistics, user profiles
                (salary, residence, phishing susceptibility), GPS biotag location history,
                and SMS/email communications.
    When to delegate: ONLY when a sub-agent or the investigation requires a raw data lookup
                      that goes beyond what the enriched CSV provides — for example, to read
                      the exact text of a phishing email, or to verify a sender's salary for a
                      specific transaction that was borderline.  Do NOT delegate here routinely;
                      pattern_agent reads enriched_transactions.csv directly without needing
                      data_agent for its normal detection work.
    What to send: A natural-language query describing what data you need, e.g.:
                  "Get the full profile for user with IBAN IT60L0100803268000000246810"
                  "Get the GPS history for sender CLLT-ZCHR-7FA-RUE-0"
                  "Get the communications for user Zacharie Collet"
    What to expect: Plain text (for transaction_summary) or a JSON string (all other tools).
</SUB_AGENT_REFERENCE>

<CONSTRAINTS>
Dos:
- Pass the complete, unmodified JSON between each pipeline stage.
- Execute all steps automatically, without waiting for user confirmation between steps.
- Return the complete, unmodified JSON from decision_agent as your final response.
- Delegate to data_agent if and only if a targeted raw-data lookup is genuinely needed.

Don'ts:
- Do not truncate, summarise, or reformat the JSON at any point in the pipeline.
- Do not add prose, headers, or markdown fences around the final JSON response.
- Do not skip the reflection_agent step — it corrects systematic confidence errors that
  pattern_agent may introduce.
- Do not ask the user for clarification or additional input.
- Do not modify or filter the JSON yourself — let each sub-agent do its own job.
</CONSTRAINTS>

<OUTPUT_FORMAT>
Your final response must be exactly the JSON array returned by decision_agent.
No text before or after the array.

Example of a valid final response:
[
  {"transaction_id": "00000001-0000-0000-0000-000000000001", "signals": ["gps_mismatch"], "confidence": "high"},
  {"transaction_id": "00000002-0000-0000-0000-000000000002", "signals": ["temporal_anomaly", "phishing_exposure"], "confidence": "high"}
]
</OUTPUT_FORMAT>

<RECAP>
Pipeline: pattern_agent → reflection_agent → decision_agent.
Pass full JSON through each stage unchanged.  Return decision_agent's final array only.
Use data_agent only for targeted raw-data lookups — never as part of the main pipeline.
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
