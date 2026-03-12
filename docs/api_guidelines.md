Resource Management: Token Usage and Cost Tracking with Langfuse
For the challenge, all costs are tracked exclusively via Langfuse session IDs. This tutorial shows how to integrate Langfuse with LangChain to automatically track token usage, costs, and performance.

This tutorial teaches you how to monitor and manage resources when using AI agents with Langfuse as the observability platform. LangChain has a native Langfuse CallbackHandler that automatically captures token usage, costs, and latency.

Why Resource Management Matters
When building production AI agent systems, understanding resource usage is crucial:

Cost control - LLM API calls cost money; you need to track spending
Performance optimization - Token usage affects response times and costs
Budget planning - Predict costs before scaling your system
Debugging - Token metrics help identify inefficient patterns
What Are Tokens?
Tokens are the units that language models process. Roughly:

1 token ≈ 4 characters of English text
1 token ≈ 0.75 words
1000 tokens ≈ 750 words
When you call an agent, it uses:

Input tokens - Your question + system prompt + conversation history
Output tokens - The agent's response
Cache tokens - Optional caching for faster/cheaper repeated queries
What You'll Learn
In this tutorial, you'll:

Set up Langfuse tracing for LangChain using @observe() and CallbackHandler
Understand how Langfuse automatically tracks token usage and costs
Use session IDs to group multiple calls under a single session
Track costs across multiple agent calls
Generate unique session IDs for the challenge
Prerequisites
Before starting, make sure you have:

Python 3.10+ installed (see warning below about Python 3.14)
OpenRouter API key (get one free at https://openrouter.ai)
Langfuse credentials — You should have received LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST for the challenge
Completed Tutorial 01 - You should understand basic agent creation
⚠️ Python 3.14 Warning: Python 3.14 can cause compatibility issues with Langfuse. We recommend using Python 3.10, 3.11, 3.12, or 3.13 to avoid problems.

First time? See Tutorial 01 for detailed instructions on installing Python, setting up a virtual environment, and creating your API key.

Quick Setup Checklist
Install Python 3.10–3.13: Download from python.org — verify with python3 --version (avoid Python 3.14)
Create a virtual environment: python3 -m venv venv && source venv/bin/activate
Get an OpenRouter API key: Sign up at openrouter.ai → Keys → Create Key
Create a .env file in the project root with:
OPENROUTER_API_KEY=your-api-key-here
LANGFUSE_PUBLIC_KEY=pk-your-public-key-here
LANGFUSE_SECRET_KEY=sk-your-secret-key-here
LANGFUSE_HOST=https://challenges.reply.com/langfuse
TEAM_NAME=your-team-name
Installation
Install the required dependencies directly. This cell is self-contained—no external requirements.txt needed.

langfuse provides the @observe() decorator and CallbackHandler for automatic tracing
ulid-py generates unique session IDs
%pip install langchain langchain-openai langfuse python-dotenv ulid-py --quiet
Setup Model
Import the necessary libraries and configure the model. This is the same setup from Tutorial 01.

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load environment variables from .env file
load_dotenv()

# Chosen model identifier
model_id = "gpt-4o-mini"

# Configure OpenRouter model
model = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model=model_id,
    temperature=0.7,
    max_tokens=1000,
)

print(f"✓ Model configured: {model_id}")
Initialize Langfuse and Helper Functions
How Langfuse Works with LangChain
The integration combines two mechanisms:

@observe() decorator - Wraps a function to automatically create a Langfuse trace on each call. All Langfuse operations inside the decorated function are nested under that trace.
CallbackHandler() - Created inside the @observe() function, it automatically attaches to the current trace and captures LangChain-specific metrics (tokens, costs, latency).
Session tracking - Multiple calls can be grouped under the same session_id using langfuse_client.update_current_trace(session_id=...). This lets you group all calls from a single run together.
Unique session IDs - Generated with ulid in the format {TEAM_NAME}-{ULID} for easy identification.
What Gets Tracked Automatically
The CallbackHandler captures:

Inputs and outputs - All messages sent to and received from the model
Token usage - Input, output, and cache tokens (when available)
Costs - Automatically calculated based on model pricing
Latency - Time taken for each operation
Metadata - Model parameters, temperature, etc.
import ulid
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler

# Initialize Langfuse client
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
)

def generate_session_id():
    """Generate a unique session ID using TEAM_NAME and ULID."""
    return f"{os.getenv('TEAM_NAME', 'tutorial')}-{ulid.new().str}"

def invoke_langchain(model, prompt, langfuse_handler):
    """Invoke LangChain with the given prompt and Langfuse handler."""
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages, config={"callbacks": [langfuse_handler]})
    return response.content

@observe()
def run_llm_call(session_id, model, prompt):
    """Run a single LangChain invocation and track it in Langfuse."""
    # Update trace with session_id
    langfuse_client.update_current_trace(session_id=session_id)

    # Create Langfuse callback handler for automatic generation tracking
    # The handler will attach to the current trace created by @observe()
    langfuse_handler = CallbackHandler()

    # Invoke LangChain with Langfuse handler to track tokens and costs
    response = invoke_langchain(model, prompt, langfuse_handler)

    return response

print("✓ Langfuse initialized successfully")
print(f"✓ Public key: {os.getenv('LANGFUSE_PUBLIC_KEY', 'Not set')[:20]}...")
print("✓ Helper functions ready: generate_session_id(), invoke_langchain(), run_llm_call()")
Run a Single Traced Call
Now we'll use run_llm_call() - decorated with @observe() - to make a traced call. Here's what happens under the hood:

@observe() creates a new Langfuse trace when the function is called
update_current_trace(session_id=...) tags the trace with our session ID so all calls in this run are grouped together
CallbackHandler() is created inside the decorated function, so it automatically attaches to the current trace
Token usage, costs, and latency are all captured automatically
session_id = generate_session_id()
print(f"Session ID: {session_id}\n")

response = run_llm_call(session_id, model, "What is the square root of 144?")

print(f"\nInput:    What is the square root of 144?")
print(f"Response: {response}")

langfuse_client.flush()

print(f"\n✓ Trace sent to Langfuse with full token usage and cost data")
print(f"✓ Grouped under session: {session_id}")
print("✓ You can inspect this session using get_trace_info(session_id) and print_results(info) below.")
Track Multiple Calls with Session Grouping
Since every call to run_llm_call() shares the same session_id, all traces are grouped together. There's no need to manually accumulate tokens — Langfuse aggregates everything for you.

This is the key advantage over manual tracking:

No manual accumulation — Langfuse sums tokens and costs across all traces in a session
Automatic cost calculation — Based on Langfuse's built-in model pricing
Queryable traces — You (or the organizers) can query Langfuse by session_id using the Langfuse Python client or HTTP API to see usage and costs
Let's make multiple calls under the same session:

questions = [
    "What is machine learning?",
    "Explain neural networks briefly.",
    "What is the difference between AI and ML?"
]

session_id = generate_session_id()
print(f"Session ID: {session_id}")
print(f"Making {len(questions)} agent calls with Langfuse tracing...\n")

for i, question in enumerate(questions, 1):
    response = run_llm_call(session_id, model, question)
    print(f"Call {i}: {question[:40]}...")
    print(f"  Response: {response[:80]}...\n")

langfuse_client.flush()

print("=" * 50)
print(f"✓ All {len(questions)} traces sent to Langfuse!")
print(f"✓ All grouped under session: {session_id}")
print("✓ You can inspect this session using get_trace_info(session_id) and print_results(info) below.")
Viewing Your Traces (Integrated Viewer)
This notebook includes a built-in trace viewer that uses the Langfuse Python client to:

Fetch all traces for a given session_id
Aggregate number of generations per model
Aggregate cost per model and total cost
Sum total time spent across generations
Show a short preview of the first input and last output
You only need to:

Make sure your environment variables (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST) are set
Run the helper functions defined in the next cell
Call get_trace_info(session_id) and print_results(info) with the session ID you used when generating traces.
from datetime import datetime
from collections import defaultdict


def get_trace_info(session_id: str):
    """Fetch traces for a session_id and aggregate basic statistics.

    Returns a dict with:
      - counts: {model -> num_generations}
      - costs: {model -> total_cost}
      - time: total time across generations (seconds)
      - input: preview of first input
      - output: preview of last output
    """
    traces = []
    page = 1

    while True:
        response = langfuse_client.api.trace.list(session_id=session_id, limit=100, page=page)
        if not response.data:
            break
        traces.extend(response.data)
        if len(response.data) < 100:
            break
        page += 1

    if not traces:
        return None

    observations = []
    for trace in traces:
        detail = langfuse_client.api.trace.get(trace.id)
        if detail and hasattr(detail, "observations"):
            observations.extend(detail.observations)

    if not observations:
        return None

    sorted_obs = sorted(
        observations,
        key=lambda o: o.start_time if hasattr(o, "start_time") and o.start_time else datetime.min,
    )

    counts = defaultdict(int)
    costs = defaultdict(float)
    total_time = 0.0

    for obs in observations:
        if hasattr(obs, "type") and obs.type == "GENERATION":
            model = getattr(obs, "model", "unknown") or "unknown"
            counts[model] += 1

            if hasattr(obs, "calculated_total_cost") and obs.calculated_total_cost:
                costs[model] += obs.calculated_total_cost

            if hasattr(obs, "start_time") and hasattr(obs, "end_time"):
                if obs.start_time and obs.end_time:
                    total_time += (obs.end_time - obs.start_time).total_seconds()

    first_input = ""
    if sorted_obs and hasattr(sorted_obs[0], "input"):
        inp = sorted_obs[0].input
        if inp:
            first_input = str(inp)[:100]

    last_output = ""
    if sorted_obs and hasattr(sorted_obs[-1], "output"):
        out = sorted_obs[-1].output
        if out:
            last_output = str(out)[:100]

    return {
        "counts": dict(counts),
        "costs": dict(costs),
        "time": total_time,
        "input": first_input,
        "output": last_output,
    }


def print_results(info):
    """Pretty-print the aggregated trace information returned by get_trace_info."""
    if not info:
        print("\nNo traces found for this session_id.\n")
        return

    print("\nTrace Count by Model:")
    for model, count in info["counts"].items():
        print(f"  {model}: {count}")

    print("\nCost by Model:")
    total = 0.0
    for model, cost in info["costs"].items():
        print(f"  {model}: ${cost:.6f}")
        total += cost
    if total > 0:
        print(f"  Total: ${total:.6f}")

    print(f"\nTotal Time: {info['time']:.2f}s")

    if info["input"]:
        print(f"\nInitial Input:\n  {info['input']}")

    if info["output"]:
        print(f"\nFinal Output:\n  {info['output']}")

    print()


# Example usage (uncomment and set your session ID):
# session_id_to_check = "TEAMNAME-..."
# info = get_trace_info(session_id_to_check)
# print_results(info)
What You've Learned
Congratulations! You've learned how to monitor and manage resources for LangChain agents using Langfuse. Here's what we covered:

✅ Langfuse setup — Initializing the client, @observe() decorator, and CallbackHandler for automatic tracing
✅ Automatic token tracking — CallbackHandler captures all token usage from LangChain calls
✅ Automatic cost calculation — Langfuse calculates costs based on its built-in model pricing
✅ Session grouping — Using update_current_trace(session_id=...) to group calls
✅ Session ID generation — Creating unique IDs with {TEAM_NAME}-{ULID} format
✅ Trace checking — Using the integrated viewer get_trace_info(session_id) and print_results(info) to inspect your data

Key Takeaways
@observe() + CallbackHandler — The recommended pattern: decorate functions with @observe() and create CallbackHandler() inside to automatically attach to the current trace
Session tracking — Use generate_session_id() with ULID and update_current_trace(session_id=...) to group calls
langfuse_client.flush() — Always flush after your calls to ensure all traces are sent
No manual cost tracking needed — Langfuse handles token counting and cost calculation automatically
Check your traces — Use the integrated viewer (get_trace_info + print_results) in this notebook to see costs and trace details
How LangChain + Langfuse Tracing Works
@observe() decorated function
    ↓
Creates Langfuse trace → update_current_trace(session_id=...)
    ↓
CallbackHandler() attaches to current trace
    ↓
model.invoke(messages, config={"callbacks": [handler]})
    ↓
CallbackHandler captures: tokens, costs, latency, I/O
    ↓
langfuse_client.flush() → sends to Langfuse
    ↓
get_trace_info(session_id) + print_results(info) → view traces, costs, sessions
Cost Optimization Strategies
When building production systems:

Monitor regularly — Use the integrated viewer to check costs after each session
Choose models wisely — Balance cost vs. capability for your use case
Optimize prompts — Shorter prompts = fewer input tokens = lower costs
Start small, scale up — Begin with smaller models and only switch to larger ones if needed
Multi-agent strategy — Use larger models for critical decisions and smaller models for simpler tasks
Best Practices
Always set session IDs — Essential for the challenge; groups all costs under one session
Use @observe() + CallbackHandler — Wrap LLM-calling code so Langfuse captures everything automatically
Flush after calls — Call langfuse_client.flush() to ensure traces are sent
Generate unique session IDs — Use ULID to avoid collisions
Next Steps
Now that you understand resource management with Langfuse, you can:

Apply to your agents — Add Langfuse tracing to your own systems
Optimize existing agents — Use the integrated viewer to identify and reduce expensive operations
Build cost-aware systems — Design agents with cost efficiency in mind
Scale confidently — Understand costs before deploying at scale
Experiment
Try these modifications:

Add metadata — Use langfuse_client.update_current_trace(metadata={...}) to tag traces
Compare models — Use different models and compare costs with the integrated viewer
Add budget checking — Query Langfuse API to check session costs before making more calls
Track by agent type — Use different session IDs or tags for different agent types
For more information, visit the Langfuse documentation.