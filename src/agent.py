from pathlib import Path
import sys
import sqlite3
import json
import pandas as pd
import time
from datetime import datetime

# Make the project root available for imports
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import mock_llm
from src import policy_rag

# Project file paths
DB_PATH = PROJECT_DIR / "data" / "instamart.db"
OUTPUT_DIR = PROJECT_DIR / "outputs"
STATE_DIR = PROJECT_DIR / "state"
SCENARIOS_PATH = PROJECT_DIR / "data" / "raw" / "agent_scenarios.csv"

# Load policy documents and build the search index once
POLICY_CHUNKS = policy_rag.load_policy_chunks()
VECTORIZER, TFIDF_MATRIX = policy_rag.build_index(POLICY_CHUNKS)

def get_order(order_id):
    """Fetch order details from the SQLite database."""

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,)
        )

        row = cursor.fetchone()

        if row is None:
            return {"error": "order not found"}

        return {
            "order_id": row["order_id"],
            "status": row["status"],
            "order_amount": row["order_amount"],
            "delivery_minutes": row["delivery_minutes"]
        }

    finally:
        connection.close()

def search_policy(query):
    """Search the policy documents for relevant information."""

    results = policy_rag.retrieve(
        query,
        POLICY_CHUNKS,
        VECTORIZER,
        TFIDF_MATRIX,
        k=3
    )

    return {"results": results}

def calculate_refund(order_amount, delivery_minutes):
    """Calculate refund eligibility based on delivery time."""

    if delivery_minutes is not None and delivery_minutes > 45:
        refund_inr = float(order_amount)
    else:
        refund_inr = 0.0

    if refund_inr == 0:
        action = "no_refund"
    elif refund_inr > 1000:
        action = "escalate_to_human"
    else:
        action = "issue_refund"

    return {
        "refund_inr": refund_inr,
        "action": action
    }


def issue_refund(order_id, refund_inr):
    """Record an automatically issued refund without duplicates."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ledger_path = OUTPUT_DIR / "refund_ledger.jsonl"

    # Check whether a refund was already recorded for this order.
    if ledger_path.exists():
        with ledger_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                entry = json.loads(line)

                if (
                    entry.get("tool") == "issue_refund"
                    and entry.get("order_id") == order_id
                ):
                    existing_amount = float(entry["refund_inr"])

                    if existing_amount != float(refund_inr):
                        raise ValueError(
                            "A different refund already exists for this order."
                        )

                    return {
                        "refund_inr": existing_amount,
                        "status": "already_refunded"
                    }

    ledger_entry = {
        "timestamp": datetime.now().isoformat(),
        "tool": "issue_refund",
        "order_id": order_id,
        "refund_inr": float(refund_inr)
    }

    with ledger_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(ledger_entry) + "\n")

    return {
        "refund_inr": float(refund_inr),
        "status": "issued"
    }

def escalate_to_human(order_id, refund_inr):
    """Record a refund requiring human approval."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ledger_entry = {
        "timestamp": datetime.now().isoformat(),
        "tool": "escalate_to_human",
        "order_id": order_id,
        "refund_inr": float(refund_inr)
    }

    ledger_path = OUTPUT_DIR / "refund_ledger.jsonl"

    with ledger_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(ledger_entry) + "\n")

    return {"refund_inr": float(refund_inr)}

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Get order details using an order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search policy documents for relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate refund eligibility and determine whether human escalation is required.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {"type": ["number", "null"]}
                },
                "required": ["order_amount", "delivery_minutes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Record an approved automatic refund.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Send a refund requiring human approval for escalation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    }
]

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

AGENT_SYSTEM_PROMPT = """
You are an AGENT for Instamart customer support.

Use the available tools to investigate customer requests.

Rules:
1. Always check the order using get_order.
2. Search the relevant policy before deciding.
3. Calculate refund eligibility using calculate_refund.
4. Never issue a refund greater than INR 1000 automatically.
5. Use escalate_to_human for refunds greater than INR 1000.
6. Use issue_refund only when the calculated refund is greater than zero
   and does not exceed INR 1000.
7. If the order is not found, return NOT_FOUND.
8. If the order is cancelled or is not eligible for a refund, return DENIED.
9. Give a concise final answer beginning with REFUND, ESCALATE,
   DENIED, or NOT_FOUND as appropriate.
"""

def save_checkpoint(
    run_id,
    scenario_id,
    messages,
    tokens_used,
    step_number,
    steps,
    llm_calls
):
    """Save the agent's current progress for later resumption."""

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_number": step_number,
        "steps": steps,
        "llm_calls": llm_calls
    }

    checkpoint_path = STATE_DIR / f"{run_id}.json"

    with checkpoint_path.open("w", encoding="utf-8") as file:
        json.dump(checkpoint, file, indent=4)

    return checkpoint_path

def _continue_agent(
    run_id,
    scenario_id,
    messages,
    steps,
    llm_calls,
    tokens_used,
    max_iterations,
    token_limit,
    time_limit_seconds,
    crash_after_tools=None
):
    """Continue an agent run from its current conversation state."""

    start_time = time.monotonic()

    def fallback(status, explanation):
        return {
            "status": status,
            "final_answer": explanation,
            "steps": steps,
            "llm_calls": llm_calls,
            "tokens_used": tokens_used
        }

    while llm_calls < max_iterations:

        # Check the time limit before calling the model.
        if time.monotonic() - start_time >= time_limit_seconds:
            return fallback(
                "FALLBACK_TIMEOUT",
                "FALLBACK_TIMEOUT: Request handed off to a human because the time limit was reached."
            )

        response = mock_llm.chat(messages, tools=TOOLS_SPEC)

        llm_calls += 1
        tokens_used += response.usage.total_tokens

        # Stop before executing a tool if the token limit is exceeded.
        if tokens_used > token_limit:
            return fallback(
                "FALLBACK_TOKEN_LIMIT",
                "FALLBACK_TOKEN_LIMIT: Request handed off to a human because the token limit was exceeded."
            )

        message = response.choices[0].message

        # If there are no tool calls, the model has produced its final answer.
        if not message.tool_calls:
            final_answer = message.content or ""

            return {
                "status": final_answer.split(":")[0].strip(),
                "final_answer": final_answer,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used
            }

        # Add the assistant's tool requests to the conversation.
        assistant_tool_calls = []

        for tool_call in message.tool_calls:
            assistant_tool_calls.append({
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments)
                }
            })

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_tool_calls
        })

        # Execute each requested tool and save its observation.
        for tool_call in message.tool_calls:

            tool_name = tool_call.name
            arguments = tool_call.arguments

            if tool_name not in TOOL_MAP:
                observation = {"error": "unknown tool"}
            else:
                observation = TOOL_MAP[tool_name](**arguments)

            steps.append({
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(observation)
            })

            # Save only after the tool observation is recorded.
            save_checkpoint(
                run_id=run_id,
                scenario_id=scenario_id,
                messages=messages,
                tokens_used=tokens_used,
                step_number=len(steps),
                steps=steps,
                llm_calls=llm_calls
            )

            # This simulates a crash after a completed tool execution.
            if crash_after_tools is not None and len(steps) >= crash_after_tools:
                raise RuntimeError("Simulated crash after tool execution.")

    return fallback(
        "FALLBACK_MAX_ITERATIONS",
        "FALLBACK_MAX_ITERATIONS: Request handed off to a human because the maximum number of iterations was reached."
    )

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tools=None
):
    """Start a new customer support agent run."""

    if run_id is None:
        run_id = scenario_id

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    return _continue_agent(
        run_id=run_id,
        scenario_id=scenario_id,
        messages=messages,
        steps=[],
        llm_calls=0,
        tokens_used=0,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_after_tools=crash_after_tools
    )

def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
):
    """Resume an agent run from its saved checkpoint."""

    checkpoint_path = STATE_DIR / f"{run_id}.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for run: {run_id}"
        )

    with checkpoint_path.open("r", encoding="utf-8") as file:
        checkpoint = json.load(file)

    return _continue_agent(
        run_id=run_id,
        scenario_id=checkpoint["scenario_id"],
        messages=checkpoint["messages"],
        steps=checkpoint["steps"],
        llm_calls=checkpoint["llm_calls"],
        tokens_used=checkpoint["tokens_used"],
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds
    )

def run_all_scenarios():
    """Run all agent scenarios and save a fresh JSONL run log."""

    scenarios = pd.read_csv(SCENARIOS_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    log_path = OUTPUT_DIR / "agent_runs.jsonl"
    ledger_path = OUTPUT_DIR / "refund_ledger.jsonl"

    # Start with a clean ledger for this complete scenario run.
    ledger_path.write_text("", encoding="utf-8")

    with log_path.open("w", encoding="utf-8") as log_file:

        for _, scenario in scenarios.iterrows():

            scenario_id = scenario["scenario_id"]
            message = scenario["message"]

            print(f"\nRunning scenario: {scenario_id}")

            result = run_agent(
                user_message=message,
                scenario_id=scenario_id,
                run_id=scenario_id
            )

            record = {
                "scenario_id": scenario_id,
                "status": result["status"],
                "final_answer": result["final_answer"],
                "llm_calls": result["llm_calls"],
                "tokens_used": result["tokens_used"],
                "steps": result["steps"],
                "expected_outcome": scenario["expected_outcome"],
                "expected_refund_inr": scenario["expected_refund_inr"]
            }

            log_file.write(json.dumps(record) + "\n")

            print(f"Status: {result['status']}")
            print(f"Expected: {scenario['expected_outcome']}")

    print(f"\nAgent run log saved to: {log_path}")
    print(f"Total scenario records: {len(scenarios)}")

if __name__ == "__main__":
    run_all_scenarios()