
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

# --------------------------------------------------
# 1. PROJECT SETUP
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
STATE_DIR = OUTPUT_DIR / "state"

sys.path.insert(0, str(PROJECT_DIR))
import mock_llm

OUTPUT_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)

ORDERS_FILE = DATA_DIR / "clean" / "orders.csv"
SCENARIOS_FILE = DATA_DIR / "raw" / "agent_scenarios.csv"
REFUND_POLICY_FILE = DATA_DIR / "policies" / "refund_policy.txt"

RUNS_FILE = OUTPUT_DIR / "agent_runs.jsonl"
LEDGER_FILE = OUTPUT_DIR / "refund_ledger.jsonl"


# --------------------------------------------------
# 2. JSONL AND STATE MANAGEMENT
# --------------------------------------------------

def write_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def read_jsonl(path):
    if not path.exists():
        return []

    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def save_state(scenario_id, state):
    path = STATE_DIR / f"{scenario_id}.json"
    temp_path = path.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)

    temp_path.replace(path)


def load_state(scenario_id):
    path = STATE_DIR / f"{scenario_id}.json"

    if path.exists():
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    return None


# --------------------------------------------------
# 3. TOOL 1: GET ORDER
# --------------------------------------------------

def get_order(order_id):
    orders = pd.read_csv(ORDERS_FILE)

    match = orders[orders["order_id"] == order_id]

    if match.empty:
        return {
            "error": "Order not found",
            "order_id": order_id
        }

    order = match.iloc[0]

    delivery = order["delivery_minutes"]

    if pd.isna(delivery):
        delivery = None
    else:
        delivery = float(delivery)

    return {
        "order_id": str(order["order_id"]),
        "status": str(order["status"]),
        "order_amount": float(order["order_amount"]),
        "delivery_minutes": delivery
    }


# --------------------------------------------------
# 4. TOOL 2: SEARCH POLICY
# --------------------------------------------------

def search_policy(question):
    if not REFUND_POLICY_FILE.exists():
        return {"error": "Refund policy not found"}

    policy = REFUND_POLICY_FILE.read_text(encoding="utf-8")

    return {
        "source": "refund_policy.txt",
        "policy": policy,
        "question": question
    }


# --------------------------------------------------
# 5. TOOL 3: CALCULATE REFUND
# --------------------------------------------------

def calculate_refund(order_amount, delivery_minutes, status):
    if status != "completed":
        amount = 0
        reason = "Only completed orders are eligible."

    elif delivery_minutes is None or delivery_minutes <= 45:
        amount = 0
        reason = "Delivery was not more than 45 minutes."

    else:
        amount = round(float(order_amount), 2)

        if amount > 1000:
            reason = "Refund above INR 1000 requires human approval."
        else:
            reason = "Eligible for automatic refund."

    # refund_inr is required by mock_llm._agent
    return {
        "refund_inr": amount,
        "refund_amount": amount,
        "reason": reason
    }


# --------------------------------------------------
# 6. TOOL 4: ISSUE REFUND
# --------------------------------------------------

def issue_refund(order_id, amount, scenario_id):
    amount = round(float(amount), 2)

    idempotency_key = f"{scenario_id}-{order_id}"

    existing_records = read_jsonl(LEDGER_FILE)

    # Prevent duplicate refunds
    for entry in existing_records:
        if entry.get("idempotency_key") == idempotency_key:
            return {
                "status": "already_issued",
                "refund_inr": entry["refund_amount"],
                "refund_amount": entry["refund_amount"],
                "message": "Refund was already recorded."
            }

    record = {
        "idempotency_key": idempotency_key,
        "scenario_id": scenario_id,
        "order_id": order_id,
        "refund_inr": amount,
        "refund_amount": amount,
        "status": "issued",
        "timestamp": datetime.now().isoformat()
    }

    write_jsonl(LEDGER_FILE, record)

    return {
        "status": "issued",
        "refund_inr": amount,
        "refund_amount": amount,
        "message": "Refund recorded successfully."
    }


# --------------------------------------------------
# 7. TOOL 5: ESCALATE TO HUMAN
# --------------------------------------------------

def escalate_to_human(order_id, amount, reason):
    return {
        "status": "escalated",
        "order_id": order_id,
        "refund_inr": float(amount),
        "amount": float(amount),
        "reason": reason,
        "message": "Sent to human support for approval."
    }


# --------------------------------------------------
# 8. TOOL EXECUTION
# --------------------------------------------------

def execute_tool(name, arguments, scenario_id):

    if isinstance(arguments, str):
        arguments = json.loads(arguments)

    if name == "get_order":
        return get_order(arguments["order_id"])

    if name == "search_policy":
        return search_policy(
            arguments.get("question", "Refund policy")
        )

    if name == "calculate_refund":

        order_id = arguments.get("order_id")

        # Use actual order status whenever order ID is provided
        if order_id:
            order_details = get_order(order_id)
            status = order_details.get("status", "completed")
        else:
            status = arguments.get("status", "completed")

        return calculate_refund(
            arguments["order_amount"],
            arguments.get("delivery_minutes"),
            status
        )

    if name == "issue_refund":
        return issue_refund(
            arguments["order_id"],
            arguments.get("refund_inr", arguments.get("amount")),
            scenario_id
        )

    if name == "escalate_to_human":
        return escalate_to_human(
            arguments["order_id"],
            arguments.get(
                "refund_inr",
                arguments.get("amount", 0)
            ),
            arguments.get("reason", "Human approval required")
        )

    return {"error": f"Unknown tool: {name}"}


# --------------------------------------------------
# 9. OUTCOME DETECTION
# --------------------------------------------------

def classify_outcome(final_text, tool_results, fallback=False):

    if fallback:
        return "FALLBACK"

    text = (final_text or "").upper()

    if text.startswith("REFUND:"):
        return "REFUND"

    if text.startswith("ESCALATE:"):
        return "ESCALATE"

    if text.startswith("NOT_FOUND"):
        return "NOT_FOUND"

    if text.startswith("DENIED"):
        return "DENIED"

    for result in tool_results:
        if result.get("error") == "Order not found":
            return "NOT_FOUND"

    return "FALLBACK"


# --------------------------------------------------
# 10. REFUND REACT AGENT
# --------------------------------------------------

def run_scenario(scenario, crash_after=None, resume=False):

    scenario_id = str(scenario["scenario_id"])
    customer_message = str(scenario["message"])

    print(f"\nProcessing {scenario_id}...")

    state = load_state(scenario_id) if resume else None

    if state and state.get("completed"):
        print(f"{scenario_id} already completed. Skipping.")
        return True

    if state is None:
        state = {
            "scenario_id": scenario_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a customer support refund AGENT. "
                        "Check orders and follow the refund policy."
                    )
                },
                {
                    "role": "user",
                    "content": customer_message
                }
            ],
            "tool_results": [],
            "tool_steps": 0,
            "completed": False
        }

    consecutive_same_tool = 0
    previous_tool = None
    final_text = ""
    fallback = False

    for _ in range(12):

        response = mock_llm.chat(state["messages"])

        message = response.choices[0].message

        final_text = message.content or ""
        tool_calls = message.tool_calls or []

        if not tool_calls:
            break

        for tool_call in tool_calls:

            tool_name = tool_call.name
            tool_arguments = tool_call.arguments

            # Detect repeated tool calls
            if tool_name == previous_tool:
                consecutive_same_tool += 1
            else:
                consecutive_same_tool = 1

            previous_tool = tool_name

            if consecutive_same_tool >= 3:
                print("Repeated tool loop detected.")
                print("Safe fallback activated.")
                fallback = True
                break

            result = execute_tool(
                tool_name,
                tool_arguments,
                scenario_id
            )

            state["tool_results"].append(result)
            state["tool_steps"] += 1

            state["messages"].append({
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(result)
            })

            # Save progress after every tool
            save_state(scenario_id, state)

            print(f"  Tool: {tool_name}")

            # Simulate a crash for resume testing
            if crash_after is not None:
                if state["tool_steps"] >= crash_after:
                    print("\nSIMULATED CRASH: Progress saved.")
                    return False

        if fallback:
            break

    if fallback:
        final_text = "FALLBACK: Repeated tool calls detected."

    outcome = classify_outcome(
        final_text,
        state["tool_results"],
        fallback
    )

    state["completed"] = True
    state["final_text"] = final_text
    state["outcome"] = outcome

    save_state(scenario_id, state)

    run_record = {
        "scenario_id": scenario_id,
        "customer_id": str(scenario["customer_id"]),
        "message": customer_message,
        "expected_outcome": str(scenario["expected_outcome"]),
        "actual_outcome": outcome,
        "final_answer": final_text,
        "tool_steps": state["tool_steps"],
        "timestamp": datetime.now().isoformat()
    }

    write_jsonl(RUNS_FILE, run_record)

    print(f"  Expected: {scenario['expected_outcome']}")
    print(f"  Actual:   {outcome}")
    print(f"  Answer:   {final_text}")

    return True


# --------------------------------------------------
# 11. MAIN PROGRAM
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Instamart Refund ReAct Agent"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume saved progress"
    )

    parser.add_argument(
        "--crash-after",
        type=int,
        default=None,
        help="Simulate crash after a number of tool calls"
    )

    args = parser.parse_args()

    scenarios = pd.read_csv(SCENARIOS_FILE)

    print("=" * 55)
    print("INSTAMART REFUND REACT AGENT")
    print("=" * 55)

    print(f"Scenarios loaded: {len(scenarios)}")

    for _, scenario in scenarios.iterrows():

        completed = run_scenario(
            scenario,
            crash_after=args.crash_after,
            resume=args.resume
        )

        if not completed:
            print("\nStop-safe checkpoint saved.")
            print(
                "Restart with: "
                "python3 src/refund_agent.py --resume"
            )
            return

        args.crash_after = None

    print("\n" + "=" * 55)
    print("AGENT RUN SUMMARY")
    print("=" * 55)

    results = read_jsonl(RUNS_FILE)

    latest = {}

    for record in results:
        latest[record["scenario_id"]] = record

    for scenario_id, record in latest.items():
        print(
            f"{scenario_id}: expected={record['expected_outcome']}, "
            f"actual={record['actual_outcome']}"
        )

    print(f"\nRun log: {RUNS_FILE}")
    print(f"Refund ledger: {LEDGER_FILE}")
    print(f"Saved states: {STATE_DIR}")

    print("\nREFUND AGENT COMPLETED!")


if __name__ == "__main__":
    main()

