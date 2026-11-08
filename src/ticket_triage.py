
import json
import re
import sys
from pathlib import Path

import pandas as pd

# ==========================================
# 1. PROJECT PATH AND MOCK LLM IMPORT
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_DIR))

import mock_llm

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_CATEGORIES = ["billing", "delivery", "technical", "fraud"]

INPUT_RATE = 0.10
OUTPUT_RATE = 0.30
MAX_RETRIES = 3


# ==========================================
# 2. LOAD CUSTOMER SUPPORT TICKETS
# ==========================================

tickets = pd.read_csv(PROJECT_DIR / "data/raw/tickets.csv")

tickets["message"] = tickets["text"].fillna("").astype(str)

print("CUSTOMER SUPPORT TICKET CLASSIFICATION")
print("Total tickets:", len(tickets))


# ==========================================
# 3. INPUT VALIDATION
# ==========================================

empty_count = (tickets["message"].str.strip() == "").sum()
long_count = (tickets["message"].str.len() > 2000).sum()

print("\nINPUT VALIDATION")
print("Empty messages:", empty_count)
print("Long messages:", long_count)


# ==========================================
# 4. PII REDACTION
# ==========================================

def redact_pii(message):

    message = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[EMAIL]",
        message
    )

    message = re.sub(
        r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)",
        "[PHONE]",
        message
    )

    message = re.sub(
        r"\bORD-\d{6}\b",
        "[ORDER_ID]",
        message
    )

    return message


tickets["safe_message"] = tickets["message"].apply(redact_pii)


# ==========================================
# 5. RULE-BASED FALLBACK CLASSIFIER
# ==========================================

KEYWORDS = {
    "fraud": ["unauthorized", "fraud", "hacked", "stolen"],
    "billing": ["charged", "invoice", "bill", "refund", "deducted", "payment"],
    "technical": ["app", "crash", "login", "otp", "error", "freezes"],
    "delivery": ["arrived", "rider", "delivery", "delivered", "late"]
}


def fallback_classifier(message):

    text = message.lower()

    for category in ["fraud", "technical", "billing", "delivery"]:
        for word in KEYWORDS[category]:
            if word in text:
                return category

    return "technical"


# ==========================================
# 6. TOKEN AND COST CALCULATION
# ==========================================

def calculate_cost(prompt_tokens, completion_tokens):

    input_cost = (prompt_tokens / 1000) * INPUT_RATE
    output_cost = (completion_tokens / 1000) * OUTPUT_RATE

    return input_cost + output_cost


# ==========================================
# 7. MOCK LLM CLASSIFICATION
# ==========================================

def classify_ticket(message):

    system_prompt = """
    CLASSIFY the customer support message.

    EXAMPLES:
    Unauthorized transaction means fraud.
    App login problem means technical.
    Refund or payment issue means billing.
    Late parcel means delivery.

    Return only valid JSON:
    {"category": "billing", "reason": "short explanation"}

    Allowed categories: billing, delivery, technical, fraud.
    """

    user_prompt = "Customer message: " + message

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    retries = 0

    while retries < MAX_RETRIES:

        try:

            response = mock_llm.chat(messages)

            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens

            answer = response.choices[0].message.content

            try:

                result = json.loads(answer)

                category = result.get("category", "").lower()
                reason = result.get("reason", "No reason provided")

                if category not in ALLOWED_CATEGORIES:
                    raise ValueError("Invalid category")

                return {
                    "category": category,
                    "reason": reason,
                    "status": "llm_success",
                    "retries": retries,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens
                }

            except (json.JSONDecodeError, ValueError, AttributeError):

                category = fallback_classifier(message)

                return {
                    "category": category,
                    "reason": "Rule-based fallback for invalid LLM response",
                    "status": "fallback_invalid_response",
                    "retries": retries,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens
                }

        except mock_llm.TransientLLMError:

            retries += 1

    return {
        "category": fallback_classifier(message),
        "reason": "Rule-based fallback after retry limit",
        "status": "fallback_retry_limit",
        "retries": retries,
        "prompt_tokens": 0,
        "completion_tokens": 0
    }


# ==========================================
# 8. CLASSIFY ALL TICKETS
# ==========================================

mock_llm.reset()

results = []

for _, row in tickets.iterrows():

    message = row["safe_message"]

    if not message.strip():

        result = {
            "category": "technical",
            "reason": "Empty message; manual review recommended",
            "status": "fallback_empty_message",
            "retries": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0
        }

    else:

        result = classify_ticket(message)

    cost = calculate_cost(
        result["prompt_tokens"],
        result["completion_tokens"]
    )

    results.append({
        "ticket_id": row["ticket_id"],
        "customer_id": row["customer_id"],
        "order_id": row["order_id"],
        "safe_message": message,
        "predicted_category": result["category"],
        "reason": result["reason"],
        "status": result["status"],
        "retries": result["retries"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "estimated_cost_inr": round(cost, 6),
        "true_category": row["true_category"]
    })


# ==========================================
# 9. SAVE CLASSIFICATION RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "triage_results.csv",
    index=False
)

print("\nCLASSIFICATION SUMMARY")
print(results_df["predicted_category"].value_counts())

print("\nSTATUS SUMMARY")
print(results_df["status"].value_counts())

print("\nTOTAL RETRIES:", results_df["retries"].sum())

total_cost = results_df["estimated_cost_inr"].sum()

print("TOTAL ESTIMATED COST: INR", round(total_cost, 4))

print("\nSaved: outputs/triage_results.csv")


# ==========================================
# 10. BUDGET ROUTING DEMONSTRATION
# ==========================================

def run_budget_demo():

    demo_rows = []

    system_prompt = (
        "CLASSIFY. EXAMPLES: Return a JSON category and reason."
    )

    budgets = [0.05, 5.00]

    for budget in budgets:

        spent = 0.0

        for _, row in tickets.iterrows():

            message = row["safe_message"]

            estimated_prompt = (
                mock_llm.est_tokens(system_prompt)
                + mock_llm.est_tokens(message)
            )

            estimated_completion = 20

            estimated_cost = calculate_cost(
                estimated_prompt,
                estimated_completion
            )

            if not message.strip():

                route = "empty_message_manual_review"

            elif spent + estimated_cost <= budget:

                route = "mock_llm"
                spent += estimated_cost

            else:

                route = "rule_based_fallback"

            demo_rows.append({
                "ticket_id": row["ticket_id"],
                "budget_inr": budget,
                "estimated_cost_inr": round(estimated_cost, 6),
                "route": route,
                "cumulative_spend_inr": round(spent, 6)
            })

    demo_df = pd.DataFrame(demo_rows)

    demo_df.to_csv(
        OUTPUT_DIR / "triage_budget_demo.csv",
        index=False
    )

    print("\nBUDGET ROUTING SUMMARY")

    print(
        demo_df.groupby(["budget_inr", "route"]).size()
    )

    print("\nSaved: outputs/triage_budget_demo.csv")


run_budget_demo()

print("\nTICKET TRIAGE COMPLETED SUCCESSFULLY!")
