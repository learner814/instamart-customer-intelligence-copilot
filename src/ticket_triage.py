
import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import mock_llm

OUTPUT_DIR = PROJECT_DIR / "outputs"
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED = {"billing", "delivery", "technical", "fraud"}

TOKEN_BUDGET_DEMO = 6000
TOKEN_BUDGET_FINAL = 100000
MAX_RETRIES = 3
BASE_DELAY = 0.05

ZERO_SHOT_PROMPT = """
CLASSIFY the customer support ticket.
Choose exactly one category: billing, delivery, technical, fraud.
Return JSON only in this format:
{"category":"billing","reason":"short explanation"}
Do not include markdown or additional text.
"""

FEW_SHOT_PROMPT = """
CLASSIFY the customer support ticket.
Choose exactly one category: billing, delivery, technical, fraud.

EXAMPLES:
Message: Someone used my account without permission.
Output: {"category":"fraud","reason":"unauthorized account activity"}

Message: My parcel has not arrived and the rider is late.
Output: {"category":"delivery","reason":"late delivery"}

Message: I was charged twice for my order.
Output: {"category":"billing","reason":"duplicate payment"}

Message: The application crashes when I try to log in.
Output: {"category":"technical","reason":"application problem"}

Return JSON only:
{"category":"billing","reason":"short explanation"}
"""

# --------------------------------------------------
# LOAD TICKETS
# --------------------------------------------------

tickets = pd.read_csv(RAW_DIR / "tickets.csv")
tickets["message"] = tickets["text"].fillna("").astype(str)

print("CUSTOMER SUPPORT TICKET CLASSIFICATION")
print("Total tickets:", len(tickets))

# Add customer segment information when available.
segment_file = OUTPUT_DIR / "customer_segments.csv"

if segment_file.exists():
    segments = pd.read_csv(segment_file)

    if "customer_id" in segments.columns:
        extra_columns = [
            c for c in ["customer_id", "segment_name", "segment_id",
                        "is_premium"]
            if c in segments.columns
        ]

        if len(extra_columns) > 1:
            segment_data = segments[extra_columns].drop_duplicates(
                "customer_id"
            )
            tickets = tickets.merge(
                segment_data,
                on="customer_id",
                how="left",
                suffixes=("", "_segment")
            )

# Derive segment names only if they are not already available.
if "segment_name" not in tickets.columns:
    if "segment_id" in tickets.columns:
        names = {
            0: "Regular",
            1: "Premium",
            2: "Loyal",
            3: "At Risk"
        }
        tickets["segment_name"] = tickets["segment_id"].map(names)
    else:
        tickets["segment_name"] = "Unknown"

tickets["segment_name"] = (
    tickets["segment_name"].fillna("Unknown").astype(str)
)

if "is_premium" not in tickets.columns:
    tickets["is_premium"] = (
        tickets["segment_name"].str.lower().str.contains("premium")
    )

tickets["is_premium"] = tickets["is_premium"].fillna(False).astype(bool)

# --------------------------------------------------
# INPUT VALIDATION
# --------------------------------------------------

tickets["est_tokens"] = tickets["message"].apply(mock_llm.est_tokens)

empty_mask = tickets["message"].str.strip().eq("")
long_mask = tickets["est_tokens"] > 400

print("\nINPUT VALIDATION")
print("Empty messages:", int(empty_mask.sum()))
print("Long messages:", int(long_mask.sum()))

# --------------------------------------------------
# PII REDACTION
# --------------------------------------------------

redaction_counts = {
    "email": 0,
    "phone": 0,
    "order_id": 0
}


def redact_pii(message):
    message, n = re.subn(
        r"[\w.+-]+@[\w-]+\.[\w.]",
        "[EMAIL]",
        message
    )
    redaction_counts["email"] += n

    message, n = re.subn(
        r"\b[6-9]\d{9}\b",
        "[PHONE]",
        message
    )
    redaction_counts["phone"] += n

    message, n = re.subn(
        r"ORD-\d{6}",
        "[ORDER_ID]",
        message
    )
    redaction_counts["order_id"] += n

    return message


tickets["safe_message"] = tickets["message"].apply(redact_pii)

print("\nPII REDACTION")
print("Email redactions:", redaction_counts["email"])
print("Phone redactions:", redaction_counts["phone"])
print("Order ID redactions:", redaction_counts["order_id"])
print("Total redactions:", sum(redaction_counts.values()))

# --------------------------------------------------
# PROMPT AND CLASSIFICATION FUNCTIONS
# --------------------------------------------------

def keyword_fallback(text):
    words = re.findall(r"[a-z]+", text.lower())
    matches = [word for word in words if word in ALLOWED]
    unique_matches = set(matches)

    if len(unique_matches) == 1:
        return {
            "category": next(iter(unique_matches)),
            "status": "classified_keyword"
        }

    return {
        "category": "",
        "status": "flagged_invalid_output"
    }


def classify_one(message, prompt_type, spent, budget):
    system_prompt = (
        FEW_SHOT_PROMPT
        if prompt_type == "few_shot"
        else ZERO_SHOT_PROMPT
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]

    estimated_prompt = mock_llm.est_tokens(system_prompt) + mock_llm.est_tokens(message)

    # Reserve an estimated completion allowance before calling the model.
    if spent + estimated_prompt + 20 > budget:
        return {
            "category": "",
            "status": "budget_exhausted",
            "reason": "Token budget reached",
            "tokens_used": 0,
            "retries": 0
        }

    retries = 0

    while True:
        try:
            response = mock_llm.chat(messages)

            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = prompt_tokens + completion_tokens

            answer = response.choices[0].message.content or ""

            try:
                result = json.loads(answer)
                category = result.get("category", "").lower()

                if category not in ALLOWED:
                    raise ValueError("Invalid category")

                return {
                    "category": category,
                    "status": "classified_json",
                    "reason": result.get("reason", ""),
                    "tokens_used": total_tokens,
                    "retries": retries
                }

            except (json.JSONDecodeError, ValueError, AttributeError):
                fallback = keyword_fallback(answer)

                return {
                    "category": fallback["category"],
                    "status": fallback["status"],
                    "reason": "Keyword fallback after invalid JSON",
                    "tokens_used": total_tokens,
                    "retries": retries
                }

        except mock_llm.TransientLLMError:
            if retries >= MAX_RETRIES:
                return {
                    "category": "",
                    "status": "flagged_invalid_output",
                    "reason": "Retry limit reached",
                    "tokens_used": 0,
                    "retries": retries
                }

            retries += 1
            delay = BASE_DELAY * (2 ** (retries - 1)) * random.uniform(0.9, 1.15)

            print(
                f"Retry: ticket classification, "
                f"retry #{retries}, wait {delay:.4f} seconds"
            )
            time.sleep(delay)


def run_classification(prompt_type, budget):
    # Reset immediately before every independent full run.
    mock_llm.reset()

    rows = []
    spent = 0

    for _, row in tickets.iterrows():
        message = row["safe_message"]

        if not message.strip():
            result = {
                "category": "",
                "status": "skipped_empty",
                "reason": "Empty message",
                "tokens_used": 0,
                "retries": 0
            }

        elif row["est_tokens"] > 400:
            result = {
                "category": "",
                "status": "flagged_long",
                "reason": "Message exceeds 400 estimated tokens",
                "tokens_used": 0,
                "retries": 0
            }

        else:
            result = classify_one(
                message,
                prompt_type,
                spent,
                budget
            )

            spent += result["tokens_used"]

        rows.append({
            "ticket_id": row["ticket_id"],
            "customer_id": row["customer_id"],
            "true_category": row["true_category"],
            "category": result["category"],
            "status": result["status"],
            "reason": result["reason"],
            "tokens_used": result["tokens_used"],
            "retries": result["retries"]
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# ZERO-SHOT AND FEW-SHOT COMPARISON
# --------------------------------------------------

print("\nZERO-SHOT CLASSIFICATION")
zero_df = run_classification("zero_shot", TOKEN_BUDGET_FINAL)

print("\nFEW-SHOT CLASSIFICATION")
few_df = run_classification("few_shot", TOKEN_BUDGET_FINAL)


def show_metrics(name, df):
    classified = df[df["status"].isin(
        ["classified_json", "classified_keyword"]
    )]

    accuracy = (
        (classified["category"] == classified["true_category"]).mean()
        if len(classified) else 0
    )

    print(f"\n{name} RESULTS")
    print("Classified tickets:", len(classified))
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Tokens used:", int(df["tokens_used"].sum()))
    print("Retries:", int(df["retries"].sum()))
    print("\nStatus counts:")
    print(df["status"].value_counts())

    if len(classified):
        print("\nConfusion table:")
        print(pd.crosstab(
            classified["true_category"],
            classified["category"]
        ))

    return accuracy


zero_accuracy = show_metrics("ZERO-SHOT", zero_df)
few_accuracy = show_metrics("FEW-SHOT", few_df)

print("\nPROMPT COMPARISON")
print("Zero-shot accuracy:", round(zero_accuracy * 100, 2), "%")
print("Few-shot accuracy:", round(few_accuracy * 100, 2), "%")

# --------------------------------------------------
# ZERO-SHOT TOKEN BUDGET DEMONSTRATION
# --------------------------------------------------

print("\nZERO-SHOT TOKEN BUDGET DEMONSTRATION")

budget_df = run_classification(
    "zero_shot",
    TOKEN_BUDGET_DEMO
)

budget_df.to_csv(
    OUTPUT_DIR / "triage_budget_demo.csv",
    index=False
)

print("Token budget:", TOKEN_BUDGET_DEMO)
print("Tokens used:", int(budget_df["tokens_used"].sum()))
print("Budget statuses:")
print(budget_df["status"].value_counts())
print("Saved: outputs/triage_budget_demo.csv")

# --------------------------------------------------
# FINAL FEW-SHOT RUN
# --------------------------------------------------

print("\nFINAL FEW-SHOT CLASSIFICATION")

final_df = run_classification(
    "few_shot",
    TOKEN_BUDGET_FINAL
)

# Add segment and priority information.
segment_lookup = tickets.set_index("ticket_id")["segment_name"].to_dict()
premium_lookup = tickets.set_index("ticket_id")["is_premium"].to_dict()

final_df["segment_name"] = final_df["ticket_id"].map(segment_lookup)
final_df["is_premium"] = final_df["ticket_id"].map(premium_lookup)

def route_ticket(row):
    status = row["status"]
    category = row["category"]

    if status in {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted"
    }:
        return "human_review", "NORMAL"

    if category == "fraud":
        return "security_team", "HIGH"

    if row["is_premium"]:
        return "premium_desk", "HIGH"

    return "support_queue", "NORMAL"


routing = final_df.apply(route_ticket, axis=1)
final_df["route"] = routing.apply(lambda x: x[0])
final_df["priority"] = routing.apply(lambda x: x[1])

# Save only the assignment's required final output fields.
final_output = final_df[[
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used"
]].copy()

final_output.to_csv(
    OUTPUT_DIR / "triage_results.csv",
    index=False
)

print("\nFINAL STATUS SUMMARY")
print(final_output["status"].value_counts())

print("\nFINAL ROUTING SUMMARY")
print(final_output["route"].value_counts())

print("\nFINAL PRIORITY SUMMARY")
print(final_output["priority"].value_counts())

print("\nTOTAL FINAL TOKENS:", int(final_output["tokens_used"].sum()))
print("Saved: outputs/triage_results.csv")

print("\nTICKET TRIAGE COMPLETED SUCCESSFULLY!")

