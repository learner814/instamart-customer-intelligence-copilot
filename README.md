# Instamart Customer Intelligence & Support Copilot

## Overview

The Instamart Customer Intelligence & Support Copilot is an offline Python capstone project combining data engineering, SQL analytics, machine learning, ticket classification, policy retrieval (RAG), and a tool-using refund support agent.

The project uses generated sample Instamart-style data and a local mock LLM. It does not connect to Instamart, a payment gateway, or an external LLM API. The supplied `generate_data.py` and `mock_llm.py` are used without modification.

## Reproduce

### Requirements

- Python 3.11 or later
- Packages listed in `requirements.txt`
- macOS, Linux, or Windows with Python available

### Setup

From the project root, create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1` and use `python` instead of `python3` if needed.

### Run the complete pipeline

Run these commands from the project root, in this order:

```bash
python3 generate_data.py
python3 src/clean_data.py
python3 src/validate.py
python3 src/customer_segments.py
python3 src/cluster_customers.py
python3 src/lapse_prediction.py
python3 src/ticket_triage.py
python3 src/policy_rag.py
python3 src/refund_agent.py
```

The scripts create or refresh the cleaned data, SQLite database, models, evaluation tables, logs, refund ledger, and checkpoint files under `data/`, `outputs/`, and `state/`.

## Part A

### Data generation and cleaning

`generate_data.py` creates synthetic customers, products, orders, support tickets, policy documents, policy evaluation questions, and refund-agent scenarios. `src/clean_data.py` audits and cleans the generated CSV data and builds `data/instamart.db` with the `customers`, `products`, and `orders` tables.

### SQL analysis

`sql/queries.sql` contains seven queries covering customer and product counts, order status, completed-order revenue, highest-spending customers, average delivery time by product category, and orders by city.

### Validation

`src/validate.py` checks expected row counts, unique order IDs, removal of pending orders, missing values, delivery-time conditions, completed revenue, city count, and SQLite tables. The reproduced run passed all 12 validation assertions.

## Part B

### RFM and customer segmentation

The project calculates customer recency, frequency, and monetary value, then applies K-Means clustering. It compares cluster counts using inertia and silhouette scores, profiles the four selected customer segments, and uses PCA to reduce the feature space for analysis. Outputs include `outputs/rfm_analysis.csv` and `outputs/customer_segments.csv`.

### Lapse prediction

`src/lapse_prediction.py` builds pre-September 2025 customer features and uses September 2025 completed-order activity as the lapse label. Features are created from orders before 1 September 2025; September activity is used for the target label, not as a predictor. Logistic Regression and Random Forest models are trained and evaluated, including a class-balancing experiment. The final fitted model is saved as `outputs/lapse_model.joblib`, and a premium-customer retention shortlist is saved as `outputs/retention_list.csv`.

In the reproduced run, the Random Forest test F1-score was approximately 0.659. Model scores are based on synthetic data and are not evidence of real-world business performance.

## Part C

### Privacy-aware support ticket triage

`src/ticket_triage.py` processes 164 generated tickets. It applies empty-input and estimated-token-length guardrails, redacts email addresses, phone numbers, and order IDs before model calls, and uses zero-shot and few-shot classification prompts for comparison.

The implementation includes JSON response validation, a single-category keyword fallback, retry handling with exponential jitter, token-budget accounting, and final routing to human review, security, premium, or general support queues. It writes:

- `outputs/triage_prompt_comparison.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/triage_results.csv`

In the reproduced run, 4 empty and 4 long tickets were guarded, all 278 planted PII instances were redacted, and few-shot accuracy was 100% across the 147 tickets that received a valid classification. Accuracy is measured against the generated ticket labels.

## Part D
ReAct vs Plan-and-Execute
- ReAct combines reasoning and actions step by step. The agent observes the result of each tool call before deciding what to do next.
- Plan-and-Execute creates a complete plan first and then executes the planned steps.
- This project uses ReAct because refund decisions depend on order details and policy information retrieved during execution. It allows the agent to adjust its next action based on the tool observations.

### Policy RAG

Five text policy documents are split into 40-word chunks with 10-word overlap. A default `TfidfVectorizer()` indexes the chunks, and cosine similarity retrieves the top three chunks for a question. The policy assistant is instructed to answer only from retrieved context and to return a fixed fallback when the information is unavailable.

The evaluation uses 14 questions: 12 answerable and 2 unsupported. The project also repeats the evaluation with zero overlap to demonstrate the effect of chunk boundaries. Outputs are `outputs/policy_eval_results.csv`, `outputs/policy_eval_zero_overlap.csv`, and `outputs/policy_rag_report.md`.

In the reproduced run, the 10-word-overlap configuration achieved 12/12 source matches, 12/12 expected-fact matches, and 2/2 correct unsupported-question fallbacks. The zero-overlap run achieved 10/12 expected-fact matches, illustrating why overlap is useful.

### Refund ReAct agent

`src/refund_agent.py` implements five tools: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, and `escalate_to_human`. The agent checks order status and delivery time, applies the refund threshold, and requires human approval for refunds above INR 1000. It also uses iteration, token, and time limits with human-handoff fallbacks.

The agent records tool observations in JSONL logs, writes a refund ledger, checkpoints after tool observations, and supports resuming an interrupted run. Idempotency checks prevent duplicate refund entries. Seven scenarios cover automatic refunds, escalation, denials, missing orders, and a repeated-tool loop. Additional demonstrations cover token-limit, timeout, and maximum-iteration fallbacks.

The reproduced run matched all seven expected scenario outcomes. The resume demonstration completed the tool sequence `get_order`, `search_policy`, `calculate_refund`, and `issue_refund`.

## Limitations and Next Steps

- All customer, order, ticket, and policy-evaluation data are synthetic.
- The local mock LLM simulates model responses and transient failures; it is not a production language model.
- The SQLite database and refund ledger are local demonstrations, not live Instamart or payment systems.
- The machine-learning results may not generalise to real customer behaviour.
- A production version would require approved data access, security review, monitoring, human approval workflows, model evaluation on representative data, and integration testing.
