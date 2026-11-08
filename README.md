# Instamart Customer Intelligence & Support Copilot

## 1. Project Overview

The Instamart Customer Intelligence & Support Copilot is a Python-based project that combines Data Analytics, Machine Learning, and AI-based techniques to understand customer behaviour and improve customer support operations.

The project analyses customer information, order history, and support tickets. It identifies customer segments, predicts customer lapse risk, classifies support tickets, retrieves information from policy documents, and demonstrates an automated refund support agent.

This is an educational prototype developed using sample datasets.

## 2. Project Objectives

- Generate and clean customer, product, order, and support ticket data.
- Perform SQL-based data analysis.
- Analyse customer purchasing behaviour using RFM analysis.
- Group customers using K-Means clustering.
- Predict customers who may stop purchasing.
- Classify customer support tickets into different categories.
- Retrieve relevant information from company policy documents.
- Automate refund-related support tasks using a ReAct-style agent.
- Demonstrate error handling, checkpointing, and duplicate-refund protection.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data cleaning and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning and clustering |
| Matplotlib | Data visualisation |
| SQLite | Database management |
| JSON and CSV | Data storage and exchange |
| VS Code | Development environment |

## 4. Project Modules

### Part A: Data Generation and Cleaning

Generated sample datasets containing:
- Customers
- Products
- Orders
- Customer support tickets

Cleaned the datasets by handling duplicate records, missing values, invalid values, and inconsistent formats.

Created a SQLite database and performed SQL queries to analyse the data.

Data validation was performed to ensure the cleaned datasets were suitable for further analysis.

### Part B: Customer Intelligence and Machine Learning

**1. RFM Analysis**

RFM stands for:
- Recency: How recently a customer purchased.
- Frequency: How often a customer purchased.
- Monetary: How much money a customer spent.

RFM analysis helps understand customer purchasing behaviour.

**2. Customer Segmentation**

Used the K-Means clustering algorithm to divide customers into four groups based on their purchasing behaviour.

PCA was used to reduce the dimensions of customer data for visualisation.

**3. Customer Lapse Prediction**

Developed Logistic Regression and Random Forest models to predict whether customers may become inactive.

The Random Forest model achieved an F1-score of approximately 0.66 on the test dataset.

### Part C: Customer Support Ticket Classification

Developed a ticket classification system to categorise customer support messages into:

- Billing
- Delivery
- Fraud
- Technical

Implemented:
- Input validation
- Personal information masking
- LLM response parsing
- Retry handling
- Invalid response handling
- Rule-based fallback
- Simulated budget-based routing

A local mock LLM was used to simulate AI responses and failure conditions. No external LLM API was connected.

### Part D: Policy RAG and Refund ReAct Agent

**1. Policy RAG System**

Created five policy documents:

- Delivery Policy
- Escalation Policy
- Payment Policy
- Privacy Policy
- Refund Policy

Developed a retrieval-based question-answering system that searches policy information and provides answers using the local mock LLM.

Evaluated the system using a set of 14 questions.

**2. Refund ReAct Agent**

Developed a refund support agent that uses the following tools:

- get_order
- search_policy
- calculate_refund
- issue_refund
- escalate_to_human

The agent checks order details, retrieves refund policies, calculates eligible refund amounts, processes eligible refunds, and escalates cases requiring human approval.

**3. Reliability and Safety**

Implemented:
- JSONL execution logs
- Refund ledger
- Checkpoint storage
- Resume functionality
- Duplicate-refund protection
- Repeated-tool loop detection
- Safe fallback handling

Tested seven scenarios covering successful refunds, escalation, denied refunds, missing orders, and repeated tool calls.

All seven scenarios produced their expected outcomes in the test run.

## 5. Project Folder Structure

```text
instamart-copilot/
│
├── data/
│   ├── raw/
│   ├── clean/
│   ├── policies/
│   └── instamart.db
│
├── src/
│   ├── clean_data.py
│   ├── validate.py
│   ├── customer_segments.py
│   ├── cluster_customers.py
│   ├── lapse_prediction.py
│   ├── ticket_triage.py
│   ├── policy_rag.py
│   └── refund_agent.py
│
├── sql/
│   └── queries.sql
│
├── outputs/
│   ├── rfm_analysis.csv
│   ├── customer_segments.csv
│   ├── retention_list.csv
│   ├── lapse_model.joblib
│   ├── triage_results.csv
│   ├── triage_budget_demo.csv
│   ├── policy_eval_results.csv
│   ├── agent_runs.jsonl
│   ├── refund_ledger.jsonl
│   └── state/
│
├── generate_data.py
├── mock_llm.py
└── README.md
```

## 6. Installation and Execution

### Step 1: Open the project folder

Open the `instamart-copilot` folder in Visual Studio Code.

### Step 2: Activate the virtual environment

```bash
source .venv/bin/activate
```

### Step 3: Generate sample data

```bash
python3 generate_data.py
```

### Step 4: Clean and validate the data

```bash
python3 src/clean_data.py
python3 src/validate.py
```

### Step 5: Run customer analysis

```bash
python3 src/customer_segments.py
python3 src/cluster_customers.py
python3 src/lapse_prediction.py
```

### Step 6: Run ticket classification

```bash
python3 src/ticket_triage.py
```

### Step 7: Run the policy RAG system

```bash
python3 src/policy_rag.py
```

### Step 8: Run the refund agent

```bash
python3 src/refund_agent.py
```

**Note:** Run the scripts in the order shown. The project uses generated sample data and a local mock LLM.

## 7. Testing and Results

The project includes the following tests:

- Data cleaning and validation checks.
- RFM customer analysis.
- K-Means clustering evaluation.
- PCA explained variance analysis.
- Logistic Regression and Random Forest model evaluation.
- Customer support ticket classification.
- Retry and fallback handling.
- Policy retrieval evaluation.
- Seven refund agent test scenarios.
- Crash and resume testing.
- Duplicate-refund verification.

The customer lapse prediction Random Forest model achieved approximately 0.66 F1-score on the test dataset.

The refund agent successfully matched the expected outcomes in all seven test scenarios.

These results apply to the generated sample data and test scenarios; they do not guarantee performance on real customer data.

## 8. Limitations

- The project uses generated sample data rather than actual Instamart customer data.
- The LLM is simulated using a local mock implementation.
- The system is not connected to the real Instamart platform.
- Refund processing is demonstrated using a local ledger, not a real payment gateway.
- Model performance may change when tested with real-world data.

## 9. Conclusion

The Instamart Customer Intelligence & Support Copilot demonstrates how Data Analytics, Machine Learning, policy retrieval, and automated support workflows can be combined in a single project.

It helps analyse customer behaviour, identify potentially inactive customers, classify support requests, retrieve policy information, and demonstrate refund handling.

This project provides practical experience in Python programming, data analysis, machine learning, SQL, and AI-based application development.

## 10. Author

**Project:** Instamart Customer Intelligence & Support Copilot

**Project Type:** Academic / Capstone Project