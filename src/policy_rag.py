
import re
import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

POLICY_DIR = PROJECT_DIR / "data" / "policies"
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

EVALUATION_FILE = RAW_DIR / "policy_eval.csv"
RESULT_FILE = OUTPUT_DIR / "policy_eval_results.csv"

sys.path.insert(0, str(PROJECT_DIR))
import mock_llm


# --------------------------------------------------
# 2. LOAD POLICY DOCUMENTS
# --------------------------------------------------

def load_policies():

    policies = {}

    for file_path in POLICY_DIR.glob("*.txt"):

        policies[file_path.name] = file_path.read_text(
            encoding="utf-8"
        )

    return policies


# --------------------------------------------------
# 3. SPLIT POLICY DOCUMENTS INTO SENTENCES
# --------------------------------------------------

def split_sentences(policies):

    chunks = []

    for filename, content in policies.items():

        sentences = re.split(r"(?<=[.!?])\s+|\n+", content)

        for sentence in sentences:

            sentence = sentence.strip()

            if sentence:

                chunks.append({
                    "source": filename,
                    "text": sentence
                })

    return chunks


# --------------------------------------------------
# 4. RETRIEVE RELEVANT POLICY SENTENCES
# --------------------------------------------------

def tokenize(text):

    return set(
        re.findall(r"\b[a-z0-9]+\b", text.lower())
    )


def retrieve(question, chunks, top_k=3):

    question_words = tokenize(question)

    scored_chunks = []

    for chunk in chunks:

        sentence_words = tokenize(chunk["text"])

        overlap = len(
            question_words.intersection(sentence_words)
        )

        score = overlap

        question_lower = question.lower()
        sentence_lower = chunk["text"].lower()

        # Keyword relevance boosts

        if "refund" in question_lower and "refund" in sentence_lower:
            score += 3

        if "delivery" in question_lower and "delivery" in sentence_lower:
            score += 2

        if "payment" in question_lower and "payment" in sentence_lower:
            score += 2

        if "fraud" in question_lower and "fraud" in sentence_lower:
            score += 3

        if "privacy" in question_lower and any(
            word in sentence_lower
            for word in ["personal", "address", "phone", "email"]
        ):
            score += 2

        if score > 0:

            scored_chunks.append((score, chunk))

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        chunk
        for score, chunk in scored_chunks[:top_k]
    ]


# --------------------------------------------------
# 5. POLICY ASSISTANT
# --------------------------------------------------

def ask_policy_assistant(question, chunks):

    retrieved = retrieve(question, chunks, top_k=3)

    if not retrieved:

        return (
            "I could not find that information in the policy documents.",
            ""
        )

    context = "\n".join(
        f"[{item['source']}] {item['text']}"
        for item in retrieved
    )

    messages = [

        {
            "role": "system",
            "content": (
                "You are a POLICY ASSISTANT. "
                "Answer using only the supplied policy context. "
                "Do not invent information."
            )
        },

        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n"
                f"QUESTION:\n{question}"
            )
        }

    ]

    try:

        response = mock_llm.chat(messages)

        answer = response.choices[0].message.content

        if not answer:

            answer = (
                "I could not find that information in the policy documents."
            )

    except Exception as error:

        print("Policy assistant error:", error)

        answer = (
            "I could not find that information in the policy documents."
        )

    source = retrieved[0]["source"]

    return answer, source


# --------------------------------------------------
# 6. SAMPLE QUESTIONS
# --------------------------------------------------

def run_sample_questions(chunks):

    questions = [

        "What is refund policy?",

        "When can a customer receive a refund?",

        "What happens if delivery is late?",

        "How are payment issues handled?",

        "How is customer personal information protected?"

    ]

    print("\nPOLICY ASSISTANT SAMPLE QUESTIONS")

    for question in questions:

        answer, source = ask_policy_assistant(
            question,
            chunks
        )

        print("\nQuestion:", question)
        print("Answer:", answer)
        print("Source:", source)


# --------------------------------------------------
# 7. POLICY EVALUATION
# --------------------------------------------------

def evaluate_policy(chunks):

    if not EVALUATION_FILE.exists():

        print("\nEvaluation file not found:", EVALUATION_FILE)
        return

    evaluation_df = pd.read_csv(EVALUATION_FILE)

    results = []

    for _, row in evaluation_df.iterrows():

        question = str(row["question"])

        # Read expected document

        expected_doc = (
            ""
            if pd.isna(row["expected_doc"])
            else str(row["expected_doc"]).strip()
        )

        # Treat NONE, NAN and blank as unsupported

        if expected_doc.upper() in ["NONE", "NAN", ""]:
            expected_doc = ""

        # Read expected fact

        expected_fact = (
            ""
            if pd.isna(row["expected_fact"])
            else str(row["expected_fact"]).strip()
        )

        if expected_fact.upper() in ["NONE", "NAN", ""]:
            expected_fact = ""

        # Ask the policy assistant

        answer, predicted_source = ask_policy_assistant(
            question,
            chunks
        )

        # Evaluate answerable questions

        if expected_doc:

            source_correct = (
                predicted_source == expected_doc
            )

            fact_correct = (
                expected_fact.lower() in answer.lower()
                if expected_fact
                else False
            )

        # Evaluate unsupported questions

        else:

            source_correct = False

            fact_correct = (
                "could not find" in answer.lower()
                or "not available" in answer.lower()
            )

        results.append({

            "question": question,

            "expected_doc": expected_doc,

            "predicted_source": predicted_source,

            "expected_fact": expected_fact,

            "answer": answer,

            "source_correct": source_correct,

            "fact_correct": fact_correct

        })

    results_df = pd.DataFrame(results)

    # Save evaluation results

    results_df.to_csv(
        RESULT_FILE,
        index=False
    )

    # Separate supported and unsupported questions

    answerable = (
        results_df["expected_doc"].fillna("").str.strip() != ""
    )

    answerable_count = int(answerable.sum())

    unanswerable_count = int((~answerable).sum())

    # Calculate source and fact accuracy

    if answerable_count > 0:

        source_accuracy = (
            results_df.loc[answerable, "source_correct"].mean() * 100
        )

        fact_accuracy = (
            results_df.loc[answerable, "fact_correct"].mean() * 100
        )

    else:

        source_accuracy = 0
        fact_accuracy = 0

    # Calculate unsupported-question abstention accuracy

    if unanswerable_count > 0:

        abstention_accuracy = (
            results_df.loc[~answerable, "fact_correct"].mean() * 100
        )

    else:

        abstention_accuracy = 0

    # Display evaluation summary

    print("\nPOLICY EVALUATION RESULTS")
    print("--------------------------------")

    print("Total questions:", len(results_df))

    print("Answerable questions:", answerable_count)

    print("Unanswerable questions:", unanswerable_count)

    print(f"Source accuracy: {source_accuracy:.2f}%")

    print(f"Fact accuracy: {fact_accuracy:.2f}%")

    print(
        f"Unsupported-question abstention accuracy: "
        f"{abstention_accuracy:.2f}%"
    )

    # Display detailed results

    print("\nDETAILED RESULTS")

    print(
        results_df[
            [
                "question",
                "predicted_source",
                "source_correct",
                "fact_correct"
            ]
        ].to_string(index=False)
    )

    print("\nEvaluation results saved to:")

    print(RESULT_FILE)


# --------------------------------------------------
# 8. MAIN PROGRAM
# --------------------------------------------------

def main():

    print("CUSTOMER SUPPORT POLICY RAG SYSTEM")
    print("----------------------------------")

    mock_llm.reset()

    policies = load_policies()

    print("Policy documents loaded:", len(policies))

    chunks = split_sentences(policies)

    print("Policy sentences created:", len(chunks))

    run_sample_questions(chunks)

    evaluate_policy(chunks)

    print("\nPOLICY RAG COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":

    main()

