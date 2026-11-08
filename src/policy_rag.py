
"""Customer Support Policy RAG using TF-IDF and an offline mock LLM."""

import csv
import re
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mock_llm import chat, reset


POLICY_DIR = ROOT / "data" / "policies"
EVAL_FILE = ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_FILE = ROOT / "outputs" / "policy_eval_results.csv"

CHUNK_SIZE = 40
CHUNK_OVERLAP = 10
TOP_K = 3


def load_policies():
    """Load policy documents from the policies folder."""
    policies = {}

    for path in sorted(POLICY_DIR.glob("*.txt")):
        policies[path.name] = path.read_text(encoding="utf-8")

    return policies


def split_into_chunks(policies, chunk_size=CHUNK_SIZE,
                      overlap=CHUNK_OVERLAP):
    """Split policy documents into overlapping word chunks."""

    chunks = []

    for source, text in policies.items():
        words = text.split()

        if not words:
            continue

        step = max(1, chunk_size - overlap)

        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]

            if not chunk_words:
                break

            chunks.append({
                "source": source,
                "text": " ".join(chunk_words)
            })

            if start + chunk_size >= len(words):
                break

    return chunks


def build_index(chunks):
    """Build the TF-IDF search index."""

    if not chunks:
        raise ValueError("No policy chunks were created.")

    vectorizer = TfidfVectorizer(stop_words="english")

    matrix = vectorizer.fit_transform(
        [chunk["text"] for chunk in chunks]
    )

    return vectorizer, matrix


def retrieve(question, chunks, vectorizer, matrix, top_k=TOP_K):
    """Find the most relevant policy chunks."""

    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(
        question_vector, matrix
    ).flatten()

    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices:
        if similarities[index] > 0:
            results.append(chunks[index])

        if len(results) >= top_k:
            break

    return results


def answer_question(question, chunks, vectorizer, matrix):
    """Generate an answer using retrieved policy context."""

    retrieved = retrieve(
        question, chunks, vectorizer, matrix
    )

    if not retrieved:
        return (
            "I could not find that information in the policy documents.",
            ""
        )

    context = "\n".join(
        f"{chunk['text']} [{chunk['source']}]"
        for chunk in retrieved
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a POLICY ASSISTANT. Answer questions using "
                "only the supplied policy context. Do not invent facts."
            )
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION: {question}"
            )
        }
    ]

    response = chat(messages)
    answer = response.choices[0].message.content

    source = retrieved[0]["source"]

    return answer, source


def evaluate(chunks, vectorizer, matrix):
    """Evaluate the RAG system using policy_eval.csv."""

    if not EVAL_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {EVAL_FILE}"
        )

    with EVAL_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        questions = list(reader)

    required_columns = {
        "question", "expected_doc", "expected_fact"
    }

    if not required_columns.issubset(reader.fieldnames or []):
        raise ValueError(
            "CSV must contain: question, expected_doc, expected_fact"
        )

    results = []

    answerable_total = 0
    source_correct_count = 0
    fact_correct_count = 0

    unanswerable_total = 0
    abstention_correct_count = 0

    for item in questions:
        question = item["question"].strip()
        expected_source = item["expected_doc"].strip()
        expected_fact = item["expected_fact"].strip()

        answer, predicted_source = answer_question(
            question, chunks, vectorizer, matrix
        )

        source_correct = (
            predicted_source == expected_source
            if expected_source else not predicted_source
        )

        if expected_fact:
            answerable_total += 1

            if source_correct:
                source_correct_count += 1

            fact_correct = (
                expected_fact.casefold() in answer.casefold()
            )

            if fact_correct:
                fact_correct_count += 1

        else:
            unanswerable_total += 1

            fact_correct = (
                "could not find" in answer.casefold()
                or "not found" in answer.casefold()
                or "unable" in answer.casefold()
            )

            if fact_correct:
                abstention_correct_count += 1

        results.append({
            "question": question,
            "expected_doc": expected_source,
            "predicted_source": predicted_source,
            "expected_fact": expected_fact,
            "answer": answer,
            "source_correct": source_correct,
            "fact_correct": fact_correct
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        fieldnames = [
            "question",
            "expected_doc",
            "predicted_source",
            "expected_fact",
            "answer",
            "source_correct",
            "fact_correct"
        ]

        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nPOLICY EVALUATION RESULTS")
    print("--------------------------------")
    print(f"Total questions: {len(questions)}")
    print(f"Answerable questions: {answerable_total}")
    print(f"Unanswerable questions: {unanswerable_total}")

    if answerable_total:
        print(
            "Source accuracy: "
            f"{source_correct_count / answerable_total * 100:.2f}%"
        )

        print(
            "Fact accuracy: "
            f"{fact_correct_count / answerable_total * 100:.2f}%"
        )

    if unanswerable_total:
        print(
            "Unsupported-question abstention accuracy: "
            f"{abstention_correct_count / unanswerable_total * 100:.2f}%"
        )

    print("\nDETAILED RESULTS")

    for result in results:
        print("\nQuestion:", result["question"])
        print("Expected fact:", result["expected_fact"])
        print("Generated answer:", result["answer"])
        print("Predicted source:", result["predicted_source"])
        print("Source correct:", result["source_correct"])
        print("Fact correct:", result["fact_correct"])

    print("\nEvaluation results saved to:")
    print(OUTPUT_FILE)


def main():
    """Run the complete policy RAG system."""

    print("CUSTOMER SUPPORT POLICY RAG SYSTEM")
    print("----------------------------------")

    policies = load_policies()

    print(f"Policy documents loaded: {len(policies)}")

    chunks = split_into_chunks(policies)

    print(f"Policy chunks created: {len(chunks)}")
    print(f"Chunk size: {CHUNK_SIZE} words")
    print(f"Chunk overlap: {CHUNK_OVERLAP} words")

    vectorizer, matrix = build_index(chunks)

    print("TF-IDF index created successfully.")

    print("\nPOLICY ASSISTANT SAMPLE QUESTIONS")

    sample_questions = [
        "What is refund policy?",
        "When can a customer receive a refund?",
        "What happens if delivery is late?",
        "How are payment issues handled?",
        "How soon is money returned when a payment fails but is deducted?",
        "How is customer personal information protected?"
    ]

    for question in sample_questions:
        answer, source = answer_question(
            question, chunks, vectorizer, matrix
        )

        print("\nQuestion:", question)
        print("Answer:", answer)
        print("Source:", source)

    reset()

    evaluate(chunks, vectorizer, matrix)

    print("\nPOLICY RAG COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
