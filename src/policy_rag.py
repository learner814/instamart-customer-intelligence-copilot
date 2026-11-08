
"""Customer Support Policy RAG using TF-IDF and an offline mock LLM."""

import csv
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mock_llm import chat, reset


POLICY_DIR = ROOT / "data" / "policies"
EVAL_FILE = ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = ROOT / "outputs"

CHUNK_SIZE = 40
CHUNK_OVERLAP = 10
TOP_K = 3

FALLBACK = (
    "I could not find that information in the policy documents."
)


def load_policies():
    """Load policy documents."""
    policies = {}

    for path in sorted(POLICY_DIR.glob("*.txt")):
        policies[path.name] = path.read_text(encoding="utf-8")

    return policies


def split_into_chunks(policies, chunk_size=CHUNK_SIZE,
                      overlap=CHUNK_OVERLAP):
    """Split documents into overlapping word chunks."""
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk overlap.")

    chunks = []
    step = chunk_size - overlap

    for source, text in policies.items():
        words = text.split()

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
    """Build the TF-IDF index."""
    if not chunks:
        raise ValueError("No policy chunks were created.")

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(
        [chunk["text"] for chunk in chunks]
    )

    return vectorizer, matrix


def retrieve(question, chunks, vectorizer, matrix, top_k=TOP_K):
    """Retrieve the top matching policy chunks."""
    question_vector = vectorizer.transform([question])

    scores = cosine_similarity(
        question_vector, matrix
    ).flatten()

    ranked_indices = scores.argsort()[::-1][:top_k]

    return [
        {
            "source": chunks[index]["source"],
            "score": float(scores[index]),
            "text": chunks[index]["text"]
        }
        for index in ranked_indices
    ]


def answer_question(question, chunks, vectorizer, matrix):
    """Generate an answer from retrieved policy context."""
    retrieved = retrieve(
        question, chunks, vectorizer, matrix
    )

    if not retrieved or retrieved[0]["score"] <= 0:
        return FALLBACK, [], retrieved

    # Do not send zero-score chunks to the mock LLM.
    retrieved = [
        item for item in retrieved if item["score"] > 0
    ]

    context = "\n".join(
        f"[{item['source']}]\n{item['text']}"
        for item in retrieved
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a POLICY ASSISTANT. Answer only using "
                "the supplied policy context. Do not invent facts. "
                "If the answer is not present in the context, "
                "reply exactly: " + FALLBACK
            )
        },
        {
            "role": "user",
            "content": (
                "CONTEXT:\n"
                + context
                + "\nQUESTION:\n"
                + question
            )
        }
    ]

    response = chat(messages)
    answer = response.choices[0].message.content

    sources = list(dict.fromkeys(
        item["source"] for item in retrieved
    ))

    return answer, sources, retrieved


def evaluate(chunks, vectorizer, matrix, overlap, output_name):
    """Evaluate retrieval and answer quality."""
    if not EVAL_FILE.exists():
        raise FileNotFoundError(EVAL_FILE)

    with EVAL_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)
        questions = list(reader)
        columns = reader.fieldnames or []

    required = {"question", "expected_doc", "expected_fact"}

    if not required.issubset(columns):
        raise ValueError(
            "CSV must contain question, expected_doc "
            "and expected_fact."
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

        # Treat NONE as an unsupported question.
        if expected_source.upper() in ("NONE", "N/A", ""):
            expected_source = ""

        if expected_fact.upper() in ("NONE", "N/A", ""):
            expected_fact = ""

        answer, sources, retrieved = answer_question(
            question, chunks, vectorizer, matrix
        )

        retrieved_sources = ";".join(sources)

        if expected_source:
            answerable_total += 1

            source_correct = expected_source in sources

            if source_correct:
                source_correct_count += 1

            fact_correct = (
                expected_fact.casefold() in answer.casefold()
            )

            if fact_correct:
                fact_correct_count += 1

        else:
            unanswerable_total += 1
            source_correct = not sources

            fact_correct = answer.strip().startswith(
                "I could not find"
            )

            if fact_correct:
                abstention_correct_count += 1

        results.append({
            "question": question,
            "expected_doc": expected_source or "NONE",
            "retrieved_sources": retrieved_sources,
            "expected_fact": expected_fact,
            "answer": answer,
            "source_correct": source_correct,
            "fact_correct": fact_correct
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / output_name

    fieldnames = [
        "question",
        "expected_doc",
        "retrieved_sources",
        "expected_fact",
        "answer",
        "source_correct",
        "fact_correct"
    ]

    with output_file.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nPOLICY EVALUATION RESULTS")
    print("--------------------------------")
    print("Chunk overlap:", overlap)
    print("Total questions:", len(questions))
    print("Answerable questions:", answerable_total)
    print("Unsupported questions:", unanswerable_total)

    if answerable_total:
        print(
            "Source accuracy:",
            f"{source_correct_count / answerable_total * 100:.2f}%"
        )
        print(
            "Fact accuracy:",
            f"{fact_correct_count / answerable_total * 100:.2f}%"
        )

    if unanswerable_total:
        print(
            "Abstention accuracy:",
            f"{abstention_correct_count / unanswerable_total * 100:.2f}%"
        )

    print("\nDETAILED ANSWERS")

    for result in results:
        print("\nQuestion:", result["question"])
        print("Answer:", result["answer"])
        print("Expected document:", result["expected_doc"])
        print("Retrieved documents:", result["retrieved_sources"])
        print("Fact correct:", result["fact_correct"])

    print("\nResults saved to:", output_file)

    return {
        "source_correct": source_correct_count,
        "fact_correct": fact_correct_count,
        "answerable_total": answerable_total,
        "abstention_correct": abstention_correct_count,
        "unanswerable_total": unanswerable_total
    }


def main():
    """Run both required overlap evaluations."""
    print("CUSTOMER SUPPORT POLICY RAG SYSTEM")
    print("----------------------------------")

    policies = load_policies()
    print("Policy documents loaded:", len(policies))

    # Evaluation with 10-word overlap.
    chunks = split_into_chunks(
        policies, CHUNK_SIZE, CHUNK_OVERLAP
    )

    print("Policy chunks created:", len(chunks))
    print("Chunk size:", CHUNK_SIZE)
    print("Chunk overlap:", CHUNK_OVERLAP)

    vectorizer, matrix = build_index(chunks)
    print("TF-IDF index created successfully.")

    reset()
    evaluate(
        chunks,
        vectorizer,
        matrix,
        CHUNK_OVERLAP,
        "policy_eval_results.csv"
    )

    # Evaluation with zero overlap.
    zero_chunks = split_into_chunks(
        policies, CHUNK_SIZE, 0
    )

    zero_vectorizer, zero_matrix = build_index(zero_chunks)

    reset()
    evaluate(
        zero_chunks,
        zero_vectorizer,
        zero_matrix,
        0,
        "policy_eval_zero_overlap.csv"
    )

    print("\nPOLICY RAG COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
