from pathlib import Path
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Find the project root and make it available for imports
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import mock_llm

# Find the project and policy folders
POLICY_DIR = PROJECT_DIR / "data" / "policies"

CHUNK_SIZE = 40
CHUNK_OVERLAP = 10


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping word-based chunks."""

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # Stop when the end of the document is reached
        if end == len(words):
            break

        # Move forward while keeping the required overlap
        start = end - overlap

    return chunks


def load_policy_chunks(overlap=CHUNK_OVERLAP):
    """Load all policy files and create chunks."""

    all_chunks = []

    for file_path in sorted(POLICY_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")

        file_chunks = chunk_text(text, overlap=overlap)

        for chunk in file_chunks:
            all_chunks.append({
                "source": file_path.name,
                "text": chunk
            })

        print(f"{file_path.name}: {len(file_chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    return all_chunks


def build_index(chunks):
    """Convert policy chunks into TF-IDF vectors."""

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(
        [chunk["text"] for chunk in chunks]
    )

    return vectorizer, tfidf_matrix


def retrieve(query, chunks, vectorizer, tfidf_matrix, k=3):
    """Return the k most relevant policy chunks."""

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    ranked_indices = similarities.argsort()[::-1][:k]

    results = []

    for index in ranked_indices:
        results.append({
            "source": chunks[index]["source"],
            "score": float(similarities[index]),
            "text": chunks[index]["text"]
        })

    return results


def answer(question, chunks, vectorizer, tfidf_matrix):
    """Answer a policy question using retrieved policy context."""

    results = retrieve(
        question,
        chunks,
        vectorizer,
        tfidf_matrix,
        k=3
    )

    context_parts = []

    for result in results:
        context_parts.append(
            f"[{result['source']}]\n{result['text']}"
        )

    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT. Answer using only the "
                "provided policy context. If the answer is "
                "not available in the context, say: "
                "I could not find that information in the policy documents."
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

    response = mock_llm.chat(messages)

    return {
        "answer": response.choices[0].message.content,
        "sources": [result["source"] for result in results]
    }


def evaluate_policy_rag(chunks, vectorizer, tfidf_matrix, eval_df):
    """Evaluate document retrieval, fact matching and fallback."""

    document_correct = 0
    fact_correct = 0
    fallback_correct = 0
    answerable_count = 0
    unanswerable_count = 0

    for _, row in eval_df.iterrows():
        question = row["question"]
        expected_doc = str(row["expected_doc"]).strip()
        expected_fact = str(row["expected_fact"]).strip()

        result = answer(
            question,
            chunks,
            vectorizer,
            tfidf_matrix
        )

        retrieved_sources = result["sources"]
        generated_answer = result["answer"]

        if expected_doc.upper() == "NONE":
            unanswerable_count += 1

            if generated_answer.startswith("I could not find"):
                fallback_correct += 1

        else:
            answerable_count += 1

            if expected_doc in retrieved_sources:
                document_correct += 1

            if expected_fact in generated_answer:
                fact_correct += 1

    print("\nEvaluation results")
    print("------------------")
    print(
        f"Document retrieval: {document_correct}/{answerable_count}"
    )
    print(
        f"Expected fact match: {fact_correct}/{answerable_count}"
    )
    print(
        f"Unanswerable fallback: {fallback_correct}/{unanswerable_count}"
    )


def evaluate_zero_overlap(eval_df):
    """Evaluate RAG again with zero chunk overlap."""

    print("\n\nZERO-OVERLAP EVALUATION")
    print("=======================")

    chunks = load_policy_chunks(overlap=0)

    vectorizer, tfidf_matrix = build_index(chunks)

    failed_questions = []

    for _, row in eval_df.iterrows():
        expected_doc = str(row["expected_doc"]).strip()
        expected_fact = str(row["expected_fact"]).strip()

        if expected_doc.upper() == "NONE":
            continue

        result = answer(
            row["question"],
            chunks,
            vectorizer,
            tfidf_matrix
        )

        doc_found = expected_doc in result["sources"]
        fact_found = expected_fact in result["answer"]

        if not doc_found or not fact_found:
            failed_questions.append({
                "question": row["question"],
                "document_found": doc_found,
                "fact_found": fact_found
            })

    print(
        f"\nZero-overlap failed questions: "
        f"{len(failed_questions)}"
    )

    for failure in failed_questions:
        print("\nQuestion:", failure["question"])
        print("Expected document found:", failure["document_found"])
        print("Expected fact found:", failure["fact_found"])

    return failed_questions

if __name__ == "__main__":
    eval_path = PROJECT_DIR / "data" / "raw" / "policy_eval.csv"
    eval_df = pd.read_csv(eval_path).fillna("")

    print("EVALUATION WITH 10-WORD OVERLAP")
    print("================================")

    chunks = load_policy_chunks()

    vectorizer, tfidf_matrix = build_index(chunks)

    evaluate_policy_rag(
        chunks,
        vectorizer,
        tfidf_matrix,
        eval_df
    )

    evaluate_zero_overlap(eval_df)