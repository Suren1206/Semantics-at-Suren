# =============================================================================
# Semantics @ Suren — Module 3
#
# Q177 — ATOMIC CHUNK RETRIEVAL ANALYSIS
# =============================================================================

from pathlib import Path
import re
import requests
import numpy as np


# =============================================================================
# BLOCK 1 — Q177 QUERY
# =============================================================================

QUESTION_ID = "177"

QUESTION = (
    "After returning from more than two consecutive working days of absence, "
    "what document must employees submit?"
)


# =============================================================================
# BLOCK 2 — THE FIVE RETRIEVED CHUNKS
# =============================================================================

TOP5_CHUNKS = {

    "chunk_2": (
        "Long-term absence exceeding 30 calendar days will require a structured "
        "case review every 30 days thereafter. Employees on medical absence "
        "exceeding 45 calendar days may be required to participate in a "
        "fitness-for-work assessment prior to resuming duties. Return-to-work "
        "discussions must occur within 48 hours of resumption following any "
        "absence exceeding 3 working days."
    ),

    "chunk_0": (
        "Employees must notify their direct manager at least 30 minutes before "
        "their scheduled start time unless physically incapacitated. Absence "
        "exceeding 2 consecutive working days requires submission of a "
        "self-declaration form within 24 hours of return. Any absence extending "
        "beyond 5 consecutive working days must be supported by medical "
        "certification issued no later than the 7th calendar day of absence."
    ),

    "chunk_3": (
        "Employees are expected to maintain reasonable contact during extended "
        "absence, typically not less than once every 14 calendar days unless "
        "otherwise agreed. Adjusted or phased return arrangements may be granted "
        "for a period not normally exceeding 20 working days, unless extended "
        "by management discretion."
    ),

    "chunk_1": (
        "Three separate absence instances within a rolling 90-day period may "
        "trigger a formal attendance review meeting. Where total absence exceeds "
        "8 working days within a rolling 6-month window, management may initiate "
        "an attendance improvement plan. Absence of 10 consecutive working days "
        "without contact may be treated as voluntary abandonment of employment, "
        "subject to documented outreach attempts on at least two separate days."
    ),

    "chunk_10": (
        "Employees are expected to avoid engaging in paid external employment "
        "during annual leave where it may reasonably impact fitness for duty upon "
        "return. More than two cancellations of approved leave within a 12-month "
        "period may require managerial review. Failure to return from leave on "
        "the agreed date without notification exceeding 1 working day may be "
        "treated as unauthorised absence."
    )
}


# =============================================================================
# BLOCK 3 — OLLAMA CONFIGURATION
# =============================================================================

OLLAMA_URL = "http://localhost:11434/api/embed"

EMBEDDING_MODEL = "mxbai-embed-large:335m"


# =============================================================================
# BLOCK 4 — CREATE ATOMIC CHUNKS
# =============================================================================

def create_atomic_chunks(chunk_id, text):

    sentences = re.split(r"\.", text)

    atomic_chunks = []

    for sentence in sentences:

        sentence = sentence.strip()

        if sentence:
            atomic_chunks.append({
                "parent_chunk": chunk_id,
                "atomic_text": sentence + "."
            })

    return atomic_chunks


atomic_chunks = []

for chunk_id, text in TOP5_CHUNKS.items():

    atomic_chunks.extend(
        create_atomic_chunks(
            chunk_id,
            text
        )
    )


# =============================================================================
# BLOCK 5 — HEADER
# =============================================================================

print("=" * 100)
print("Semantics @ Suren — Module 3")
print("Q177 — ATOMIC CHUNK RETRIEVAL ANALYSIS")
print("=" * 100)

print("\nQuestion:")
print(QUESTION)


# =============================================================================
# BLOCK 6 — ATOMIC CHUNKS
# =============================================================================

print("\n")
print("=" * 100)
print("ATOMIC CHUNK GENERATION")
print("=" * 100)

print(
    f"\nOriginal Top-5 chunks : {len(TOP5_CHUNKS)}"
)

print(
    f"Atomic chunks created  : {len(atomic_chunks)}"
)

print("\n")
print("-" * 100)
print("ATOMIC CHUNKS")
print("-" * 100)

for i, item in enumerate(atomic_chunks, start=1):

    print(
        f"\nAtomic A{i:02d} | Parent: {item['parent_chunk']}"
    )

    print(
        f"Text: {item['atomic_text']}"
    )


# =============================================================================
# BLOCK 7 — EMBEDDING FUNCTION
# =============================================================================

def get_embedding(text):

    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return np.array(
        data["embeddings"][0],
        dtype=np.float32
    )


# =============================================================================
# BLOCK 8 — QUERY EMBEDDING
# =============================================================================

print("\n")
print("=" * 100)
print("GENERATING EMBEDDINGS")
print("=" * 100)

print(
    f"\nEmbedding model : {EMBEDDING_MODEL}"
)

print("\nEmbedding query...")

query_vector = get_embedding(QUESTION)

print(
    f"Query dimension : {query_vector.shape[0]}"
)


# =============================================================================
# BLOCK 9 — ATOMIC CHUNK EMBEDDINGS
# =============================================================================

atomic_vectors = []

for i, item in enumerate(atomic_chunks, start=1):

    print(
        f"Embedding atomic chunk A{i:02d}..."
    )

    vector = get_embedding(
        item["atomic_text"]
    )

    atomic_vectors.append(vector)


atomic_vectors = np.vstack(atomic_vectors)


# =============================================================================
# BLOCK 10 — COSINE SIMILARITY
# =============================================================================

def cosine_similarity(query_vector, matrix):

    query_norm = np.linalg.norm(query_vector)

    matrix_norms = np.linalg.norm(
        matrix,
        axis=1
    )

    return (
        matrix @ query_vector
    ) / (
        matrix_norms * query_norm
    )


similarities = cosine_similarity(
    query_vector,
    atomic_vectors
)


# =============================================================================
# BLOCK 11 — RANK ALL ATOMIC CHUNKS
# =============================================================================

ranking = np.argsort(
    similarities
)[::-1]


# =============================================================================
# BLOCK 12 — TOP-10 ATOMIC CHUNKS WITH COSINE SCORES
# =============================================================================

print("\n")
print("=" * 100)
print("Q177 — TOP-10 ATOMIC CHUNKS BY QUERY COSINE SIMILARITY")
print("=" * 100)

print("\n")
print(
    f"{'Rank':<8}"
    f"{'Atomic ID':<12}"
    f"{'Parent Chunk':<16}"
    f"{'Cosine Score':<16}"
    f"Atomic Chunk"
)

print("-" * 100)

top_n = min(
    10,
    len(ranking)
)

for rank in range(top_n):

    idx = ranking[rank]

    parent_chunk = atomic_chunks[idx]["parent_chunk"]

    atomic_text = atomic_chunks[idx]["atomic_text"]

    score = similarities[idx]

    print(
        f"{rank + 1:<8}"
        f"{'A' + str(idx + 1):<12}"
        f"{parent_chunk:<16}"
        f"{score:<16.6f}"
        f"{atomic_text}"
    )


# =============================================================================
# BLOCK 13 — ALL ATOMIC CHUNKS WITH COSINE SCORES
# =============================================================================

print("\n")
print("=" * 100)
print("ALL ATOMIC CHUNKS — COSINE SCORE AGAINST Q177")
print("=" * 100)

print("\n")
print(
    f"{'Atomic ID':<12}"
    f"{'Parent Chunk':<16}"
    f"{'Cosine Score':<16}"
    f"Atomic Chunk"
)

print("-" * 100)

for idx, item in enumerate(atomic_chunks):

    print(
        f"{'A' + str(idx + 1):<12}"
        f"{item['parent_chunk']:<16}"
        f"{similarities[idx]:<16.6f}"
        f"{item['atomic_text']}"
    )


# =============================================================================
# BLOCK 14 — EXPERIMENT COMPLETE
# =============================================================================

print("\n")
print("=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)

print(
    f"\nQuestion ID             : {QUESTION_ID}"
)

print(
    f"Atomic chunks compared : {len(atomic_chunks)}"
)

print(
    f"Top-10 reported        : {top_n}"
)

print(
    f"Embedding model        : {EMBEDDING_MODEL}"
)

print("\nNo mean-centering was applied.")
print("No PCA was applied.")
print("No whitening was applied.")
print("No dimensionality reduction was applied.")
print("Existing B01 embeddings were not modified.")