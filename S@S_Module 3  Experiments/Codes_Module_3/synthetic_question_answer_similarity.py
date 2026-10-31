"""
===============================================================================
Semantics @ Suren — Question vs Seven Answers
===============================================================================

Objective
---------
Use Question A as the query and compare it against seven candidate answers.

Question A:
"What are the chief advantages of AI Vision?"

Calculate:
    Query <-> Answer cosine similarity

Then rank all seven answers from highest to lowest cosine.

No vector store is created or modified.
Embeddings exist only temporarily in memory.
===============================================================================
"""

import numpy as np
from langchain_ollama import OllamaEmbeddings


# =============================================================================
# QUESTION
# =============================================================================

QUESTION = (
    "What is the usefulness of learning machine learning?")



# =============================================================================
# SEVEN ANSWERS
# =============================================================================

ANSWERS = {

    "A1": (
        "Visualise embeddings as points or vectors and inspect their "
        "distances and directions."
    ),

    "A2": (
        "Study mathematical properties such as cosine similarity "
        "and Euclidean distance."
    ),

    "A3": (
        "Use controlled experiments that perturb text and observe how "
        "its embedding changes."
    ),

    "B1": (
        "IAI Vision can analyse images and videos to identify objects, patterns, and visual features."        
    ),

    "B2": (
        "AI Vision can automate visual inspection, enabling systems to detect defects or anomalies in manufactured products. "        
    ),

    "C1": (
        "Machine learning can be used for medical diagnosis "
        "and prediction."
    ),

    "C2": (
        "Machine learning can be used for financial fraud detection."
    )
}


# =============================================================================
# EMBEDDING MODEL
# =============================================================================

MODEL_NAME = "mxbai-embed-large:335m"

embeddings_model = OllamaEmbeddings(
    model=MODEL_NAME
)


# =============================================================================
# COSINE FUNCTION
# =============================================================================

def cosine_similarity(a, b):

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# =============================================================================
# GENERATE QUERY EMBEDDING
# =============================================================================

print("=" * 100)
print("Semantics @ Suren — Question vs Seven Answers")
print("=" * 100)

print()
print("Embedding model :", MODEL_NAME)

print()
print("QUESTION")
print("-" * 100)
print(QUESTION)


query_vector = embeddings_model.embed_query(
    QUESTION
)


# =============================================================================
# GENERATE ANSWER EMBEDDINGS + COSINE SCORES
# =============================================================================

results = []

for answer_id, answer_text in ANSWERS.items():

    answer_vector = embeddings_model.embed_query(
        answer_text
    )

    cosine = cosine_similarity(
        query_vector,
        answer_vector
    )

    results.append(
        {
            "id": answer_id,
            "text": answer_text,
            "cosine": cosine
        }
    )


# =============================================================================
# RANK
# =============================================================================

results.sort(
    key=lambda x: x["cosine"],
    reverse=True
)


# =============================================================================
# PRINT RANKED RESULTS
# =============================================================================

print()
print("=" * 100)
print("RANKED ANSWERS — HIGHEST TO LOWEST COSINE")
print("=" * 100)

for rank, result in enumerate(results, start=1):

    print()
    print(f"RANK {rank}")
    print("-" * 100)

    print(
        f"Answer ID : {result['id']}"
    )

    print(
        f"Cosine    : {result['cosine']:.6f}"
    )

    print(
        f"Answer    : {result['text']}"
    )


# =============================================================================
# SUMMARY
# =============================================================================

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

for rank, result in enumerate(results, start=1):

    print(
        f"{rank}. {result['id']} | "
        f"Cosine: {result['cosine']:.6f}"
    )

print()
print("Experiment complete.")
print("No vector store was created or modified.")
print("=" * 100)