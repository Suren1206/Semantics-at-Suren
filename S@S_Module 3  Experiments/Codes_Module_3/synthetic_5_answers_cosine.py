# ======================================================================
# Semantics @ Suren — Question vs Five Answers
# ======================================================================
#
# Objective:
# Calculate cosine similarity between the question and each of
# the five candidate answers.
#
# Embedding model:
# MXBAI
#
# No vector store is created or modified.
# Embeddings exist only temporarily in memory.
# ======================================================================

import numpy as np
from langchain_ollama import OllamaEmbeddings


# ======================================================================
# MODEL
# ======================================================================

MODEL = "mxbai-embed-large:335m"

embeddings_model = OllamaEmbeddings(
    model=MODEL
)


# ======================================================================
# QUESTION
# ======================================================================

QUESTION = (
    "What is the easiest way to study geometric representations "
    "of semantic expressions?"
)


# ======================================================================
# FIVE ANSWERS
# ======================================================================

ANSWERS = {

    "A1": (
        "Visualise embeddings as points or vectors and inspect their "
        "distances and directions."
    ),

    "A2": (
        "Study mathematical properties such as cosine similarity and "
        "Euclidean distance."
    ),

    "A3": (
        "Use controlled experiments that perturb text and observe how "
        "its embedding changes."
    ),

    "A4": (
        "A good way to study geometric representations is to plot "
        "embedding vectors in a reduced-dimensional space and observe "
        "how related expressions cluster or separate."
    ),

    "A5": (
        "Start with simple examples, generate their embeddings, and "
        "compare how changes in meaning alter their position and "
        "relationships in the embedding space."
    )
}


# ======================================================================
# COSINE SIMILARITY
# ======================================================================

def cosine_similarity(a, b):

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# ======================================================================
# EMBED QUESTION
# ======================================================================

print("=" * 100)
print("Semantics @ Suren — Question vs Five Answers")
print("=" * 100)

print()
print("Embedding model :", MODEL)
print()
print("Question:")
print(QUESTION)

print()
print("Generating question embedding...")

query_vector = np.array(
    embeddings_model.embed_query(QUESTION),
    dtype=np.float32
)


# ======================================================================
# EMBED ANSWERS AND CALCULATE COSINE
# ======================================================================

results = []

for answer_id, answer_text in ANSWERS.items():

    print(f"Embedding {answer_id}...")

    answer_vector = np.array(
        embeddings_model.embed_query(answer_text),
        dtype=np.float32
    )

    cosine = cosine_similarity(
        query_vector,
        answer_vector
    )

    results.append(
        (
            answer_id,
            answer_text,
            cosine
        )
    )


# ======================================================================
# RANK BY COSINE — HIGHEST TO LOWEST
# ======================================================================

results.sort(
    key=lambda x: x[2],
    reverse=True
)


# ======================================================================
# PRINT RESULTS
# ======================================================================

print()
print("=" * 100)
print("RANKED ANSWERS — HIGHEST TO LOWEST COSINE")
print("=" * 100)

for rank, (answer_id, answer_text, cosine) in enumerate(
    results,
    start=1
):

    print()
    print(f"RANK {rank}")
    print("-" * 100)

    print(f"Answer ID : {answer_id}")
    print(f"Cosine    : {cosine:.6f}")
    print(f"Answer    : {answer_text}")


# ======================================================================
# COMPACT SUMMARY
# ======================================================================

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

for rank, (answer_id, answer_text, cosine) in enumerate(
    results,
    start=1
):

    print(
        f"{rank}. {answer_id} | Cosine: {cosine:.6f}"
    )


# ======================================================================
# END
# ======================================================================

print()
print("=" * 100)
print("Experiment complete.")
print("No vector store was created or modified.")
print("=" * 100)