# ======================================================================
# Semantics @ Suren — Local Density of Five Answers
# ======================================================================

import numpy as np
from langchain_ollama import OllamaEmbeddings


# ======================================================================
# MODEL
# ======================================================================

MODEL = "mxbai-embed-large:335m"

embedding_model = OllamaEmbeddings(
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
# FIVE ANSWERS — EXACTLY AS PROVIDED
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
        "Seetha is the wife of Rama; Ravana got into love with her "
        "and forcefully took her away"
    ),

    "A4": (
        "a"
    ),

    "A5": (
        "Use controlled experiments that perturb text and observe how "
        "its embedding changes."
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
# GENERATE EMBEDDINGS
# ======================================================================

print("=" * 100)
print("Semantics @ Suren — Local Density of Five Answers")
print("=" * 100)

print()
print("Model :", MODEL)
print("Question :", QUESTION)

print()
print("Generating embeddings...")
print()

vectors = {}

for answer_id, answer_text in ANSWERS.items():

    print(f"Embedding {answer_id}...")

    vectors[answer_id] = np.array(
        embedding_model.embed_query(answer_text),
        dtype=np.float32
    )


# ======================================================================
# CALCULATE LOCAL DENSITY
# ======================================================================

answer_ids = list(ANSWERS.keys())

local_densities = {}

for answer_id in answer_ids:

    similarities = []

    for other_id in answer_ids:

        if answer_id == other_id:
            continue

        similarity = cosine_similarity(
            vectors[answer_id],
            vectors[other_id]
        )

        similarities.append(similarity)

    local_densities[answer_id] = np.mean(
        similarities
    )


# ======================================================================
# PRINT ANSWERS + LOCAL DENSITY
# ======================================================================

print()
print("=" * 100)
print("LOCAL DENSITY — EACH ANSWER AGAINST OTHER FOUR ANSWERS")
print("=" * 100)

for answer_id in answer_ids:

    print()
    print(f"Answer {answer_id}")
    print("-" * 100)

    print(
        f"Local Density : "
        f"{local_densities[answer_id]:.6f}"
    )

    print(
        f"Answer       : "
        f"{ANSWERS[answer_id]}"
    )


# ======================================================================
# RANK BY LOCAL DENSITY
# ======================================================================

print()
print("=" * 100)
print("LOCAL DENSITY RANKING — HIGHEST TO LOWEST")
print("=" * 100)

density_ranking = sorted(
    local_densities.items(),
    key=lambda x: x[1],
    reverse=True
)

print()

for rank, (answer_id, density) in enumerate(
    density_ranking,
    start=1
):

    print(
        f"{rank}. {answer_id} | "
        f"Local Density: {density:.6f}"
    )


# ======================================================================
# SAVE EMBEDDINGS
# ======================================================================

embedding_matrix = np.vstack(
    [
        vectors[answer_id]
        for answer_id in answer_ids
    ]
)

np.save(
    "five_answers_mxbai_embeddings.npy",
    embedding_matrix
)


# ======================================================================
# SAVE LOCAL DENSITIES
# ======================================================================

density_values = np.array(
    [
        local_densities[answer_id]
        for answer_id in answer_ids
    ],
    dtype=np.float32
)

np.save(
    "five_answers_local_density.npy",
    density_values
)


# ======================================================================
# FINAL
# ======================================================================

print()
print("=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)

print()
print("Embedding matrix shape :", embedding_matrix.shape)
print("Embedding file         : five_answers_mxbai_embeddings.npy")
print("Density file            : five_answers_local_density.npy")
print()
print("No vector store was created or modified.")
print("=" * 100)