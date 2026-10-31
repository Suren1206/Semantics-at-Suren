# ======================================================================
# Semantics @ Suren — Pairwise Cosine Similarity Matrix
# ======================================================================

import numpy as np
from langchain_ollama import OllamaEmbeddings


# ======================================================================
# EMBEDDING MODEL
# ======================================================================

MODEL = "mxbai-embed-large:335m"

embeddings_model = OllamaEmbeddings(
    model=MODEL
)


# ======================================================================
# SEVEN ANSWERS
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

    "B1": (
        "AI Vision can analyse images and videos to identify objects, patterns, and visual features."
        
    ),

    "B2": (
        "AI Vision can automate visual inspection, enabling systems to detect defects or anomalies in manufactured products. "        
    ),

    "C1": (
        "Machine learning can be used for medical diagnosis and prediction."
    ),

    "C2": (
        "Machine learning can be used for financial fraud detection."
    ),
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
print("Semantics @ Suren — Pairwise Cosine Similarity Matrix")
print("=" * 100)

print()
print("Embedding model :", MODEL)
print("Generating embeddings...")
print()

vectors = {}

for answer_id, answer_text in ANSWERS.items():

    print(f"Embedding {answer_id}...")

    vectors[answer_id] = np.array(
        embeddings_model.embed_query(answer_text),
        dtype=np.float32
    )


# ======================================================================
# PAIRWISE COSINE MATRIX
# ======================================================================

ids = list(ANSWERS.keys())

matrix = np.zeros(
    (len(ids), len(ids)),
    dtype=np.float32
)

for i, id1 in enumerate(ids):

    for j, id2 in enumerate(ids):

        matrix[i, j] = cosine_similarity(
            vectors[id1],
            vectors[id2]
        )


# ======================================================================
# PRINT MATRIX
# ======================================================================

print()
print("=" * 100)
print("PAIRWISE COSINE SIMILARITY MATRIX")
print("=" * 100)

print()

print(f"{'':8}", end="")

for answer_id in ids:
    print(f"{answer_id:>10}", end="")

print()

print("-" * 78)

for i, answer_id in enumerate(ids):

    print(f"{answer_id:<8}", end="")

    for j in range(len(ids)):

        print(f"{matrix[i, j]:10.6f}", end="")

    print()


# ======================================================================
# PAIRWISE RANKING
# ======================================================================

print()
print("=" * 100)
print("PAIRWISE COMPARISON — HIGHEST TO LOWEST")
print("=" * 100)

pairs = []

for i in range(len(ids)):

    for j in range(i + 1, len(ids)):

        pairs.append(
            (
                ids[i],
                ids[j],
                matrix[i, j]
            )
        )

pairs.sort(
    key=lambda x: x[2],
    reverse=True
)

print()

for rank, (id1, id2, score) in enumerate(pairs, start=1):

    print(
        f"{rank:2}. "
        f"{id1} <-> {id2} | "
        f"Cosine: {score:.6f}"
    )


# ======================================================================
# SAVE MATRIX
# ======================================================================

output_file = "pairwise_cosine_matrix_mxbai_modified.npy"

np.save(
    output_file,
    matrix
)


# ======================================================================
# SUMMARY
# ======================================================================

print()
print("=" * 100)
print("COMPLETE")
print("=" * 100)

print()
print("Model        :", MODEL)
print("Vectors      :", len(ids))
print("Matrix shape :", matrix.shape)
print("Output file  :", output_file)
print()
print("No existing vector store was created or modified.")
print("=" * 100)