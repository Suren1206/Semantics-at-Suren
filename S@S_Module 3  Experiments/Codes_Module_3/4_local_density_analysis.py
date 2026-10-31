"""
===============================================================================
Semantics @ Suren — Local Density of Retrieved Top-5 Chunks
===============================================================================

Calculates Local Density for the Top-5 retrieved chunks of:

    Q7
    Q140

Local Density of a chunk =
    Average cosine similarity between that chunk and the other 4
    chunks in the same Top-5 retrieval set.

Existing vector stores are NOT modified.

Input:
    Q7_top5_embeddings.npy
    Q140_top5_embeddings.npy

Each file should contain:
    5 x 1024

Output:
    Local density for every retrieved chunk, Q7 followed by Q140.
===============================================================================
"""

import numpy as np
from pathlib import Path


# =============================================================================
# PROJECT DIRECTORY
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent


# =============================================================================
# INPUT FILES
# =============================================================================


FILES = {
    "Q7": BASE_DIR / "Vector_Files" / "2_Q7_top5_embeddings.npy",
    "Q140": BASE_DIR / "Vector_Files" / "2_Q140_top5_embeddings.npy"
}

# =============================================================================
# LOCAL DENSITY CALCULATION
# =============================================================================

def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return np.dot(a, b) / denominator


def calculate_local_density(embeddings):

    n = len(embeddings)

    densities = []

    for i in range(n):

        similarities = []

        for j in range(n):

            if i == j:
                continue

            similarity = cosine_similarity(
                embeddings[i],
                embeddings[j]
            )

            similarities.append(similarity)

        density = np.mean(similarities)

        densities.append(density)

    return np.array(densities)


# =============================================================================
# RUN EXPERIMENT
# =============================================================================

print("=" * 100)
print("Semantics @ Suren — Local Density Analysis")
print("=" * 100)

for query_id, file_path in FILES.items():

    print()
    print("=" * 100)
    print(f"{query_id} — TOP 5 RETRIEVED CHUNKS")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Load embeddings
    # -------------------------------------------------------------------------

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nEmbedding file not found:\n{file_path}\n"
        )

    embeddings = np.load(
        file_path
    )

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    if embeddings.shape[0] != 5:

        raise ValueError(
            f"{query_id} expected 5 embeddings, "
            f"but found {embeddings.shape[0]}."
        )

    # -------------------------------------------------------------------------
    # Calculate local density
    # -------------------------------------------------------------------------

    densities = calculate_local_density(
        embeddings
    )

    # -------------------------------------------------------------------------
    # Print
    # -------------------------------------------------------------------------

    for i, density in enumerate(densities, start=1):

        print(
            f"Rank {i} | "
            f"Local Density : {density:.6f}"
        )

    # -------------------------------------------------------------------------
    # Print overall density of the retrieval set
    # -------------------------------------------------------------------------

    upper_triangle = []

    for i in range(5):

        for j in range(i + 1, 5):

            similarity = cosine_similarity(
                embeddings[i],
                embeddings[j]
            )

            upper_triangle.append(
                similarity
            )

    overall_density = np.mean(
        upper_triangle
    )

    print()
    print(
        f"Overall Top-5 Local Density : "
        f"{overall_density:.6f}"
    )


print()
print("=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)