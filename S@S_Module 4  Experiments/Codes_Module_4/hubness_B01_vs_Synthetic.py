import numpy as np
from pathlib import Path


# ==========================================================================================
# Semantics @ Suren — Module 4
# HUBNESS ANALYSIS
# B01 vs SYNTHETIC
#
# Definition used:
# For every chunk:
#     1. Find its top-5 nearest neighbours.
#     2. Exclude the chunk itself.
#     3. Count how frequently each chunk appears in other chunks' top-5 lists.
#
# The chunk with the highest occurrence count is the strongest hub candidate.
# ==========================================================================================


# ==========================================================================================
# PATHS
# ==========================================================================================

BASE_DIR = Path(
    "/home/surendran/LangChain/Semantics@Suren/"
    "S@S_Module 4  Experiments"
)

B01_FILE = (
    BASE_DIR
    / "20260424_B01"
    / "embeddings.npy"
)

SYNTHETIC_FILE = (
    BASE_DIR
    / "Output_Files"
    / "synthetic_raw_embeddings.npy"
)


# ==========================================================================================
# PARAMETERS
# ==========================================================================================

TOP_K = 5


# ==========================================================================================
# HEADER
# ==========================================================================================

print("=" * 100)
print("Semantics @ Suren — Module 4")
print("HUBNESS ANALYSIS — TOP-5 NEAREST NEIGHBOUR OCCURRENCES")
print("=" * 100)


# ==========================================================================================
# LOAD EMBEDDINGS
# ==========================================================================================

def load_embeddings(file_path, name):

    print()
    print(f"Loading {name} embeddings...")
    print(f"File : {file_path}")

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nFile not found:\n{file_path}"
        )

    embeddings = np.load(
        file_path
    ).astype(np.float64)

    if embeddings.ndim != 2:

        raise ValueError(
            f"{name} embeddings must be a 2-dimensional matrix."
        )

    if not np.all(np.isfinite(embeddings)):

        raise ValueError(
            f"{name} embeddings contain NaN or infinite values."
        )

    print(f"Shape : {embeddings.shape}")

    return embeddings


b01 = load_embeddings(
    B01_FILE,
    "B01"
)

synthetic = load_embeddings(
    SYNTHETIC_FILE,
    "Synthetic"
)


# ==========================================================================================
# HUBNESS CALCULATION
# ==========================================================================================

def calculate_hubness(embeddings, top_k=5):

    """
    Calculate how frequently each vector appears
    in the top-k nearest-neighbour lists of all other vectors.

    Returns:
        occurrence_counts
        top_neighbours
        cosine_matrix
    """

    n_vectors = embeddings.shape[0]

    if n_vectors <= top_k:

        raise ValueError(
            f"Need more than {top_k} vectors for top-{top_k} hubness analysis."
        )

    # ------------------------------------------------------------------
    # L2 normalize
    # ------------------------------------------------------------------

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    normalized = embeddings / norms

    # ------------------------------------------------------------------
    # Complete cosine similarity matrix
    # ------------------------------------------------------------------

    cosine_matrix = (
        normalized @ normalized.T
    )

    # ------------------------------------------------------------------
    # Remove self-similarity
    #
    # Set diagonal to -infinity so a chunk can never
    # select itself as its own nearest neighbour.
    # ------------------------------------------------------------------

    np.fill_diagonal(
        cosine_matrix,
        -np.inf
    )

    # ------------------------------------------------------------------
    # Get top-k neighbours for every chunk
    # ------------------------------------------------------------------

    top_neighbours = np.argpartition(
        -cosine_matrix,
        kth=top_k - 1,
        axis=1
    )[:, :top_k]

    # Sort the selected neighbours by cosine score
    # so that neighbour rank is also available.
    # ------------------------------------------------------------------

    rows = np.arange(n_vectors)[:, None]

    selected_scores = cosine_matrix[
        rows,
        top_neighbours
    ]

    ordering = np.argsort(
        -selected_scores,
        axis=1
    )

    top_neighbours = np.take_along_axis(
        top_neighbours,
        ordering,
        axis=1
    )

    top_scores = np.take_along_axis(
        selected_scores,
        ordering,
        axis=1
    )

    # ------------------------------------------------------------------
    # Count how many times each chunk appears
    # in the top-k lists.
    # ------------------------------------------------------------------

    occurrence_counts = np.bincount(
        top_neighbours.ravel(),
        minlength=n_vectors
    )

    return (
        occurrence_counts,
        top_neighbours,
        top_scores,
        cosine_matrix
    )


# ==========================================================================================
# RUN HUBNESS
# ==========================================================================================

print()
print("=" * 100)
print("CALCULATING HUBNESS")
print("=" * 100)

print()
print(f"Top-k nearest neighbours : {TOP_K}")

print()
print("B01...")
(
    b01_counts,
    b01_neighbours,
    b01_scores,
    b01_cosine
) = calculate_hubness(
    b01,
    TOP_K
)

print("Synthetic...")
(
    synthetic_counts,
    synthetic_neighbours,
    synthetic_scores,
    synthetic_cosine
) = calculate_hubness(
    synthetic,
    TOP_K
)


# ==========================================================================================
# BASIC INFORMATION
# ==========================================================================================

print()
print("=" * 100)
print("HUBNESS POPULATION")
print("=" * 100)

print()

print(
    f"{'Measure':<40}"
    f"{'B01':>20}"
    f"{'Synthetic':>20}"
)

print("-" * 80)

print(
    f"{'Number of vectors':<40}"
    f"{len(b01):>20}"
    f"{len(synthetic):>20}"
)

print(
    f"{'Top-k neighbours per query':<40}"
    f"{TOP_K:>20}"
    f"{TOP_K:>20}"
)

print(
    f"{'Total neighbour assignments':<40}"
    f"{len(b01) * TOP_K:>20}"
    f"{len(synthetic) * TOP_K:>20}"
)


# ==========================================================================================
# TOP HUBS
# ==========================================================================================

def get_top_hubs(counts, top_n=10):

    ordering = np.argsort(
        -counts
    )

    return ordering[:top_n]


b01_top_hubs = get_top_hubs(
    b01_counts,
    10
)

synthetic_top_hubs = get_top_hubs(
    synthetic_counts,
    10
)


# ==========================================================================================
# TOP HUB TABLE
# ==========================================================================================

print()
print("=" * 100)
print("TOP HUB CANDIDATES")
print("=" * 100)

print()
print(
    f"{'Rank':<10}"
    f"{'B01 Chunk':<20}"
    f"{'B01 Count':>15}"
    f"{'B01 % of Queries':>20}"
    f"{'Synthetic Chunk':<20}"
    f"{'Synthetic Count':>18}"
    f"{'Synthetic %':>15}"
)

print("-" * 118)


for rank in range(10):

    b01_idx = b01_top_hubs[rank]
    synthetic_idx = synthetic_top_hubs[rank]

    b01_count = b01_counts[b01_idx]
    synthetic_count = synthetic_counts[synthetic_idx]

    b01_percentage = (
        b01_count / len(b01)
    ) * 100

    synthetic_percentage = (
        synthetic_count / len(synthetic)
    ) * 100

    print(
        f"{rank + 1:<10}"
        f"{b01_idx:<20}"
        f"{b01_count:>15}"
        f"{b01_percentage:>19.2f}%"
        f"{synthetic_idx:<20}"
        f"{synthetic_count:>18}"
        f"{synthetic_percentage:>14.2f}%"
    )


# ==========================================================================================
# STRONGEST HUB
# ==========================================================================================

b01_hub = b01_top_hubs[0]
synthetic_hub = synthetic_top_hubs[0]

print()
print("=" * 100)
print("MOST FREQUENT HUB")
print("=" * 100)

print()

print(
    f"B01:\n"
    f"  Chunk index       : {b01_hub}\n"
    f"  Occurrences       : {b01_counts[b01_hub]}\n"
    f"  Percentage        : "
    f"{(b01_counts[b01_hub] / len(b01)) * 100:.2f}%"
)

print()

print(
    f"Synthetic:\n"
    f"  Chunk index       : {synthetic_hub}\n"
    f"  Occurrences       : {synthetic_counts[synthetic_hub]}\n"
    f"  Percentage        : "
    f"{(synthetic_counts[synthetic_hub] / len(synthetic)) * 100:.2f}%"
)


# ==========================================================================================
# TOP HUB'S NEIGHBOUR RELATIONSHIPS
# ==========================================================================================

def show_hub_relationships(
    embeddings,
    counts,
    top_hubs,
    name
):

    print()
    print("=" * 100)
    print(f"{name} — TOP HUB RELATIONSHIPS")
    print("=" * 100)

    for rank, hub_idx in enumerate(top_hubs[:5], start=1):

        print()
        print(
            f"Hub Rank {rank} — Chunk {hub_idx}"
        )

        print(
            f"Appears in top-{TOP_K} of "
            f"{counts[hub_idx]} other chunks"
        )

        print(
            "-" * 80
        )

        # Find all chunks for which this hub appears
        # in the top-k neighbour list.
        query_indices = np.where(
            np.any(
                calculate_hubness(
                    embeddings,
                    TOP_K
                )[1] == hub_idx,
                axis=1
            )
        )[0]

        # Exclude the hub itself
        query_indices = query_indices[
            query_indices != hub_idx
        ]

        print(
            f"Queried by {len(query_indices)} chunks."
        )


# ==========================================================================================
# HUB OCCURRENCE DISTRIBUTION
# ==========================================================================================

def print_occurrence_statistics(counts, name):

    print()
    print("=" * 100)
    print(f"{name} — HUB OCCURRENCE DISTRIBUTION")
    print("=" * 100)

    print()

    print(
        f"Mean occurrences : {np.mean(counts):.4f}"
    )

    print(
        f"Median occurrences : {np.median(counts):.4f}"
    )

    print(
        f"Std deviation     : {np.std(counts):.4f}"
    )

    print(
        f"Minimum           : {np.min(counts)}"
    )

    print(
        f"Maximum           : {np.max(counts)}"
    )

    print(
        f"Chunks never appearing in top-{TOP_K}: "
        f"{np.sum(counts == 0)}"
    )


print_occurrence_statistics(
    b01_counts,
    "B01"
)

print_occurrence_statistics(
    synthetic_counts,
    "Synthetic"
)


# ==========================================================================================
# SAVE RAW HUBNESS RESULTS
# ==========================================================================================

b01_hubness_file = (
    BASE_DIR
    / "Output_Files"
    / "B01_hubness_occurrence_counts.npy"
)

synthetic_hubness_file = (
    BASE_DIR
    / "Output_Files"
    / "Synthetic_hubness_occurrence_counts.npy"
)

np.save(
    b01_hubness_file,
    b01_counts
)

np.save(
    synthetic_hubness_file,
    synthetic_counts
)

print()
print("=" * 100)
print("OUTPUT FILES")
print("=" * 100)

print()
print(
    f"B01 hubness counts       : {b01_hubness_file}"
)

print(
    f"Synthetic hubness counts : {synthetic_hubness_file}"
)


# ==========================================================================================
# COMPLETE
# ==========================================================================================

print()
print("=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)

print()
print("Definition used:")
print(
    "For every chunk, its top-5 nearest neighbours were identified."
)

print(
    "Self-similarity was excluded."
)

print(
    "Each neighbour occurrence was counted across the population."
)

print(
    "The chunk with the highest occurrence count is the "
    "strongest hub candidate."
)

print()
print("No mean-centering was applied.")
print("No PCA was applied.")
print("No whitening was applied.")
print("No dimensionality reduction was applied.")

print()
print("Original embedding files were NOT modified.")

print("=" * 100)