"""
================================================================================
Semantics @ Suren — Module 4
GROUP-WISE PAIRWISE COSINE ANALYSIS
================================================================================

Corpus:
    Synthetic_Corpus.ods

Column A:
    Group / semantic category
    -> METADATA ONLY

Column D:
    Actual textual content
    -> This is what was embedded

Input embeddings:
    Output_Files/synthetic_raw_embeddings.npy

Analysis:
    1. Number of vectors in each group
    2. Within-group pairwise cosine statistics
    3. Mean
    4. Median
    5. Standard deviation
    6. Minimum
    7. Maximum
    8. Group × group pairwise cosine similarity matrix

IMPORTANT:
    No mean-centering
    No PCA
    No whitening
    No dimensionality reduction
    No transformation of the embeddings

The embeddings are used exactly as stored.

For the Group × Group matrix:
    - Diagonal = mean cosine similarity among distinct vectors
      within that group.
    - Off-diagonal = mean cosine similarity across all vector pairs
      belonging to the two groups.

================================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# This script is inside:
#
# S@S_Module 4  Experiments/
#     └── Codes_Module_4/
#
# Therefore:

CODE_DIR = Path(__file__).resolve().parent

PROJECT_DIR = CODE_DIR.parent


# Corpus
CORPUS_FILE = PROJECT_DIR / "Synthetic_Corpus.ods"


# Raw embeddings generated from Column D
EMBEDDING_FILE = (
    PROJECT_DIR
    / "Output_Files"
    / "synthetic_raw_embeddings.npy"
)


# =============================================================================
# HEADER
# =============================================================================

print("=" * 100)
print("Semantics @ Suren — Module 4")
print("GROUP-WISE PAIRWISE COSINE ANALYSIS")
print("=" * 100)


# =============================================================================
# CHECK FILES
# =============================================================================

print()
print("Input files")
print("-" * 100)

print(f"Corpus     : {CORPUS_FILE}")
print(f"Embeddings : {EMBEDDING_FILE}")


if not CORPUS_FILE.exists():

    raise FileNotFoundError(
        f"\nCorpus file not found:\n{CORPUS_FILE}"
    )


if not EMBEDDING_FILE.exists():

    raise FileNotFoundError(
        f"\nEmbedding file not found:\n{EMBEDDING_FILE}"
    )


# =============================================================================
# LOAD CORPUS
# =============================================================================

print()
print("Loading corpus...")

# Read without assuming that the first row is a header.
#
# This preserves every row of the corpus.
#
# Column A -> index 0
# Column D -> index 3

df = pd.read_excel(
    CORPUS_FILE,
    engine="odf",
    header=None
)

print(f"Corpus shape : {df.shape}")


if df.shape[1] < 4:

    raise ValueError(
        "The corpus must contain at least four columns."
    )


# =============================================================================
# LOAD EMBEDDINGS
# =============================================================================

print()
print("Loading embeddings...")

embeddings = np.load(EMBEDDING_FILE)

print(f"Embedding shape : {embeddings.shape}")


if embeddings.ndim != 2:

    raise ValueError(
        f"Expected a 2-D embedding matrix. "
        f"Found shape: {embeddings.shape}"
    )


n_vectors, dimensions = embeddings.shape

# =============================================================================
# ALIGN CORPUS ROWS WITH EMBEDDINGS
# =============================================================================

if len(df) == n_vectors + 1:

    print()
    print("Corpus contains one additional row compared with embeddings.")
    print("Using the 50 rows corresponding to the existing embeddings.")

    df = df.iloc[:n_vectors].copy()

elif len(df) != n_vectors:

    raise ValueError(
        "\nMismatch between corpus rows and embeddings.\n"
        f"Corpus rows : {len(df)}\n"
        f"Embeddings  : {n_vectors}\n"
    )


# =============================================================================
# EXTRACT GROUP LABELS
# =============================================================================

# Column A is metadata only.
# It is NEVER passed to the embedding model.

groups = df.iloc[:, 0].astype(str).str.strip().values


# =============================================================================
# DISPLAY GROUP COUNTS
# =============================================================================

unique_groups = list(dict.fromkeys(groups))


print()
print("=" * 100)
print("GROUP INFORMATION")
print("=" * 100)

print()
print(f"Total vectors : {n_vectors}")
print(f"Dimensions    : {dimensions}")
print(f"Groups        : {len(unique_groups)}")

print()
print(
    f"{'Group':<35}"
    f"{'Vectors':>12}"
)

print("-" * 100)

for group in unique_groups:

    count = np.sum(groups == group)

    print(
        f"{group:<35}"
        f"{count:>12}"
    )


# =============================================================================
# TEMPORARY NORMALIZATION FOR COSINE CALCULATION
# =============================================================================

# IMPORTANT:
#
# We do NOT modify the original embeddings.
#
# This is only a temporary normalized copy required to calculate
# cosine similarity.

norms = np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)


if np.any(norms == 0):

    zero_vectors = np.where(norms.flatten() == 0)[0]

    raise ValueError(
        f"Zero-norm embeddings found at indices: {zero_vectors.tolist()}"
    )


normalized_embeddings = embeddings / norms


# =============================================================================
# GLOBAL COSINE MATRIX
# =============================================================================

# cosine_matrix[i, j] = cosine similarity between vector i and vector j

cosine_matrix = (
    normalized_embeddings @ normalized_embeddings.T
)


# =============================================================================
# WITHIN-GROUP ANALYSIS
# =============================================================================

print()
print("=" * 100)
print("WITHIN-GROUP PAIRWISE COSINE ANALYSIS")
print("=" * 100)


group_statistics = {}


for group in unique_groups:

    indices = np.where(groups == group)[0]

    group_size = len(indices)

    print()
    print("-" * 100)
    print(f"GROUP: {group}")
    print("-" * 100)

    print(f"Number of vectors : {group_size}")

    # A group with one vector has no unique within-group pair.
    if group_size < 2:

        print("Within-group pairs: 0")
        print("Mean              : N/A")
        print("Median            : N/A")
        print("Std Deviation     : N/A")
        print("Minimum           : N/A")
        print("Maximum           : N/A")

        group_statistics[group] = None

        continue


    # Extract the group's cosine sub-matrix
    group_cosine_matrix = cosine_matrix[
        np.ix_(indices, indices)
    ]


    # Upper triangle excludes:
    # - diagonal self-similarity
    # - duplicate pairs

    upper_indices = np.triu_indices(
        group_size,
        k=1
    )


    within_cosines = group_cosine_matrix[
        upper_indices
    ]


    group_statistics[group] = within_cosines


    print(f"Within-group pairs: {len(within_cosines):,}")

    print()
    print("COSINE STATISTICS")
    print("-" * 100)

    print(
        f"Mean              : "
        f"{np.mean(within_cosines):.8f}"
    )

    print(
        f"Median            : "
        f"{np.median(within_cosines):.8f}"
    )

    print(
        f"Std Deviation     : "
        f"{np.std(within_cosines):.8f}"
    )

    print(
        f"Minimum           : "
        f"{np.min(within_cosines):.8f}"
    )

    print(
        f"Maximum           : "
        f"{np.max(within_cosines):.8f}"
    )


# =============================================================================
# GROUP × GROUP SIMILARITY MATRIX
# =============================================================================

print()
print("=" * 100)
print("GROUP × GROUP PAIRWISE COSINE SIMILARITY MATRIX")
print("=" * 100)

print()
print("Each cell = mean cosine similarity across all vector pairs")
print("between the corresponding two groups.")
print()
print("Diagonal = mean within-group cosine similarity")
print("excluding self-similarity.")


# -----------------------------------------------------------------------------
# Calculate matrix
# -----------------------------------------------------------------------------

group_matrix = np.zeros(
    (len(unique_groups), len(unique_groups))
)


for i, group_a in enumerate(unique_groups):

    indices_a = np.where(groups == group_a)[0]


    for j, group_b in enumerate(unique_groups):

        indices_b = np.where(groups == group_b)[0]


        if i == j:

            # Within-group:
            # exclude self-similarity and duplicate pairs.

            if len(indices_a) < 2:

                group_matrix[i, j] = np.nan

            else:

                values = cosine_matrix[
                    np.ix_(indices_a, indices_a)
                ]

                upper = np.triu_indices(
                    len(indices_a),
                    k=1
                )

                pair_values = values[upper]

                group_matrix[i, j] = np.mean(
                    pair_values
                )


        else:

            # Between-group:
            # every vector in A against every vector in B.

            pair_values = cosine_matrix[
                np.ix_(indices_a, indices_b)
            ]

            group_matrix[i, j] = np.mean(
                pair_values
            )


# =============================================================================
# PRINT MATRIX
# =============================================================================

print()

# Determine display width
group_width = max(
    15,
    max(len(str(g)) for g in unique_groups) + 2
)

print(
    f"{'Group':<{group_width}}",
    end=""
)

for group in unique_groups:

    print(
        f"{group[:14]:>16}",
        end=""
    )

print()

print("-" * (group_width + 16 * len(unique_groups)))


for i, group in enumerate(unique_groups):

    print(
        f"{group:<{group_width}}",
        end=""
    )

    for j in range(len(unique_groups)):

        value = group_matrix[i, j]

        if np.isnan(value):

            print(f"{'N/A':>16}", end="")

        else:

            print(
                f"{value:>16.6f}",
                end=""
            )

    print()


# =============================================================================
# BETWEEN-GROUP PAIR DETAILS
# =============================================================================

print()
print("=" * 100)
print("BETWEEN-GROUP MEAN COSINE SIMILARITIES")
print("=" * 100)

print()

for i in range(len(unique_groups)):

    for j in range(i + 1, len(unique_groups)):

        group_a = unique_groups[i]
        group_b = unique_groups[j]

        print(
            f"{group_a}  <->  {group_b}"
            f" : {group_matrix[i, j]:.8f}"
        )

# =============================================================================
# GROUP-WISE COSINE HISTOGRAMS
# =============================================================================

import matplotlib.pyplot as plt

print()
print("=" * 100)
print("GROUP-WISE PAIRWISE COSINE SIMILARITY HISTOGRAMS")
print("=" * 100)

# =============================================================================
# Create one small histogram for each group
# =============================================================================
num_groups = len(unique_groups)

fig, axes = plt.subplots(
    1,
    num_groups,
    figsize=(4 * num_groups, 4),
    squeeze=False
)

axes = axes.flatten()

for i, group in enumerate(unique_groups):

    ax = axes[i]

    values = group_statistics[group]

    if values is None or len(values) == 0:

        ax.text(
            0.5,
            0.5,
            "No pairwise data",
            ha="center",
            va="center"
        )

        ax.set_title(group)

        continue

    ax.hist(
        values,
        bins=15,
        edgecolor="black"
    )

    ax.set_title(group)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Frequency")

    ax.grid(
        axis="y",
        alpha=0.3
    )


fig.suptitle(
    "Within-Group Pairwise Cosine Similarity Distributions",
    fontsize=14
)

plt.tight_layout()

plt.show()


# =============================================================================
# COMPLETION
# =============================================================================

print()
print("=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)

print()
print("Analysis used:")
print("  Column A : grouping metadata ONLY")
print("  Column D : content represented by embeddings")
print()
print("No mean-centering was applied.")
print("No PCA was applied.")
print("No whitening was applied.")
print("No dimensionality reduction was applied.")
print("Original embeddings were not modified.")