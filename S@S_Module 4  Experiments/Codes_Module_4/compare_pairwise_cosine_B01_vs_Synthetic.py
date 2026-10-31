import numpy as np
import matplotlib.pyplot as plt


# ==========================================================================================
# Semantics @ Suren — Module 4
# B01 vs Synthetic Corpus
# GLOBAL PAIRWISE COSINE SIMILARITY COMPARISON
# ==========================================================================================


# ==========================================================================================
# INPUT FILES
# ==========================================================================================

B01_FILE = (
    "/home/surendran/LangChain/Semantics@Suren/"
    "S@S_Module 4  Experiments/20260424_B01/embeddings.npy"
)

SYNTHETIC_FILE = (
    "/home/surendran/LangChain/Semantics@Suren/"
    "S@S_Module 4  Experiments/Output_Files/synthetic_raw_embeddings.npy"
)


# ==========================================================================================
# LOAD EMBEDDINGS
# ==========================================================================================

print("=" * 100)
print("Semantics @ Suren — Module 4")
print("B01 vs SYNTHETIC — GLOBAL PAIRWISE COSINE SIMILARITY COMPARISON")
print("=" * 100)

print("\nLoading B01 embeddings...")
b01 = np.load(B01_FILE).astype(np.float64)

print(f"B01 shape      : {b01.shape}")

print("\nLoading Synthetic embeddings...")
synthetic = np.load(SYNTHETIC_FILE).astype(np.float64)

print(f"Synthetic shape: {synthetic.shape}")


# ==========================================================================================
# VALIDATION
# ==========================================================================================

if b01.ndim != 2:
    raise ValueError(f"B01 embeddings must be 2-dimensional. Found: {b01.ndim}")

if synthetic.ndim != 2:
    raise ValueError(
        f"Synthetic embeddings must be 2-dimensional. Found: {synthetic.ndim}"
    )

if b01.shape[1] != synthetic.shape[1]:
    raise ValueError(
        "B01 and Synthetic embeddings must have the same dimensionality.\n"
        f"B01 dimensions      : {b01.shape[1]}\n"
        f"Synthetic dimensions: {synthetic.shape[1]}"
    )


# ==========================================================================================
# PAIRWISE COSINE SIMILARITY
# ==========================================================================================

def pairwise_cosine_values(embeddings):
    """
    Calculate all unique pairwise cosine similarities.

    Assumes the input embeddings are already L2-normalized.
    The function nevertheless calculates cosine similarity explicitly
    using vector norms, so the result does not depend on that assumption.
    """

    norms = np.linalg.norm(embeddings, axis=1)

    if np.any(norms == 0):
        raise ValueError("Zero-norm vector detected. Cosine similarity is undefined.")

    normalized = embeddings / norms[:, np.newaxis]

    similarity_matrix = normalized @ normalized.T

    upper_triangle = np.triu_indices(len(embeddings), k=1)

    return similarity_matrix[upper_triangle]


# ==========================================================================================
# CALCULATE PAIRWISE DISTRIBUTIONS
# ==========================================================================================

print("\n" + "=" * 100)
print("CALCULATING PAIRWISE COSINE SIMILARITIES")
print("=" * 100)

print("\nB01...")
b01_cosines = pairwise_cosine_values(b01)

print("Synthetic...")
synthetic_cosines = pairwise_cosine_values(synthetic)


# ==========================================================================================
# SUMMARY STATISTICS
# ==========================================================================================

def calculate_statistics(values):

    return {
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
        "p01": np.percentile(values, 1),
        "p05": np.percentile(values, 5),
        "p10": np.percentile(values, 10),
        "p25": np.percentile(values, 25),
        "p50": np.percentile(values, 50),
        "p75": np.percentile(values, 75),
        "p90": np.percentile(values, 90),
        "p95": np.percentile(values, 95),
        "p99": np.percentile(values, 99),
    }


b01_stats = calculate_statistics(b01_cosines)
synthetic_stats = calculate_statistics(synthetic_cosines)


# ==========================================================================================
# PRINT BASIC INFORMATION
# ==========================================================================================

print("\n" + "=" * 100)
print("GLOBAL PAIRWISE COSINE COMPARISON")
print("=" * 100)

print(f"\n{'':25} {'B01':>20} {'Synthetic':>20}")
print("-" * 70)

print(f"{'Embeddings':25} {len(b01):>20} {len(synthetic):>20}")
print(f"{'Dimensions':25} {b01.shape[1]:>20} {synthetic.shape[1]}")
print(f"{'Unique pairs':25} {len(b01_cosines):>20} {len(synthetic_cosines):>20}")


# ==========================================================================================
# COSINE DISTRIBUTION
# ==========================================================================================

print("\n")
print("=" * 100)
print("COSINE DISTRIBUTION")
print("=" * 100)

print(
    f"\n{'Statistic':25}"
    f"{'B01':>20}"
    f"{'Synthetic':>20}"
)

print("-" * 70)

print(f"{'Mean':25}{b01_stats['mean']:>20.6f}{synthetic_stats['mean']:>20.6f}")
print(f"{'Median':25}{b01_stats['median']:>20.6f}{synthetic_stats['median']:>20.6f}")
print(f"{'Std Deviation':25}{b01_stats['std']:>20.6f}{synthetic_stats['std']:>20.6f}")
print(f"{'Minimum':25}{b01_stats['min']:>20.6f}{synthetic_stats['min']:>20.6f}")
print(f"{'Maximum':25}{b01_stats['max']:>20.6f}{synthetic_stats['max']:>20.6f}")


# ==========================================================================================
# PERCENTILES
# ==========================================================================================

print("\nPercentiles")
print("-" * 100)

print(
    f"{'Percentile':15}"
    f"{'B01':>20}"
    f"{'Synthetic':>20}"
)

print("-" * 60)

print(f"{'P01':15}{b01_stats['p01']:>20.6f}{synthetic_stats['p01']:>20.6f}")
print(f"{'P05':15}{b01_stats['p05']:>20.6f}{synthetic_stats['p05']:>20.6f}")
print(f"{'P10':15}{b01_stats['p10']:>20.6f}{synthetic_stats['p10']:>20.6f}")
print(f"{'P25':15}{b01_stats['p25']:>20.6f}{synthetic_stats['p25']:>20.6f}")
print(f"{'P50':15}{b01_stats['p50']:>20.6f}{synthetic_stats['p50']:>20.6f}")
print(f"{'P75':15}{b01_stats['p75']:>20.6f}{synthetic_stats['p75']:>20.6f}")
print(f"{'P90':15}{b01_stats['p90']:>20.6f}{synthetic_stats['p90']:>20.6f}")
print(f"{'P95':15}{b01_stats['p95']:>20.6f}{synthetic_stats['p95']:>20.6f}")
print(f"{'P99':15}{b01_stats['p99']:>20.6f}{synthetic_stats['p99']:>20.6f}")


# ==========================================================================================
# DISTRIBUTION QUANTITIES
# ==========================================================================================

print("\n")
print("=" * 100)
print("ADDITIONAL DISTRIBUTION SUMMARY")
print("=" * 100)

print(
    f"\n{'Measure':35}"
    f"{'B01':>20}"
    f"{'Synthetic':>20}"
)

print("-" * 75)

print(
    f"{'Pairs with cosine > 0.90':35}"
    f"{np.sum(b01_cosines > 0.90):>20}"
    f"{np.sum(synthetic_cosines > 0.90):>20}"
)

print(
    f"{'Pairs with cosine > 0.80':35}"
    f"{np.sum(b01_cosines > 0.80):>20}"
    f"{np.sum(synthetic_cosines > 0.80):>20}"
)

print(
    f"{'Pairs with cosine > 0.70':35}"
    f"{np.sum(b01_cosines > 0.70):>20}"
    f"{np.sum(synthetic_cosines > 0.70):>20}"
)

print(
    f"{'Pairs with cosine > 0.50':35}"
    f"{np.sum(b01_cosines > 0.50):>20}"
    f"{np.sum(synthetic_cosines > 0.50):>20}"
)

print(
    f"{'Pairs with cosine < 0.00':35}"
    f"{np.sum(b01_cosines < 0.00):>20}"
    f"{np.sum(synthetic_cosines < 0.00):>20}"
)


# ==========================================================================================
# HISTOGRAM — SIDE BY SIDE
# ==========================================================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].hist(b01_cosines, bins=40, edgecolor="black")
axes[0].set_title("B01 — Pairwise Cosine Similarity")
axes[0].set_xlabel("Cosine Similarity")
axes[0].set_ylabel("Number of Pairs")
axes[0].grid(axis="y", alpha=0.3)

axes[1].hist(synthetic_cosines, bins=40, edgecolor="black")
axes[1].set_title("Synthetic — Pairwise Cosine Similarity")
axes[1].set_xlabel("Cosine Similarity")
axes[1].set_ylabel("Number of Pairs")
axes[1].grid(axis="y", alpha=0.3)

plt.suptitle(
    "B01 vs Synthetic — Global Pairwise Cosine Distribution",
    fontsize=14
)

plt.tight_layout()

plt.show()


# ==========================================================================================
# COMPLETE
# ==========================================================================================

print("\n" + "=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)

print("\nNo mean-centering was applied.")
print("No PCA was applied.")
print("No whitening was applied.")
print("No dimensionality reduction was applied.")
print("Original embeddings were not modified.")
print("=" * 100)