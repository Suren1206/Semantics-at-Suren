import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================================================================
# Semantics @ Suren — Module 4
# EXPERIMENT 3
# PAIRWISE COSINE SIMILARITY
# BEFORE vs AFTER MEAN-CENTERING
# B01 vs SYNTHETIC
# ==========================================================================================


# ==========================================================================================
# PATHS
# ==========================================================================================

BASE_DIR = Path(
    "/home/surendran/LangChain/Semantics@Suren/"
    "S@S_Module 4  Experiments"
)

# ------------------------------
# Raw embeddings
# ------------------------------

B01_RAW_FILE = (
    BASE_DIR
    / "20260424_B01"
    / "embeddings.npy"
)

SYNTHETIC_RAW_FILE = (
    BASE_DIR
    / "Output_Files"
    / "synthetic_raw_embeddings.npy"
)

# ------------------------------
# Mean-centered embeddings
# ------------------------------

B01_CENTERED_FILE = (
    BASE_DIR
    / "Output_Files"
    / "mean_centered_B01.npy"
)

SYNTHETIC_CENTERED_FILE = (
    BASE_DIR
    / "Output_Files"
    / "mean_centered_Synthetic.npy"
)


# ==========================================================================================
# HEADER
# ==========================================================================================

print("=" * 100)
print("Semantics @ Suren — Module 4")
print("EXPERIMENT 3 — PAIRWISE COSINE SIMILARITY")
print("BEFORE vs AFTER MEAN-CENTERING")
print("B01 vs SYNTHETIC")
print("=" * 100)


# ==========================================================================================
# LOAD FILES
# ==========================================================================================

print("\nLoading B01 raw embeddings...")
b01_raw = np.load(B01_RAW_FILE).astype(np.float64)

print(f"B01 raw shape       : {b01_raw.shape}")

print("\nLoading B01 mean-centered embeddings...")
b01_centered = np.load(B01_CENTERED_FILE).astype(np.float64)

print(f"B01 centered shape  : {b01_centered.shape}")

print("\nLoading Synthetic raw embeddings...")
synthetic_raw = np.load(SYNTHETIC_RAW_FILE).astype(np.float64)

print(f"Synthetic raw shape : {synthetic_raw.shape}")

print("\nLoading Synthetic mean-centered embeddings...")
synthetic_centered = np.load(SYNTHETIC_CENTERED_FILE).astype(np.float64)

print(f"Synthetic centered shape : {synthetic_centered.shape}")


# ==========================================================================================
# VALIDATION
# ==========================================================================================

def validate_pair(raw, centered, name):

    if raw.ndim != 2:
        raise ValueError(
            f"{name} raw embeddings must be 2-dimensional."
        )

    if centered.ndim != 2:
        raise ValueError(
            f"{name} centered embeddings must be 2-dimensional."
        )

    if raw.shape != centered.shape:
        raise ValueError(
            f"\n{name} raw and centered embeddings have different shapes.\n"
            f"Raw      : {raw.shape}\n"
            f"Centered : {centered.shape}"
        )

    if np.any(np.linalg.norm(raw, axis=1) == 0):
        raise ValueError(
            f"{name} raw embeddings contain a zero-norm vector."
        )

    if np.any(np.linalg.norm(centered, axis=1) == 0):
        raise ValueError(
            f"{name} centered embeddings contain a zero-norm vector."
        )


validate_pair(
    b01_raw,
    b01_centered,
    "B01"
)

validate_pair(
    synthetic_raw,
    synthetic_centered,
    "Synthetic"
)


if b01_raw.shape[1] != synthetic_raw.shape[1]:
    raise ValueError(
        "\nB01 and Synthetic dimensions do not match.\n"
        f"B01       : {b01_raw.shape[1]}\n"
        f"Synthetic : {synthetic_raw.shape[1]}"
    )


# ==========================================================================================
# COSINE SIMILARITY
# ==========================================================================================

def pairwise_cosine_values(embeddings):

    norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    normalized = (
        embeddings / norms[:, np.newaxis]
    )

    similarity_matrix = (
        normalized @ normalized.T
    )

    upper_triangle = np.triu_indices(
        len(embeddings),
        k=1
    )

    return similarity_matrix[upper_triangle]


# ==========================================================================================
# CALCULATE ALL FOUR DISTRIBUTIONS
# ==========================================================================================

print("\n" + "=" * 100)
print("CALCULATING PAIRWISE COSINE DISTRIBUTIONS")
print("=" * 100)

print("\nB01 — Raw...")
b01_raw_cosine = pairwise_cosine_values(
    b01_raw
)

print("B01 — Mean-centered...")
b01_centered_cosine = pairwise_cosine_values(
    b01_centered
)

print("Synthetic — Raw...")
synthetic_raw_cosine = pairwise_cosine_values(
    synthetic_raw
)

print("Synthetic — Mean-centered...")
synthetic_centered_cosine = pairwise_cosine_values(
    synthetic_centered
)


# ==========================================================================================
# STATISTICS
# ==========================================================================================

def statistics(values):

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


b01_raw_stats = statistics(b01_raw_cosine)
b01_centered_stats = statistics(b01_centered_cosine)

synthetic_raw_stats = statistics(synthetic_raw_cosine)
synthetic_centered_stats = statistics(synthetic_centered_cosine)


# ==========================================================================================
# BASIC INFORMATION
# ==========================================================================================

print("\n" + "=" * 100)
print("PAIRWISE COMPARISON POPULATION")
print("=" * 100)

print()

print(
    f"{'Measure':<35}"
    f"{'B01':>18}"
    f"{'Synthetic':>18}"
)

print("-" * 75)

print(
    f"{'Number of vectors':<35}"
    f"{b01_raw.shape[0]:>18}"
    f"{synthetic_raw.shape[0]:>18}"
)

print(
    f"{'Dimensions':<35}"
    f"{b01_raw.shape[1]:>18}"
    f"{synthetic_raw.shape[1]:>18}"
)

print(
    f"{'Unique pairwise comparisons':<35}"
    f"{len(b01_raw_cosine):>18}"
    f"{len(synthetic_raw_cosine):>18}"
)


# ==========================================================================================
# COSINE DISTRIBUTION — PARALLEL
# ==========================================================================================

print("\n" + "=" * 100)
print("COSINE DISTRIBUTION — RAW vs MEAN-CENTERED")
print("=" * 100)

print()

print(
    f"{'Statistic':<25}"
    f"{'B01 Raw':>18}"
    f"{'B01 Centered':>20}"
    f"{'Synthetic Raw':>20}"
    f"{'Synthetic Centered':>23}"
)

print("-" * 110)

print(
    f"{'Mean':<25}"
    f"{b01_raw_stats['mean']:>18.6f}"
    f"{b01_centered_stats['mean']:>20.6f}"
    f"{synthetic_raw_stats['mean']:>20.6f}"
    f"{synthetic_centered_stats['mean']:>23.6f}"
)

print(
    f"{'Median':<25}"
    f"{b01_raw_stats['median']:>18.6f}"
    f"{b01_centered_stats['median']:>20.6f}"
    f"{synthetic_raw_stats['median']:>20.6f}"
    f"{synthetic_centered_stats['median']:>23.6f}"
)

print(
    f"{'Std Deviation':<25}"
    f"{b01_raw_stats['std']:>18.6f}"
    f"{b01_centered_stats['std']:>20.6f}"
    f"{synthetic_raw_stats['std']:>20.6f}"
    f"{synthetic_centered_stats['std']:>23.6f}"
)

print(
    f"{'Minimum':<25}"
    f"{b01_raw_stats['min']:>18.6f}"
    f"{b01_centered_stats['min']:>20.6f}"
    f"{synthetic_raw_stats['min']:>20.6f}"
    f"{synthetic_centered_stats['min']:>23.6f}"
)

print(
    f"{'Maximum':<25}"
    f"{b01_raw_stats['max']:>18.6f}"
    f"{b01_centered_stats['max']:>20.6f}"
    f"{synthetic_raw_stats['max']:>20.6f}"
    f"{synthetic_centered_stats['max']:>23.6f}"
)


# ==========================================================================================
# PERCENTILES
# ==========================================================================================

print("\n" + "=" * 100)
print("PERCENTILE COMPARISON")
print("=" * 100)

print()

print(
    f"{'Percentile':<15}"
    f"{'B01 Raw':>18}"
    f"{'B01 Centered':>20}"
    f"{'Synthetic Raw':>20}"
    f"{'Synthetic Centered':>23}"
)

print("-" * 100)

percentiles = [
    ("P01", "p01"),
    ("P05", "p05"),
    ("P10", "p10"),
    ("P25", "p25"),
    ("P50", "p50"),
    ("P75", "p75"),
    ("P90", "p90"),
    ("P95", "p95"),
    ("P99", "p99"),
]

for label, key in percentiles:

    print(
        f"{label:<15}"
        f"{b01_raw_stats[key]:>18.6f}"
        f"{b01_centered_stats[key]:>20.6f}"
        f"{synthetic_raw_stats[key]:>20.6f}"
        f"{synthetic_centered_stats[key]:>23.6f}"
    )


# ==========================================================================================
# CHANGE CAUSED BY MEAN-CENTERING
# ==========================================================================================

print("\n" + "=" * 100)
print("CHANGE CAUSED BY MEAN-CENTERING")
print("=" * 100)

print()

print(
    f"{'Statistic':<25}"
    f"{'B01 Change':>20}"
    f"{'Synthetic Change':>25}"
)

print("-" * 75)

print(
    f"{'Mean':<25}"
    f"{b01_centered_stats['mean'] - b01_raw_stats['mean']:>20.6f}"
    f"{synthetic_centered_stats['mean'] - synthetic_raw_stats['mean']:>25.6f}"
)

print(
    f"{'Median':<25}"
    f"{b01_centered_stats['median'] - b01_raw_stats['median']:>20.6f}"
    f"{synthetic_centered_stats['median'] - synthetic_raw_stats['median']:>25.6f}"
)

print(
    f"{'Std Deviation':<25}"
    f"{b01_centered_stats['std'] - b01_raw_stats['std']:>20.6f}"
    f"{synthetic_centered_stats['std'] - synthetic_raw_stats['std']:>25.6f}"
)

print(
    f"{'Minimum':<25}"
    f"{b01_centered_stats['min'] - b01_raw_stats['min']:>20.6f}"
    f"{synthetic_centered_stats['min'] - synthetic_raw_stats['min']:>25.6f}"
)

print(
    f"{'Maximum':<25}"
    f"{b01_centered_stats['max'] - b01_raw_stats['max']:>20.6f}"
    f"{synthetic_centered_stats['max'] - synthetic_raw_stats['max']:>25.6f}"
)


# ==========================================================================================
# HISTOGRAMS — SIDE BY SIDE
# ==========================================================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(15, 11)
)


# ------------------------------------------------------------------------------------------
# B01 Raw
# ------------------------------------------------------------------------------------------

axes[0, 0].hist(
    b01_raw_cosine,
    bins=40,
    edgecolor="black"
)

axes[0, 0].set_title(
    "B01 — Raw"
)

axes[0, 0].set_xlabel(
    "Cosine Similarity"
)

axes[0, 0].set_ylabel(
    "Number of Pairs"
)

axes[0, 0].grid(
    axis="y",
    alpha=0.3
)


# ------------------------------------------------------------------------------------------
# B01 Centered
# ------------------------------------------------------------------------------------------

axes[0, 1].hist(
    b01_centered_cosine,
    bins=40,
    edgecolor="black"
)

axes[0, 1].set_title(
    "B01 — Mean-Centered"
)

axes[0, 1].set_xlabel(
    "Cosine Similarity"
)

axes[0, 1].set_ylabel(
    "Number of Pairs"
)

axes[0, 1].grid(
    axis="y",
    alpha=0.3
)


# ------------------------------------------------------------------------------------------
# Synthetic Raw
# ------------------------------------------------------------------------------------------

axes[1, 0].hist(
    synthetic_raw_cosine,
    bins=40,
    edgecolor="black"
)

axes[1, 0].set_title(
    "Synthetic — Raw"
)

axes[1, 0].set_xlabel(
    "Cosine Similarity"
)

axes[1, 0].set_ylabel(
    "Number of Pairs"
)

axes[1, 0].grid(
    axis="y",
    alpha=0.3
)


# ------------------------------------------------------------------------------------------
# Synthetic Centered
# ------------------------------------------------------------------------------------------

axes[1, 1].hist(
    synthetic_centered_cosine,
    bins=40,
    edgecolor="black"
)

axes[1, 1].set_title(
    "Synthetic — Mean-Centered"
)

axes[1, 1].set_xlabel(
    "Cosine Similarity"
)

axes[1, 1].set_ylabel(
    "Number of Pairs"
)

axes[1, 1].grid(
    axis="y",
    alpha=0.3
)


plt.suptitle(
    "B01 vs Synthetic — Pairwise Cosine Before and After Mean-Centering",
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

print("\nComparison performed for:")

print("  B01       : Raw vs Mean-Centered")
print("  Synthetic : Raw vs Mean-Centered")

print("\nNo PCA was applied.")
print("No whitening was applied.")
print("No dimensionality reduction was applied.")

print("\nOriginal embedding files were NOT modified.")

print("=" * 100)