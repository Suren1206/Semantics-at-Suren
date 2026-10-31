import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Semantics @ Suren
# Original vs Mean-Centered Pairwise Cosine Distribution
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Input files created in the previous experiment
ORIGINAL_FILE = BASE_DIR / "20260424_B01" / "embeddings.npy"
MEAN_CENTERED_FILE = BASE_DIR / "Output_Files" / "mean_centered_160x1024.npy"


# ============================================================
# FUNCTION: CALCULATE UNIQUE PAIRWISE COSINES
# ============================================================

def get_pairwise_cosines(embeddings):

    # L2-normalize explicitly before calculating cosine
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms

    # Full cosine similarity matrix
    cosine_matrix = normalized @ normalized.T

    # Keep only unique pairs
    upper = np.triu_indices(len(embeddings), k=1)

    return cosine_matrix[upper]


# ============================================================
# FUNCTION: PRINT DISTRIBUTION
# ============================================================

def print_distribution(name, cosines):

    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)

    print(f"\nNumber of pairs     : {len(cosines):,}")

    print("\nCOSINE DISTRIBUTION")
    print("-" * 100)

    print(f"Mean                : {np.mean(cosines):.6f}")
    print(f"Median              : {np.median(cosines):.6f}")
    print(f"Std Deviation       : {np.std(cosines):.6f}")
    print(f"Minimum             : {np.min(cosines):.6f}")
    print(f"Maximum             : {np.max(cosines):.6f}")

    print("\nPercentiles")
    print("-" * 100)

    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"P{p:02d}                 : {np.percentile(cosines, p):.6f}")


# ============================================================
# LOAD ORIGINAL EMBEDDINGS
# ============================================================

print("=" * 100)
print("GLOBAL PAIRWISE COSINE COMPARISON")
print("=" * 100)

print("\nLoading original embeddings:")
print(ORIGINAL_FILE)

original = np.load(ORIGINAL_FILE).astype(np.float64)

print(f"Original shape       : {original.shape}")


# ============================================================
# LOAD MEAN-CENTERED EMBEDDINGS
# ============================================================

print("\nLoading mean-centered embeddings:")
print(MEAN_CENTERED_FILE)

mean_centered = np.load(MEAN_CENTERED_FILE).astype(np.float64)

print(f"Mean-centered shape  : {mean_centered.shape}")


# ============================================================
# VALIDATE
# ============================================================

if original.shape != mean_centered.shape:
    raise ValueError(
        f"Shape mismatch: Original {original.shape} "
        f"vs Mean-centered {mean_centered.shape}"
    )

if original.shape[0] != 160:
    print(
        f"\nWARNING: Expected 160 embeddings, "
        f"but found {original.shape[0]}."
    )


# ============================================================
# CALCULATE PAIRWISE COSINES
# ============================================================

print("\nCalculating original pairwise cosine similarities...")

original_cosines = get_pairwise_cosines(original)

print("Calculating mean-centered pairwise cosine similarities...")

centered_cosines = get_pairwise_cosines(mean_centered)


# ============================================================
# PRINT ORIGINAL DISTRIBUTION
# ============================================================

print_distribution(
    "ORIGINAL 160 VECTORS — PAIRWISE COSINE DISTRIBUTION",
    original_cosines
)


# ============================================================
# PRINT MEAN-CENTERED DISTRIBUTION
# ============================================================

print_distribution(
    "MEAN-CENTERED 160 VECTORS — PAIRWISE COSINE DISTRIBUTION",
    centered_cosines
)


# ============================================================
# DIRECT COMPARISON
# ============================================================

print("\n" + "=" * 100)
print("DIRECT COMPARISON")
print("=" * 100)

print("\nMetric                 Original        Mean-Centered       Change")
print("-" * 100)

metrics = {
    "Mean": (
        np.mean(original_cosines),
        np.mean(centered_cosines)
    ),
    "Median": (
        np.median(original_cosines),
        np.median(centered_cosines)
    ),
    "Std Deviation": (
        np.std(original_cosines),
        np.std(centered_cosines)
    ),
    "Minimum": (
        np.min(original_cosines),
        np.min(centered_cosines)
    ),
    "Maximum": (
        np.max(original_cosines),
        np.max(centered_cosines)
    ),
    "P01": (
        np.percentile(original_cosines, 1),
        np.percentile(centered_cosines, 1)
    ),
    "P05": (
        np.percentile(original_cosines, 5),
        np.percentile(centered_cosines, 5)
    ),
    "P10": (
        np.percentile(original_cosines, 10),
        np.percentile(centered_cosines, 10)
    ),
    "P25": (
        np.percentile(original_cosines, 25),
        np.percentile(centered_cosines, 25)
    ),
    "P50": (
        np.percentile(original_cosines, 50),
        np.percentile(centered_cosines, 50)
    ),
    "P75": (
        np.percentile(original_cosines, 75),
        np.percentile(centered_cosines, 75)
    ),
    "P90": (
        np.percentile(original_cosines, 90),
        np.percentile(centered_cosines, 90)
    ),
    "P95": (
        np.percentile(original_cosines, 95),
        np.percentile(centered_cosines, 95)
    ),
    "P99": (
        np.percentile(original_cosines, 99),
        np.percentile(centered_cosines, 99)
    ),
}

for metric, (original_value, centered_value) in metrics.items():

    change = centered_value - original_value

    print(
        f"{metric:<22}"
        f"{original_value:>10.6f}      "
        f"{centered_value:>10.6f}      "
        f"{change:>+10.6f}"
    )


# ============================================================
# HISTOGRAM — SIDE BY SIDE
# ============================================================

print("\n" + "=" * 100)
print("HISTOGRAM — ORIGINAL VS MEAN-CENTERED")
print("=" * 100)

plt.figure(figsize=(12, 7))

plt.hist(
    original_cosines,
    bins=40,
    alpha=0.6,
    label="Original"
)

plt.hist(
    centered_cosines,
    bins=40,
    alpha=0.6,
    label="Mean-centered"
)

plt.xlabel("Cosine Similarity")
plt.ylabel("Number of Pairs")
plt.title(
    "Pairwise Cosine Similarity: Original vs Mean-Centered Embeddings"
)

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# EXPERIMENT COMPLETE
# ============================================================

print("\n" + "=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)