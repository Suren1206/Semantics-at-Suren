import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Semantics @ Suren — Global Pairwise Cosine Analysis
# ============================================================

# Project root:
# ~/LangChain/Semantics@Suren/S@S_Module 4  Experiments
BASE_DIR = Path(__file__).resolve().parent.parent

# Vector store
#EMBEDDING_FILE = BASE_DIR / "20260424_B01" / "embeddings.npy"

# Vector store
EMBEDDING_FILE = BASE_DIR / "Output_Files" / "synthetic_raw_embeddings.npy"

# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("=" * 100)
print("GLOBAL PAIRWISE COSINE ANALYSIS")
print("=" * 100)

print(f"\nLoading embeddings from:")
print(EMBEDDING_FILE)

embeddings = np.load(EMBEDDING_FILE)

# Convert to float64 for numerical precision
embeddings = embeddings.astype(np.float64)

n_embeddings, dimensions = embeddings.shape

print(f"\nEmbeddings        : {n_embeddings}")
print(f"Dimensions        : {dimensions}")

# ============================================================
# CHECK L2 NORMALIZATION
# ============================================================

norms = np.linalg.norm(embeddings, axis=1)

normalization_status = np.allclose(norms, 1.0, atol=1e-5)

if normalization_status:
    normalization_text = "Already L2-normalized"
else:
    normalization_text = "NOT L2-normalized"

# ============================================================
# NORMALIZE FOR COSINE CALCULATION
# ============================================================

# We normalize here regardless of the original state so that
# the calculation is explicitly cosine similarity.
normalized = embeddings / norms[:, np.newaxis]

# ============================================================
# PAIRWISE COSINE SIMILARITY
# ============================================================

# Full cosine similarity matrix
cosine_matrix = normalized @ normalized.T

# Extract ONLY the unique pairs.
# The diagonal and duplicate pairs are excluded.
upper_triangle = np.triu_indices(n_embeddings, k=1)

pairwise_cosines = cosine_matrix[upper_triangle]

expected_pairs = n_embeddings * (n_embeddings - 1) // 2

print(f"Unique pairs      : {expected_pairs:,}")
print(f"Normalization     : {normalization_text}")

# ============================================================
# COSINE DISTRIBUTION
# ============================================================

print("\n" + "=" * 100)
print("COSINE DISTRIBUTION")
print("-" * 100)

print(f"Mean              : {np.mean(pairwise_cosines):.6f}")
print(f"Median            : {np.median(pairwise_cosines):.6f}")
print(f"Std Deviation     : {np.std(pairwise_cosines):.6f}")
print(f"Minimum           : {np.min(pairwise_cosines):.6f}")
print(f"Maximum           : {np.max(pairwise_cosines):.6f}")

# ============================================================
# PERCENTILES
# ============================================================

percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

print("\nPercentiles")
print("-" * 100)

for p in percentiles:
    value = np.percentile(pairwise_cosines, p)
    print(f"P{p:02d}               : {value:.6f}")

# ============================================================
# HISTOGRAM
# ============================================================

print("\n" + "=" * 100)
print("HISTOGRAM — PAIRWISE COSINE SIMILARITY")
print("-" * 100)
print("[Histogram plotted below]")

plt.figure(figsize=(10, 6))

plt.hist(
    pairwise_cosines,
    bins=40
)

plt.xlabel("Cosine Similarity")
plt.ylabel("Number of Pairs")
plt.title("Global Pairwise Cosine Similarity Distribution")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)