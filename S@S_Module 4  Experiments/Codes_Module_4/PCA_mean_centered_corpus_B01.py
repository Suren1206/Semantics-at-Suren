from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA


# ======================================================================================
# PCA — GLOBAL GEOMETRY OF MEAN-CENTERED EMBEDDINGS
# ======================================================================================

print("=" * 100)
print("PCA — GLOBAL GEOMETRY OF MEAN-CENTERED EMBEDDINGS")
print("=" * 100)


# ======================================================================================
# PATH
# ======================================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MEAN_CENTERED_FILE = (
    BASE_DIR
    / "Output_Files"
    / "mean_centered_160x1024.npy"
)


# ======================================================================================
# LOAD EMBEDDINGS
# ======================================================================================

print()
print("Loading mean-centered embeddings:")
print(MEAN_CENTERED_FILE)

embeddings = np.load(MEAN_CENTERED_FILE).astype(np.float64)

print("Shape              :", embeddings.shape)


# ======================================================================================
# VALIDATION
# ======================================================================================

n_embeddings, n_dimensions = embeddings.shape

if n_embeddings != 161:
    raise ValueError(
        f"Expected 161 embeddings, but found {n_embeddings}"
    )

if n_dimensions != 1024:
    raise ValueError(
        f"Expected 1024 dimensions, but found {n_dimensions}"
    )


# ======================================================================================
# PCA
# ======================================================================================

pca = PCA(n_components=min(n_embeddings, n_dimensions))

pca.fit(embeddings)


explained_variance = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_

cumulative_variance = np.cumsum(explained_variance_ratio)


# ======================================================================================
# BASIC INFORMATION
# ======================================================================================

print()
print("=" * 100)
print("PCA INPUT")
print("=" * 100)

print()
print(f"Embeddings          : {n_embeddings}")
print(f"Original dimensions : {n_dimensions}")
print("Input               : Mean-centered embeddings")
print(f"Number of PCs       : {len(explained_variance)}")


# ======================================================================================
# PRINCIPAL COMPONENTS
# ======================================================================================

print()
print("=" * 100)
print("PRINCIPAL COMPONENTS")
print("-" * 100)

print(
    f"{'Component':<15}"
    f"{'Explained Variance':<25}"
    f"{'Explained Variance %':<25}"
)

print("-" * 100)

for i in range(len(explained_variance)):

    print(
        f"PC{i + 1:<13}"
        f"{explained_variance[i]:<25.8f}"
        f"{explained_variance_ratio[i] * 100:<25.6f}%"
    )


# ======================================================================================
# CUMULATIVE EXPLAINED VARIANCE
# ======================================================================================

print()
print("=" * 100)
print("CUMULATIVE EXPLAINED VARIANCE")
print("-" * 100)

print(
    f"{'Component':<15}"
    f"{'Cumulative Explained Variance %':<30}"
)

print("-" * 100)


# Always show these checkpoints
checkpoints = [1, 2, 3, 5, 10, 20, 50, 100, 150, 161]

for pc in checkpoints:

    if pc <= len(cumulative_variance):

        print(
            f"PC{pc:<13}"
            f"{cumulative_variance[pc - 1] * 100:<30.6f}%"
        )


# ======================================================================================
# ADDITIONAL SUMMARY
# ======================================================================================

print()
print("=" * 100)
print("PCA SUMMARY")
print("=" * 100)

print()
print(
    f"Variance explained by PC1   : "
    f"{explained_variance_ratio[0] * 100:.6f}%"
)

print(
    f"Variance explained by PC2   : "
    f"{explained_variance_ratio[1] * 100:.6f}%"
)

print(
    f"Variance explained by PC3   : "
    f"{explained_variance_ratio[2] * 100:.6f}%"
)

print(
    f"Cumulative variance PC10    : "
    f"{cumulative_variance[9] * 100:.6f}%"
)

print(
    f"Cumulative variance PC20    : "
    f"{cumulative_variance[19] * 100:.6f}%"
)

print(
    f"Cumulative variance PC50    : "
    f"{cumulative_variance[49] * 100:.6f}%"
)

print(
    f"Cumulative variance PC100   : "
    f"{cumulative_variance[99] * 100:.6f}%"
)

print(
    f"Cumulative variance PC161   : "
    f"{cumulative_variance[-1] * 100:.6f}%"
)


# ======================================================================================
# EXPERIMENT COMPLETE
# ======================================================================================

print()
print("=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)