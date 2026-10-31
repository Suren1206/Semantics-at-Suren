import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA


# ==========================================================================================
# Semantics @ Suren — Module 4
# EXPERIMENT — PCA COMPARISON
# B01 vs SYNTHETIC CORPUS
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
# LOAD DATA
# ==========================================================================================

print("=" * 100)
print("Semantics @ Suren — Module 4")
print("PCA — B01 vs SYNTHETIC CORPUS")
print("=" * 100)

print("\nLoading B01 embeddings...")
b01 = np.load(B01_FILE).astype(np.float64)

print(f"B01 shape       : {b01.shape}")

print("\nLoading Synthetic embeddings...")
synthetic = np.load(SYNTHETIC_FILE).astype(np.float64)

print(f"Synthetic shape : {synthetic.shape}")


# ==========================================================================================
# VALIDATION
# ==========================================================================================

if b01.ndim != 2:
    raise ValueError(
        f"B01 embeddings must be 2-dimensional. Found {b01.ndim} dimensions."
    )

if synthetic.ndim != 2:
    raise ValueError(
        f"Synthetic embeddings must be 2-dimensional. Found {synthetic.ndim} dimensions."
    )

if b01.shape[1] != synthetic.shape[1]:
    raise ValueError(
        "\nThe two embedding populations have different dimensions.\n"
        f"B01 dimensions       : {b01.shape[1]}\n"
        f"Synthetic dimensions : {synthetic.shape[1]}"
    )


# ==========================================================================================
# PCA FUNCTION
# ==========================================================================================

def run_pca(embeddings):
    """
    Run PCA on the embedding population.

    sklearn PCA centers the input internally before calculating
    the principal components.
    """

    n_vectors, n_dimensions = embeddings.shape

    n_components = min(n_vectors, n_dimensions)

    pca = PCA(
        n_components=n_components
    )

    transformed = pca.fit_transform(embeddings)

    return pca, transformed


# ==========================================================================================
# RUN PCA
# ==========================================================================================

print("\n" + "=" * 100)
print("RUNNING PCA")
print("=" * 100)

print("\nB01 PCA...")
b01_pca, b01_transformed = run_pca(b01)

print("Synthetic PCA...")
synthetic_pca, synthetic_transformed = run_pca(synthetic)


# ==========================================================================================
# BASIC PCA INFORMATION
# ==========================================================================================

print("\n" + "=" * 100)
print("PCA DIMENSIONS")
print("=" * 100)

print()
print(f"{'Measure':<40}{'B01':>20}{'Synthetic':>20}")
print("-" * 80)

print(
    f"{'Number of embeddings':<40}"
    f"{b01.shape[0]:>20}"
    f"{synthetic.shape[0]:>20}"
)

print(
    f"{'Original dimensions':<40}"
    f"{b01.shape[1]:>20}"
    f"{synthetic.shape[1]:>20}"
)

print(
    f"{'Number of principal components':<40}"
    f"{len(b01_pca.explained_variance_):>20}"
    f"{len(synthetic_pca.explained_variance_):>20}"
)


# ==========================================================================================
# PRINCIPAL COMPONENTS
# ==========================================================================================

print("\n" + "=" * 100)
print("PRINCIPAL COMPONENTS")
print("=" * 100)

print()
print(
    f"{'Component':<15}"
    f"{'B01 Explained Variance':>25}"
    f"{'B01 Variance %':>20}"
    f"{'Synthetic Explained Variance':>30}"
    f"{'Synthetic Variance %':>22}"
)

print("-" * 115)

max_display = min(
    10,
    len(b01_pca.explained_variance_),
    len(synthetic_pca.explained_variance_)
)

for i in range(max_display):

    b01_variance = b01_pca.explained_variance_[i]
    b01_percentage = b01_pca.explained_variance_ratio_[i] * 100

    synthetic_variance = synthetic_pca.explained_variance_[i]
    synthetic_percentage = (
        synthetic_pca.explained_variance_ratio_[i] * 100
    )

    print(
        f"PC{i + 1:<12}"
        f"{b01_variance:>25.8f}"
        f"{b01_percentage:>19.6f}%"
        f"{synthetic_variance:>30.8f}"
        f"{synthetic_percentage:>21.6f}%"
    )


# ==========================================================================================
# CUMULATIVE EXPLAINED VARIANCE
# ==========================================================================================

b01_cumulative = np.cumsum(
    b01_pca.explained_variance_ratio_
) * 100

synthetic_cumulative = np.cumsum(
    synthetic_pca.explained_variance_ratio_
) * 100


print("\n" + "=" * 100)
print("CUMULATIVE EXPLAINED VARIANCE")
print("=" * 100)

print()
print(
    f"{'Component':<15}"
    f"{'B01':>25}"
    f"{'Synthetic':>25}"
)

print("-" * 65)

checkpoints = [
    1,
    2,
    3,
    5,
    10,
    20,
    50,
    100
]

for component in checkpoints:

    b01_index = component - 1
    synthetic_index = component - 1

    if (
        b01_index < len(b01_cumulative)
        and synthetic_index < len(synthetic_cumulative)
    ):

        print(
            f"PC{component:<12}"
            f"{b01_cumulative[b01_index]:>24.6f}%"
            f"{synthetic_cumulative[synthetic_index]:>24.6f}%"
        )


# ==========================================================================================
# NUMBER OF COMPONENTS REQUIRED FOR VARIANCE TARGETS
# ==========================================================================================

print("\n" + "=" * 100)
print("COMPONENTS REQUIRED TO REACH VARIANCE TARGETS")
print("=" * 100)

print()
print(
    f"{'Variance Target':<25}"
    f"{'B01 Components':>20}"
    f"{'Synthetic Components':>25}"
)

print("-" * 75)

targets = [50, 75, 80, 90, 95, 99]

for target in targets:

    b01_components = (
        np.argmax(b01_cumulative >= target) + 1
    )

    synthetic_components = (
        np.argmax(synthetic_cumulative >= target) + 1
    )

    print(
        f"{target}%{'':<20}"
        f"{b01_components:>20}"
        f"{synthetic_components:>25}"
    )


# ==========================================================================================
# TOP COMPONENT DOMINANCE
# ==========================================================================================

print("\n" + "=" * 100)
print("TOP COMPONENT DOMINANCE")
print("=" * 100)

print()
print(
    f"{'Measure':<40}"
    f"{'B01':>20}"
    f"{'Synthetic':>20}"
)

print("-" * 80)

print(
    f"{'PC1 variance %':<40}"
    f"{b01_pca.explained_variance_ratio_[0] * 100:>19.6f}%"
    f"{synthetic_pca.explained_variance_ratio_[0] * 100:>19.6f}%"
)

if len(b01_cumulative) >= 3 and len(synthetic_cumulative) >= 3:

    print(
        f"{'PC1-PC3 cumulative %':<40}"
        f"{b01_cumulative[2]:>19.6f}%"
        f"{synthetic_cumulative[2]:>19.6f}%"
    )

if len(b01_cumulative) >= 10 and len(synthetic_cumulative) >= 10:

    print(
        f"{'PC1-PC10 cumulative %':<40}"
        f"{b01_cumulative[9]:>19.6f}%"
        f"{synthetic_cumulative[9]:>19.6f}%"
    )


# ==========================================================================================
# SIDE-BY-SIDE EXPLAINED VARIANCE PLOT
# ==========================================================================================

max_plot_components = min(
    20,
    len(b01_pca.explained_variance_ratio_),
    len(synthetic_pca.explained_variance_ratio_)
)

components = np.arange(
    1,
    max_plot_components + 1
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 6)
)


# ------------------------------------------------------------------------------------------
# Individual explained variance
# ------------------------------------------------------------------------------------------

axes[0].plot(
    components,
    b01_pca.explained_variance_ratio_[:max_plot_components] * 100,
    marker="o",
    label="B01"
)

axes[0].plot(
    components,
    synthetic_pca.explained_variance_ratio_[:max_plot_components] * 100,
    marker="o",
    label="Synthetic"
)

axes[0].set_title(
    "Individual Explained Variance"
)

axes[0].set_xlabel(
    "Principal Component"
)

axes[0].set_ylabel(
    "Explained Variance (%)"
)

axes[0].set_xticks(components)

axes[0].grid(
    True,
    alpha=0.3
)

axes[0].legend()


# ------------------------------------------------------------------------------------------
# Cumulative explained variance
# ------------------------------------------------------------------------------------------

axes[1].plot(
    components,
    b01_cumulative[:max_plot_components],
    marker="o",
    label="B01"
)

axes[1].plot(
    components,
    synthetic_cumulative[:max_plot_components],
    marker="o",
    label="Synthetic"
)

axes[1].set_title(
    "Cumulative Explained Variance"
)

axes[1].set_xlabel(
    "Principal Component"
)

axes[1].set_ylabel(
    "Cumulative Explained Variance (%)"
)

axes[1].set_xticks(components)

axes[1].grid(
    True,
    alpha=0.3
)

axes[1].legend()


plt.suptitle(
    "B01 vs Synthetic — PCA Comparison",
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

print("\nPCA was calculated independently for B01 and Synthetic.")

print("\nNote:")
print("sklearn PCA centers each input population internally before")
print("computing the principal components.")

print("\nThe original embedding files were NOT modified.")
print("=" * 100)