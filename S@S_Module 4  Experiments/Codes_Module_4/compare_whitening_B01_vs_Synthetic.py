import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA


# ==========================================================================================
# Semantics @ Suren — Module 4
# EXPERIMENT — WHITENING
# B01 vs SYNTHETIC
# ==========================================================================================


# ==========================================================================================
# PATH CONFIGURATION
# ==========================================================================================

BASE_DIR = Path(
    "/home/surendran/LangChain/Semantics@Suren/"
    "S@S_Module 4  Experiments"
)

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

OUTPUT_DIR = BASE_DIR / "Output_Files"

B01_WHITENED_FILE = (
    OUTPUT_DIR
    / "B01_whitened_embeddings.npy"
)

SYNTHETIC_WHITENED_FILE = (
    OUTPUT_DIR
    / "Synthetic_whitened_embeddings.npy"
)


# ==========================================================================================
# WHITENING CONFIGURATION
# ==========================================================================================

N_COMPONENTS = 20


# ==========================================================================================
# HEADER
# ==========================================================================================

print("=" * 100)
print("Semantics @ Suren — Module 4")
print("WHITENING — B01 vs SYNTHETIC")
print("=" * 100)

print()
print("Experiment sequence:")
print("  Mean-centered embeddings")
print("          ↓")
print("         PCA")
print("          ↓")
print("      Whitening")
print("          ↓")
print("  Before vs After comparison")

print()
print(f"PCA components used for both datasets : {N_COMPONENTS}")


# ==========================================================================================
# LOAD EMBEDDINGS
# ==========================================================================================

print()
print("=" * 100)
print("LOADING MEAN-CENTERED EMBEDDINGS")
print("=" * 100)

print()
print(f"B01       : {B01_CENTERED_FILE}")
print(f"Synthetic : {SYNTHETIC_CENTERED_FILE}")


if not B01_CENTERED_FILE.exists():
    raise FileNotFoundError(
        f"\nB01 mean-centered file was not found:\n"
        f"{B01_CENTERED_FILE}"
    )

if not SYNTHETIC_CENTERED_FILE.exists():
    raise FileNotFoundError(
        f"\nSynthetic mean-centered file was not found:\n"
        f"{SYNTHETIC_CENTERED_FILE}"
    )


b01_centered = np.load(
    B01_CENTERED_FILE
).astype(np.float64)

synthetic_centered = np.load(
    SYNTHETIC_CENTERED_FILE
).astype(np.float64)


print()
print(f"B01 shape       : {b01_centered.shape}")
print(f"Synthetic shape : {synthetic_centered.shape}")


# ==========================================================================================
# VALIDATION
# ==========================================================================================

def validate_embeddings(embeddings, name):

    if embeddings.ndim != 2:
        raise ValueError(
            f"{name} embeddings must be a 2-dimensional matrix."
        )

    if not np.all(np.isfinite(embeddings)):
        raise ValueError(
            f"{name} embeddings contain NaN or infinite values."
        )


validate_embeddings(
    b01_centered,
    "B01"
)

validate_embeddings(
    synthetic_centered,
    "Synthetic"
)


if b01_centered.shape[1] != synthetic_centered.shape[1]:

    raise ValueError(
        "\nB01 and Synthetic have different embedding dimensions."
    )


if N_COMPONENTS > min(
    b01_centered.shape[0],
    synthetic_centered.shape[0]
):

    raise ValueError(
        "\nN_COMPONENTS is larger than the number of available "
        "observations in one of the datasets."
    )


# ==========================================================================================
# PCA + WHITENING
# ==========================================================================================

def pca_and_whiten(embeddings):

    pca = PCA(
        n_components=N_COMPONENTS,
        whiten=True
    )

    whitened = pca.fit_transform(
        embeddings
    )

    return pca, whitened


# ==========================================================================================
# RUN PCA + WHITENING
# ==========================================================================================

print()
print("=" * 100)
print("RUNNING PCA + WHITENING")
print("=" * 100)

print()

print("B01...")
b01_pca, b01_whitened = pca_and_whiten(
    b01_centered
)

print("Synthetic...")
synthetic_pca, synthetic_whitened = pca_and_whiten(
    synthetic_centered
)


# ==========================================================================================
# SAVE WHITENED EMBEDDINGS
# ==========================================================================================

np.save(
    B01_WHITENED_FILE,
    b01_whitened
)

np.save(
    SYNTHETIC_WHITENED_FILE,
    synthetic_whitened
)

print()
print("Whitened embeddings saved:")
print(f"  B01       : {B01_WHITENED_FILE}")
print(f"  Synthetic : {SYNTHETIC_WHITENED_FILE}")


# ==========================================================================================
# PCA INFORMATION
# ==========================================================================================

print()
print("=" * 100)
print("PCA INFORMATION")
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
    f"{b01_centered.shape[0]:>20}"
    f"{synthetic_centered.shape[0]:>20}"
)

print(
    f"{'Original dimensions':<40}"
    f"{b01_centered.shape[1]:>20}"
    f"{synthetic_centered.shape[1]:>20}"
)

print(
    f"{'PCA components':<40}"
    f"{N_COMPONENTS:>20}"
    f"{N_COMPONENTS:>20}"
)


# ==========================================================================================
# PCA EXPLAINED VARIANCE
# ==========================================================================================

print()
print("=" * 100)
print("PCA — EXPLAINED VARIANCE BEFORE WHITENING")
print("=" * 100)

print()

print(
    f"{'Component':<15}"
    f"{'B01 Variance %':>20}"
    f"{'B01 Cumulative %':>22}"
    f"{'Synthetic Variance %':>25}"
    f"{'Synthetic Cumulative %':>27}"
)

print("-" * 110)

b01_cumulative = (
    np.cumsum(
        b01_pca.explained_variance_ratio_
    ) * 100
)

synthetic_cumulative = (
    np.cumsum(
        synthetic_pca.explained_variance_ratio_
    ) * 100
)


for i in range(N_COMPONENTS):

    b01_pct = (
        b01_pca.explained_variance_ratio_[i]
        * 100
    )

    synthetic_pct = (
        synthetic_pca.explained_variance_ratio_[i]
        * 100
    )

    print(
        f"PC{i + 1:<12}"
        f"{b01_pct:>19.6f}%"
        f"{b01_cumulative[i]:>21.6f}%"
        f"{synthetic_pct:>24.6f}%"
        f"{synthetic_cumulative[i]:>26.6f}%"
    )


# ==========================================================================================
# VECTOR NORM ANALYSIS
# ==========================================================================================

def norm_statistics(embeddings):

    norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    return {
        "mean": np.mean(norms),
        "median": np.median(norms),
        "std": np.std(norms),
        "min": np.min(norms),
        "max": np.max(norms)
    }


b01_before_norms = norm_statistics(
    b01_centered
)

b01_after_norms = norm_statistics(
    b01_whitened
)

synthetic_before_norms = norm_statistics(
    synthetic_centered
)

synthetic_after_norms = norm_statistics(
    synthetic_whitened
)


# ==========================================================================================
# NORM COMPARISON
# ==========================================================================================

print()
print("=" * 100)
print("VECTOR NORM COMPARISON")
print("=" * 100)

print()

print(
    f"{'Statistic':<20}"
    f"{'B01 Before':>20}"
    f"{'B01 After':>20}"
    f"{'Synthetic Before':>22}"
    f"{'Synthetic After':>22}"
)

print("-" * 105)

for label, key in [
    ("Mean", "mean"),
    ("Median", "median"),
    ("Std Deviation", "std"),
    ("Minimum", "min"),
    ("Maximum", "max")
]:

    print(
        f"{label:<20}"
        f"{b01_before_norms[key]:>20.6f}"
        f"{b01_after_norms[key]:>20.6f}"
        f"{synthetic_before_norms[key]:>22.6f}"
        f"{synthetic_after_norms[key]:>22.6f}"
    )


# ==========================================================================================
# PAIRWISE COSINE
# ==========================================================================================

def pairwise_cosine_values(embeddings):

    norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    normalized = (
        embeddings /
        norms[:, np.newaxis]
    )

    similarity_matrix = (
        normalized @ normalized.T
    )

    upper_triangle = np.triu_indices(
        len(embeddings),
        k=1
    )

    return similarity_matrix[
        upper_triangle
    ]


print()
print("=" * 100)
print("CALCULATING PAIRWISE COSINE DISTRIBUTIONS")
print("=" * 100)

print()

print("B01 — Before whitening...")
b01_cosine_before = pairwise_cosine_values(
    b01_centered
)

print("B01 — After whitening...")
b01_cosine_after = pairwise_cosine_values(
    b01_whitened
)

print("Synthetic — Before whitening...")
synthetic_cosine_before = pairwise_cosine_values(
    synthetic_centered
)

print("Synthetic — After whitening...")
synthetic_cosine_after = pairwise_cosine_values(
    synthetic_whitened
)


# ==========================================================================================
# COSINE STATISTICS
# ==========================================================================================

def cosine_statistics(values):

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
        "p99": np.percentile(values, 99)
    }


b01_before = cosine_statistics(
    b01_cosine_before
)

b01_after = cosine_statistics(
    b01_cosine_after
)

synthetic_before = cosine_statistics(
    synthetic_cosine_before
)

synthetic_after = cosine_statistics(
    synthetic_cosine_after
)


# ==========================================================================================
# PAIRWISE COSINE COMPARISON
# ==========================================================================================

print()
print("=" * 100)
print("PAIRWISE COSINE — BEFORE vs AFTER WHITENING")
print("=" * 100)

print()

print(
    f"{'Statistic':<20}"
    f"{'B01 Before':>20}"
    f"{'B01 After':>20}"
    f"{'Synthetic Before':>22}"
    f"{'Synthetic After':>22}"
)

print("-" * 105)

for label, key in [
    ("Mean", "mean"),
    ("Median", "median"),
    ("Std Deviation", "std"),
    ("Minimum", "min"),
    ("Maximum", "max")
]:

    print(
        f"{label:<20}"
        f"{b01_before[key]:>20.6f}"
        f"{b01_after[key]:>20.6f}"
        f"{synthetic_before[key]:>22.6f}"
        f"{synthetic_after[key]:>22.6f}"
    )


# ==========================================================================================
# PERCENTILES
# ==========================================================================================

print()
print("=" * 100)
print("PAIRWISE COSINE — PERCENTILES")
print("=" * 100)

print()

print(
    f"{'Percentile':<15}"
    f"{'B01 Before':>20}"
    f"{'B01 After':>20}"
    f"{'Synthetic Before':>22}"
    f"{'Synthetic After':>22}"
)

print("-" * 105)

for label, key in [
    ("P01", "p01"),
    ("P05", "p05"),
    ("P10", "p10"),
    ("P25", "p25"),
    ("P50", "p50"),
    ("P75", "p75"),
    ("P90", "p90"),
    ("P95", "p95"),
    ("P99", "p99")
]:

    print(
        f"{label:<15}"
        f"{b01_before[key]:>20.6f}"
        f"{b01_after[key]:>20.6f}"
        f"{synthetic_before[key]:>22.6f}"
        f"{synthetic_after[key]:>22.6f}"
    )


# ==========================================================================================
# HISTOGRAM — FOUR DISTRIBUTIONS
# ==========================================================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(15, 11)
)


axes[0, 0].hist(
    b01_cosine_before,
    bins=40,
    edgecolor="black"
)

axes[0, 0].set_title(
    "B01 — Before Whitening"
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


axes[0, 1].hist(
    b01_cosine_after,
    bins=40,
    edgecolor="black"
)

axes[0, 1].set_title(
    "B01 — After Whitening"
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


axes[1, 0].hist(
    synthetic_cosine_before,
    bins=40,
    edgecolor="black"
)

axes[1, 0].set_title(
    "Synthetic — Before Whitening"
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


axes[1, 1].hist(
    synthetic_cosine_after,
    bins=40,
    edgecolor="black"
)

axes[1, 1].set_title(
    "Synthetic — After Whitening"
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
    "B01 vs Synthetic — Pairwise Cosine Before and After Whitening",
    fontsize=14
)

plt.tight_layout()


# ==========================================================================================
# SAVE FIGURE
# ==========================================================================================

plot_file = (
    OUTPUT_DIR
    / "whitening_cosine_comparison_B01_vs_Synthetic.png"
)

plt.savefig(
    plot_file,
    dpi=200,
    bbox_inches="tight"
)

print()
print("Histogram saved to:")
print(f"  {plot_file}")

plt.show()


# ==========================================================================================
# EXPERIMENT COMPLETE
# ==========================================================================================

print()
print("=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)

print()
print("B01:")
print("  Mean-centered → PCA(20) → Whitening → Cosine comparison")

print()
print("Synthetic:")
print("  Mean-centered → PCA(20) → Whitening → Cosine comparison")

print()
print("Original embeddings were NOT modified.")

print("Whitened embeddings were saved as separate .npy files.")

print("=" * 100)