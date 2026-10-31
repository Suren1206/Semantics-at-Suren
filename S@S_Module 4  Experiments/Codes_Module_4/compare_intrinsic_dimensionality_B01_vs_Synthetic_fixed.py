import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr


# ==========================================================================================
# Semantics @ Suren — Module 4
# EXPERIMENT — INTRINSIC DIMENSIONALITY IMPACT
# B01 vs SYNTHETIC
#
# Purpose:
#   Determine how much dimensionality reduction is required to retain 80% variance
#   and measure what happens to the useful geometry after that reduction.
#
# IMPORTANT:
#   - Input embeddings are assumed to be mean-centered.
#   - No additional mean-centering is performed.
#   - No whitening is performed.
#   - Original embeddings are NOT modified.
#   - Reduced embeddings are created separately.
# ==========================================================================================


# ==========================================================================================
# PATH CONFIGURATION
# ==========================================================================================

BASE_DIR = Path(
    "/home/surendran/LangChain/Semantics@Suren/"
    "S@S_Module 4  Experiments"
)

# ------------------------------------------------------------------
# Mean-centered embeddings
# ------------------------------------------------------------------

B01_FILE = (
    BASE_DIR
    / "Output_Files"
    / "mean_centered_B01.npy"
)

SYNTHETIC_FILE = (
    BASE_DIR
    / "Output_Files"
    / "mean_centered_Synthetic.npy"
)

# ------------------------------------------------------------------
# Output directory
# ------------------------------------------------------------------

OUTPUT_DIR = BASE_DIR / "Output_Files"

B01_REDUCED_FILE = (
    OUTPUT_DIR
    / "B01_80pct_variance_reduced.npy"
)

SYNTHETIC_REDUCED_FILE = (
    OUTPUT_DIR
    / "Synthetic_80pct_variance_reduced.npy"
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "intrinsic_dimensionality_B01_vs_Synthetic.txt"
)

PLOT_FILE = (
    OUTPUT_DIR
    / "intrinsic_dimensionality_cosine_B01_vs_Synthetic.png"
)


# ==========================================================================================
# PARAMETERS
# ==========================================================================================

TARGET_VARIANCE = 0.80

TOP_K = 5


# ==========================================================================================
# HEADER
# ==========================================================================================

print("=" * 100)
print("Semantics @ Suren — Module 4")
print("INTRINSIC DIMENSIONALITY IMPACT — B01 vs SYNTHETIC")
print("=" * 100)

print()
print("Target variance retention :", f"{TARGET_VARIANCE * 100:.0f}%")
print("Nearest neighbours        :", f"Top-{TOP_K}")

print()
print("Experiment:")
print("  Mean-centered 1024-D embeddings")
print("              ↓")
print("             PCA")
print("              ↓")
print("  Minimum PCs required for 80% variance")
print("              ↓")
print("  Compare original vs reduced geometry")


# ==========================================================================================
# LOAD DATA
# ==========================================================================================

def load_embeddings(path, name):

    print()
    print("=" * 100)
    print(f"LOADING {name.upper()} EMBEDDINGS")
    print("=" * 100)

    print()
    print(f"File : {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"\n{name} embedding file was not found:\n{path}\n\n"
            f"Please check the filename and location."
        )

    embeddings = np.load(path).astype(np.float64)

    if embeddings.ndim != 2:
        raise ValueError(
            f"{name} embeddings must be a 2-D matrix."
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
# BASIC VALIDATION
# ==========================================================================================

if b01.shape[1] != synthetic.shape[1]:

    raise ValueError(
        "\nB01 and Synthetic have different embedding dimensions.\n"
        f"B01       : {b01.shape[1]}\n"
        f"Synthetic : {synthetic.shape[1]}"
    )


# ==========================================================================================
# PCA TO 80% VARIANCE
# ==========================================================================================

def reduce_to_target_variance(embeddings, target=0.80):

    n_vectors, n_dimensions = embeddings.shape

    # PCA cannot have more components than:
    # min(number of vectors, number of dimensions)
    max_components = min(
        n_vectors,
        n_dimensions
    )

    # First calculate the full PCA spectrum.
    pca_full = PCA(
        n_components=max_components
    )

    pca_full.fit(embeddings)

    explained_ratio = (
        pca_full.explained_variance_ratio_
    )

    cumulative_ratio = np.cumsum(
        explained_ratio
    )

    # Find minimum number of PCs reaching target variance.
    components_required = (
        np.searchsorted(
            cumulative_ratio,
            target
        ) + 1
    )

    # Run PCA again using exactly the required number.
    pca_reduced = PCA(
        n_components=components_required
    )

    reduced = pca_reduced.fit_transform(
        embeddings
    )

    retained_variance = np.sum(
        pca_reduced.explained_variance_ratio_
    )

    return (
        pca_full,
        pca_reduced,
        reduced,
        components_required,
        retained_variance
    )


print()
print("=" * 100)
print("FINDING DIMENSIONALITY REQUIRED FOR 80% VARIANCE")
print("=" * 100)

print()

(
    b01_pca_full,
    b01_pca_reduced,
    b01_reduced,
    b01_components,
    b01_retained
) = reduce_to_target_variance(
    b01,
    TARGET_VARIANCE
)

(
    synthetic_pca_full,
    synthetic_pca_reduced,
    synthetic_reduced,
    synthetic_components,
    synthetic_retained
) = reduce_to_target_variance(
    synthetic,
    TARGET_VARIANCE
)


# ==========================================================================================
# DIMENSIONALITY SUMMARY
# ==========================================================================================

print()
print("=" * 100)
print("DIMENSIONALITY REDUCTION")
print("=" * 100)

print()

print(
    f"{'Measure':<35}"
    f"{'B01':>25}"
    f"{'Synthetic':>25}"
)

print("-" * 85)

print(
    f"{'Original dimensions':<35}"
    f"{b01.shape[1]:>25}"
    f"{synthetic.shape[1]:>25}"
)

print(
    f"{'PCs required for 80% variance':<35}"
    f"{b01_components:>25}"
    f"{synthetic_components:>25}"
)

print(
    f"{'Actual variance retained':<35}"
    f"{b01_retained * 100:>24.6f}%"
    f"{synthetic_retained * 100:>24.6f}%"
)

print(
    f"{'Dimensions removed':<35}"
    f"{b01.shape[1] - b01_components:>25}"
    f"{synthetic.shape[1] - synthetic_components:>25}"
)

print(
    f"{'Compression ratio':<35}"
    f"{b01.shape[1] / b01_components:>24.2f}x"
    f"{synthetic.shape[1] / synthetic_components:>24.2f}x"
)


# ==========================================================================================
# PCA VARIANCE CHECKPOINTS
# ==========================================================================================

def variance_at_component(pca, component):

    if component > len(pca.explained_variance_ratio_):
        return np.nan

    return (
        np.sum(
            pca.explained_variance_ratio_[:component]
        ) * 100
    )


print()
print("=" * 100)
print("CUMULATIVE EXPLAINED VARIANCE")
print("=" * 100)

print()

print(
    f"{'PC':<10}"
    f"{'B01':>25}"
    f"{'Synthetic':>25}"
)

print("-" * 65)

for pc in [1, 2, 3, 5, 10, 20, 24, 49, 50, 100]:

    b01_value = variance_at_component(
        b01_pca_full,
        pc
    )

    synthetic_value = variance_at_component(
        synthetic_pca_full,
        pc
    )

    b01_text = (
        f"{b01_value:.6f}%"
        if not np.isnan(b01_value)
        else "N/A"
    )

    synthetic_text = (
        f"{synthetic_value:.6f}%"
        if not np.isnan(synthetic_value)
        else "N/A"
    )

    print(
        f"PC{pc:<8}"
        f"{b01_text:>25}"
        f"{synthetic_text:>25}"
    )


# ==========================================================================================
# SAVE REDUCED EMBEDDINGS
# ==========================================================================================

np.save(
    B01_REDUCED_FILE,
    b01_reduced
)

np.save(
    SYNTHETIC_REDUCED_FILE,
    synthetic_reduced
)

print()
print("=" * 100)
print("REDUCED EMBEDDINGS SAVED")
print("=" * 100)

print()
print(f"B01       : {B01_REDUCED_FILE}")
print(f"Synthetic : {SYNTHETIC_REDUCED_FILE}")


# ==========================================================================================
# COSINE SIMILARITY
# ==========================================================================================

def pairwise_cosine_values(embeddings):

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    normalized = (
        embeddings / norms
    )

    cosine_matrix = (
        normalized @ normalized.T
    )

    upper = np.triu_indices(
        embeddings.shape[0],
        k=1
    )

    return cosine_matrix[upper]


print()
print("=" * 100)
print("PAIRWISE COSINE ANALYSIS")
print("=" * 100)

print()

b01_cosine_original = pairwise_cosine_values(
    b01
)

b01_cosine_reduced = pairwise_cosine_values(
    b01_reduced
)

synthetic_cosine_original = pairwise_cosine_values(
    synthetic
)

synthetic_cosine_reduced = pairwise_cosine_values(
    synthetic_reduced
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
        "p95": np.percentile(values, 95),
        "p99": np.percentile(values, 99)
    }


b01_cos_original = cosine_statistics(
    b01_cosine_original
)

b01_cos_reduced = cosine_statistics(
    b01_cosine_reduced
)

synthetic_cos_original = cosine_statistics(
    synthetic_cosine_original
)

synthetic_cos_reduced = cosine_statistics(
    synthetic_cosine_reduced
)


# ==========================================================================================
# COSINE COMPARISON
# ==========================================================================================

print()
print("=" * 100)
print("PAIRWISE COSINE — ORIGINAL vs REDUCED")
print("=" * 100)

print()

print(
    f"{'Statistic':<20}"
    f"{'B01 Original':>20}"
    f"{'B01 Reduced':>20}"
    f"{'Synthetic Original':>23}"
    f"{'Synthetic Reduced':>23}"
)

print("-" * 110)

for label, key in [
    ("Mean", "mean"),
    ("Median", "median"),
    ("Std Deviation", "std"),
    ("Minimum", "min"),
    ("Maximum", "max"),
    ("P95", "p95"),
    ("P99", "p99")
]:

    print(
        f"{label:<20}"
        f"{b01_cos_original[key]:>20.6f}"
        f"{b01_cos_reduced[key]:>20.6f}"
        f"{synthetic_cos_original[key]:>23.6f}"
        f"{synthetic_cos_reduced[key]:>23.6f}"
    )


# ==========================================================================================
# NEAREST NEIGHBOUR PRESERVATION
# ==========================================================================================

def nearest_neighbour_indices(embeddings):

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    normalized = embeddings / norms

    similarity = (
        normalized @ normalized.T
    )

    np.fill_diagonal(
        similarity,
        -np.inf
    )

    return np.argmax(
        similarity,
        axis=1
    )


def top_k_neighbour_indices(
    embeddings,
    k=5
):

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    normalized = embeddings / norms

    similarity = (
        normalized @ normalized.T
    )

    np.fill_diagonal(
        similarity,
        -np.inf
    )

    neighbours = np.argpartition(
        -similarity,
        kth=k - 1,
        axis=1
    )[:, :k]

    rows = np.arange(
        len(embeddings)
    )[:, None]

    scores = similarity[
        rows,
        neighbours
    ]

    order = np.argsort(
        -scores,
        axis=1
    )

    neighbours = np.take_along_axis(
        neighbours,
        order,
        axis=1
    )

    return neighbours


# ------------------------------------------------------------------
# Top-1 preservation
# ------------------------------------------------------------------

b01_nn_original = nearest_neighbour_indices(
    b01
)

b01_nn_reduced = nearest_neighbour_indices(
    b01_reduced
)

synthetic_nn_original = nearest_neighbour_indices(
    synthetic
)

synthetic_nn_reduced = nearest_neighbour_indices(
    synthetic_reduced
)


b01_top1_preserved = np.mean(
    b01_nn_original == b01_nn_reduced
) * 100

synthetic_top1_preserved = np.mean(
    synthetic_nn_original == synthetic_nn_reduced
) * 100


# ------------------------------------------------------------------
# Top-5 overlap
# ------------------------------------------------------------------

b01_top5_original = top_k_neighbour_indices(
    b01,
    TOP_K
)

b01_top5_reduced = top_k_neighbour_indices(
    b01_reduced,
    TOP_K
)

synthetic_top5_original = top_k_neighbour_indices(
    synthetic,
    TOP_K
)

synthetic_top5_reduced = top_k_neighbour_indices(
    synthetic_reduced,
    TOP_K
)


def average_top_k_overlap(
    original,
    reduced
):

    overlaps = []

    for i in range(len(original)):

        original_set = set(
            original[i]
        )

        reduced_set = set(
            reduced[i]
        )

        overlap = len(
            original_set.intersection(
                reduced_set
            )
        )

        overlaps.append(
            overlap
        )

    return np.mean(overlaps)


b01_top5_overlap = average_top_k_overlap(
    b01_top5_original,
    b01_top5_reduced
)

synthetic_top5_overlap = average_top_k_overlap(
    synthetic_top5_original,
    synthetic_top5_reduced
)


# ==========================================================================================
# RANKING / SIMILARITY PRESERVATION
# ==========================================================================================

b01_pearson = pearsonr(
    b01_cosine_original,
    b01_cosine_reduced
)

synthetic_pearson = pearsonr(
    synthetic_cosine_original,
    synthetic_cosine_reduced
)

b01_spearman = spearmanr(
    b01_cosine_original,
    b01_cosine_reduced
)

synthetic_spearman = spearmanr(
    synthetic_cosine_original,
    synthetic_cosine_reduced
)


# ==========================================================================================
# PRESERVATION SUMMARY
# ==========================================================================================

print()
print("=" * 100)
print("NEAREST-NEIGHBOUR PRESERVATION")
print("=" * 100)

print()

print(
    f"{'Measure':<35}"
    f"{'B01':>25}"
    f"{'Synthetic':>25}"
)

print("-" * 85)

print(
    f"{'Top-1 neighbour preserved':<35}"
    f"{b01_top1_preserved:>24.2f}%"
    f"{synthetic_top1_preserved:>24.2f}%"
)

print(
    f"{'Average Top-5 overlap':<35}"
    f"{b01_top5_overlap:>24.2f} / 5"
    f"{synthetic_top5_overlap:>24.2f} / 5"
)


print()
print("=" * 100)
print("SIMILARITY / RANKING PRESERVATION")
print("=" * 100)

print()

print(
    f"{'Measure':<35}"
    f"{'B01':>25}"
    f"{'Synthetic':>25}"
)

print("-" * 85)

print(
    f"{'Pearson correlation':<35}"
    f"{b01_pearson.statistic:>25.6f}"
    f"{synthetic_pearson.statistic:>25.6f}"
)

print(
    f"{'Spearman correlation':<35}"
    f"{b01_spearman.statistic:>25.6f}"
    f"{synthetic_spearman.statistic:>25.6f}"
)


# ==========================================================================================
# SYNTHETIC GROUP ANALYSIS
# ==========================================================================================
#
# We need the original corpus to determine which vectors belong to which group.
# This section loads the ODS corpus used for the Synthetic embeddings.
#
# It is metadata only.
# Column A is used for grouping.
# No metadata is embedded.
# ==========================================================================================

SYNTHETIC_CORPUS_FILE = (
    BASE_DIR
    / "Synthetic_Corpus.ods"
)


def load_synthetic_groups():

    if not SYNTHETIC_CORPUS_FILE.exists():

        print()
        print(
            "Synthetic corpus CSV was not found."
        )

        print(
            f"Expected:\n{SYNTHETIC_CORPUS_FILE}"
        )

        print(
            "Skipping Synthetic group analysis."
        )

        return None

    import pandas as pd

    # The ODS contains one header row.
    # Read it as a header so that only the 50 actual corpus rows
    # are aligned with the 50 embedding vectors.
    df = pd.read_excel(
        SYNTHETIC_CORPUS_FILE,
        engine="odf",
        header=0
    )

    if len(df) != len(synthetic):

        raise ValueError(
            "\nSynthetic corpus and embeddings are not aligned.\n"
            f"Corpus data rows : {len(df)}\n"
            f"Embedding rows   : {len(synthetic)}"
        )

    # Column A contains the group name.
    groups = (
        df.iloc[:, 0]
        .astype(str)
        .str.strip()
    )

    return groups


synthetic_groups = load_synthetic_groups()


# ==========================================================================================
# GROUP-WISE TOP-1 PRESERVATION
# ==========================================================================================

if synthetic_groups is not None:

    print()
    print("=" * 100)
    print("SYNTHETIC — GROUP-WISE NEAREST-NEIGHBOUR PRESERVATION")
    print("=" * 100)

    print()

    unique_groups = synthetic_groups.unique()

    print(
        f"{'Group':<35}"
        f"{'Vectors':>10}"
        f"{'Top-1 Preserved':>22}"
        f"{'Top-5 Overlap':>22}"
    )

    print("-" * 95)

    for group in unique_groups:

        indices = np.where(
            synthetic_groups.values == group
        )[0]

        if len(indices) < 2:

            print(
                f"{group:<35}"
                f"{len(indices):>10}"
                f"{'N/A':>22}"
                f"{'N/A':>22}"
            )

            continue

        # Top-1 preservation within the group's queries
        top1_original = (
            synthetic_nn_original[indices]
        )

        top1_reduced = (
            synthetic_nn_reduced[indices]
        )

        top1_percentage = np.mean(
            top1_original == top1_reduced
        ) * 100

        # Top-5 overlap
        overlaps = []

        for idx in indices:

            original_set = set(
                synthetic_top5_original[idx]
            )

            reduced_set = set(
                synthetic_top5_reduced[idx]
            )

            overlap = len(
                original_set.intersection(
                    reduced_set
                )
            )

            overlaps.append(
                overlap
            )

        mean_overlap = np.mean(
            overlaps
        )

        print(
            f"{group:<35}"
            f"{len(indices):>10}"
            f"{top1_percentage:>21.2f}%"
            f"{mean_overlap:>21.2f} / 5"
        )


# ==========================================================================================
# HISTOGRAM
# ==========================================================================================

print()
print("=" * 100)
print("GENERATING COSINE DISTRIBUTION PLOT")
print("=" * 100)


fig, axes = plt.subplots(
    2,
    2,
    figsize=(15, 11)
)


# B01 Original
axes[0, 0].hist(
    b01_cosine_original,
    bins=40,
    edgecolor="black"
)

axes[0, 0].set_title(
    "B01 — Original 1024-D"
)

axes[0, 0].set_xlabel(
    "Pairwise Cosine Similarity"
)

axes[0, 0].set_ylabel(
    "Number of Pairs"
)

axes[0, 0].grid(
    axis="y",
    alpha=0.3
)


# B01 Reduced
axes[0, 1].hist(
    b01_cosine_reduced,
    bins=40,
    edgecolor="black"
)

axes[0, 1].set_title(
    f"B01 — Reduced to {b01_components} PCs"
)

axes[0, 1].set_xlabel(
    "Pairwise Cosine Similarity"
)

axes[0, 1].set_ylabel(
    "Number of Pairs"
)

axes[0, 1].grid(
    axis="y",
    alpha=0.3
)


# Synthetic Original
axes[1, 0].hist(
    synthetic_cosine_original,
    bins=40,
    edgecolor="black"
)

axes[1, 0].set_title(
    "Synthetic — Original 1024-D"
)

axes[1, 0].set_xlabel(
    "Pairwise Cosine Similarity"
)

axes[1, 0].set_ylabel(
    "Number of Pairs"
)

axes[1, 0].grid(
    axis="y",
    alpha=0.3
)


# Synthetic Reduced
axes[1, 1].hist(
    synthetic_cosine_reduced,
    bins=40,
    edgecolor="black"
)

axes[1, 1].set_title(
    f"Synthetic — Reduced to {synthetic_components} PCs"
)

axes[1, 1].set_xlabel(
    "Pairwise Cosine Similarity"
)

axes[1, 1].set_ylabel(
    "Number of Pairs"

)

axes[1, 1].grid(
    axis="y",
    alpha=0.3
)


plt.suptitle(
    "Intrinsic Dimensionality — Cosine Geometry Before vs After Reduction",
    fontsize=14
)

plt.tight_layout()


plt.savefig(
    PLOT_FILE,
    dpi=200,
    bbox_inches="tight"
)

print()
print(
    f"Plot saved to:\n{PLOT_FILE}"
)

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
print(
    f"  1024 dimensions → {b01_components} PCs "
    f"for {b01_retained * 100:.4f}% variance"
)

print()

print("Synthetic:")
print(
    f"  1024 dimensions → {synthetic_components} PCs "
    f"for {synthetic_retained * 100:.4f}% variance"
)

print()

print("The following were measured:")
print("  • Dimensionality reduction")
print("  • Variance retention")
print("  • Pairwise cosine distribution")
print("  • Top-1 nearest-neighbour preservation")
print("  • Top-5 neighbour overlap")
print("  • Pearson similarity preservation")
print("  • Spearman ranking preservation")
print("  • Synthetic group-wise preservation")

print()

print("Original embeddings were NOT modified.")

print("=" * 100)