import numpy as np
from pathlib import Path


# ==========================================================================================
# Semantics @ Suren — Module 4
# EXPERIMENT 2
# MEAN VECTOR / MEAN-CENTERING
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

OUTPUT_DIR = BASE_DIR / "Output_Files"

B01_CENTERED_FILE = (
    OUTPUT_DIR
    / "mean_centered_B01.npy"
)

SYNTHETIC_CENTERED_FILE = (
    OUTPUT_DIR
    / "mean_centered_Synthetic.npy"
)


# ==========================================================================================
# UTILITY FUNCTIONS
# ==========================================================================================

def analyse_population(name, embeddings):

    mean_vector = np.mean(embeddings, axis=0)

    raw_norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    centered_embeddings = (
        embeddings - mean_vector
    )

    centered_mean_vector = np.mean(
        centered_embeddings,
        axis=0
    )

    centered_norms = np.linalg.norm(
        centered_embeddings,
        axis=1
    )

    return {
        "name": name,
        "embeddings": embeddings,
        "mean_vector": mean_vector,
        "centered_embeddings": centered_embeddings,

        "number_of_vectors": embeddings.shape[0],
        "dimensions": embeddings.shape[1],

        "mean_vector_norm": np.linalg.norm(mean_vector),

        "mean_vector_mean": np.mean(mean_vector),
        "mean_vector_std": np.std(mean_vector),
        "mean_vector_min": np.min(mean_vector),
        "mean_vector_max": np.max(mean_vector),

        "raw_mean": np.mean(embeddings),
        "raw_std": np.std(embeddings),
        "raw_min": np.min(embeddings),
        "raw_max": np.max(embeddings),

        "raw_norm_mean": np.mean(raw_norms),
        "raw_norm_median": np.median(raw_norms),
        "raw_norm_std": np.std(raw_norms),
        "raw_norm_min": np.min(raw_norms),
        "raw_norm_max": np.max(raw_norms),

        "centered_mean": np.mean(centered_embeddings),
        "centered_std": np.std(centered_embeddings),
        "centered_min": np.min(centered_embeddings),
        "centered_max": np.max(centered_embeddings),

        "centered_norm_mean": np.mean(centered_norms),
        "centered_norm_median": np.median(centered_norms),
        "centered_norm_std": np.std(centered_norms),
        "centered_norm_min": np.min(centered_norms),
        "centered_norm_max": np.max(centered_norms),

        "centered_mean_vector_max_abs":
            np.max(np.abs(centered_mean_vector))
    }


def print_comparison_row(label, b01_value, synthetic_value, precision=8):

    if isinstance(b01_value, (int, np.integer)):
        print(
            f"{label:<42}"
            f"{b01_value:>18}"
            f"{synthetic_value:>18}"
        )
    else:
        print(
            f"{label:<42}"
            f"{b01_value:>18.{precision}f}"
            f"{synthetic_value:>18.{precision}f}"
        )


# ==========================================================================================
# HEADER
# ==========================================================================================

print("=" * 100)
print("Semantics @ Suren — Module 4")
print("EXPERIMENT 2 — MEAN VECTOR / MEAN-CENTERING")
print("B01 vs SYNTHETIC CORPUS")
print("=" * 100)


# ==========================================================================================
# LOAD EMBEDDINGS
# ==========================================================================================

print("\nLoading B01 embeddings...")
b01_embeddings = np.load(B01_FILE).astype(np.float64)

print(f"B01 shape       : {b01_embeddings.shape}")

print("\nLoading Synthetic embeddings...")
synthetic_embeddings = np.load(SYNTHETIC_FILE).astype(np.float64)

print(f"Synthetic shape : {synthetic_embeddings.shape}")


# ==========================================================================================
# VALIDATION
# ==========================================================================================

if b01_embeddings.ndim != 2:
    raise ValueError(
        f"B01 embeddings must be 2-D. Found {b01_embeddings.ndim}-D."
    )

if synthetic_embeddings.ndim != 2:
    raise ValueError(
        f"Synthetic embeddings must be 2-D. Found {synthetic_embeddings.ndim}-D."
    )

if b01_embeddings.shape[1] != synthetic_embeddings.shape[1]:
    raise ValueError(
        "\nB01 and Synthetic embeddings have different dimensions.\n"
        f"B01       : {b01_embeddings.shape[1]}\n"
        f"Synthetic : {synthetic_embeddings.shape[1]}"
    )


# ==========================================================================================
# ANALYSIS
# ==========================================================================================

b01 = analyse_population(
    "B01",
    b01_embeddings
)

synthetic = analyse_population(
    "Synthetic",
    synthetic_embeddings
)


# ==========================================================================================
# RAW POPULATION
# ==========================================================================================

print("\n")
print("=" * 100)
print("RAW EMBEDDING POPULATION")
print("=" * 100)

print()

print(
    f"{'Measure':<42}"
    f"{'B01':>18}"
    f"{'Synthetic':>18}"
)

print("-" * 78)

print_comparison_row(
    "Number of vectors",
    b01["number_of_vectors"],
    synthetic["number_of_vectors"]
)

print_comparison_row(
    "Dimensions",
    b01["dimensions"],
    synthetic["dimensions"]
)

print_comparison_row(
    "Overall embedding-value mean",
    b01["raw_mean"],
    synthetic["raw_mean"]
)

print_comparison_row(
    "Overall embedding-value std",
    b01["raw_std"],
    synthetic["raw_std"]
)

print_comparison_row(
    "Minimum embedding value",
    b01["raw_min"],
    synthetic["raw_min"]
)

print_comparison_row(
    "Maximum embedding value",
    b01["raw_max"],
    synthetic["raw_max"]
)


# ==========================================================================================
# RAW VECTOR NORMS
# ==========================================================================================

print("\n")
print("=" * 100)
print("RAW VECTOR L2 NORMS")
print("=" * 100)

print()

print(
    f"{'Measure':<42}"
    f"{'B01':>18}"
    f"{'Synthetic':>18}"
)

print("-" * 78)

print_comparison_row(
    "Mean L2 norm",
    b01["raw_norm_mean"],
    synthetic["raw_norm_mean"]
)

print_comparison_row(
    "Median L2 norm",
    b01["raw_norm_median"],
    synthetic["raw_norm_median"]
)

print_comparison_row(
    "Std L2 norm",
    b01["raw_norm_std"],
    synthetic["raw_norm_std"]
)

print_comparison_row(
    "Minimum L2 norm",
    b01["raw_norm_min"],
    synthetic["raw_norm_min"]
)

print_comparison_row(
    "Maximum L2 norm",
    b01["raw_norm_max"],
    synthetic["raw_norm_max"]
)


# ==========================================================================================
# MEAN VECTOR
# ==========================================================================================

print("\n")
print("=" * 100)
print("MEAN VECTOR")
print("=" * 100)

print()

print(
    f"{'Measure':<42}"
    f"{'B01':>18}"
    f"{'Synthetic':>18}"
)

print("-" * 78)

print_comparison_row(
    "Mean vector L2 norm",
    b01["mean_vector_norm"],
    synthetic["mean_vector_norm"]
)

print_comparison_row(
    "Mean of mean-vector components",
    b01["mean_vector_mean"],
    synthetic["mean_vector_mean"]
)

print_comparison_row(
    "Std of mean-vector components",
    b01["mean_vector_std"],
    synthetic["mean_vector_std"]
)

print_comparison_row(
    "Minimum mean-vector component",
    b01["mean_vector_min"],
    synthetic["mean_vector_min"]
)

print_comparison_row(
    "Maximum mean-vector component",
    b01["mean_vector_max"],
    synthetic["mean_vector_max"]
)


# ==========================================================================================
# MEAN-CENTERING
# ==========================================================================================

print("\n")
print("=" * 100)
print("MEAN-CENTERING")
print("=" * 100)

print()
print("Operation:")
print("    x_centered = x - mean_vector")

print()
print("No other transformation is applied.")


# ==========================================================================================
# CENTERED POPULATION
# ==========================================================================================

print("\n")
print("=" * 100)
print("MEAN-CENTERED EMBEDDING POPULATION")
print("=" * 100)

print()

print(
    f"{'Measure':<42}"
    f"{'B01':>18}"
    f"{'Synthetic':>18}"
)

print("-" * 78)

print_comparison_row(
    "Overall mean",
    b01["centered_mean"],
    synthetic["centered_mean"],
    precision=12
)

print_comparison_row(
    "Overall std",
    b01["centered_std"],
    synthetic["centered_std"]
)

print_comparison_row(
    "Minimum value",
    b01["centered_min"],
    synthetic["centered_min"]
)

print_comparison_row(
    "Maximum value",
    b01["centered_max"],
    synthetic["centered_max"]
)


# ==========================================================================================
# CENTERED VECTOR NORMS
# ==========================================================================================

print("\n")
print("=" * 100)
print("MEAN-CENTERED VECTOR L2 NORMS")
print("=" * 100)

print()

print(
    f"{'Measure':<42}"
    f"{'B01':>18}"
    f"{'Synthetic':>18}"
)

print("-" * 78)

print_comparison_row(
    "Mean L2 norm",
    b01["centered_norm_mean"],
    synthetic["centered_norm_mean"]
)

print_comparison_row(
    "Median L2 norm",
    b01["centered_norm_median"],
    synthetic["centered_norm_median"]
)

print_comparison_row(
    "Std L2 norm",
    b01["centered_norm_std"],
    synthetic["centered_norm_std"]
)

print_comparison_row(
    "Minimum L2 norm",
    b01["centered_norm_min"],
    synthetic["centered_norm_min"]
)

print_comparison_row(
    "Maximum L2 norm",
    b01["centered_norm_max"],
    synthetic["centered_norm_max"]
)


# ==========================================================================================
# CENTERING VERIFICATION
# ==========================================================================================

print("\n")
print("=" * 100)
print("MEAN-CENTERING VERIFICATION")
print("=" * 100)

print()

print(
    f"{'Measure':<42}"
    f"{'B01':>18}"
    f"{'Synthetic':>18}"
)

print("-" * 78)

print_comparison_row(
    "Maximum absolute centered-mean component",
    b01["centered_mean_vector_max_abs"],
    synthetic["centered_mean_vector_max_abs"],
    precision=12
)

print()
print(
    "Values close to zero confirm that the corresponding "
    "embedding population has been mean-centered."
)


# ==========================================================================================
# SAVE CENTERED EMBEDDINGS
# ==========================================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

np.save(
    B01_CENTERED_FILE,
    b01["centered_embeddings"]
)

np.save(
    SYNTHETIC_CENTERED_FILE,
    synthetic["centered_embeddings"]
)


# ==========================================================================================
# OUTPUT FILES
# ==========================================================================================

print("\n")
print("=" * 100)
print("OUTPUT FILES")
print("=" * 100)

print()
print(f"B01 mean-centered embeddings:")
print(B01_CENTERED_FILE)

print()
print(f"Synthetic mean-centered embeddings:")
print(SYNTHETIC_CENTERED_FILE)


# ==========================================================================================
# COMPLETE
# ==========================================================================================

print("\n")
print("=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)

print()
print("Original B01 embeddings were NOT modified.")
print("Original Synthetic embeddings were NOT modified.")

print()
print("Only mean-centering was applied:")
print("    x_centered = x - mean_vector")

print()
print("No PCA.")
print("No whitening.")
print("No dimensionality reduction.")
print("=" * 100)