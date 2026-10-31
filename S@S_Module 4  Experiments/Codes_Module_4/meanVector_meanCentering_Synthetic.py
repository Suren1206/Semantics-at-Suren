import numpy as np
from pathlib import Path


# ==========================================================================================
# Semantics @ Suren — Module 4
# EXPERIMENT 2 — MEAN VECTOR / MEAN-CENTERING
# ==========================================================================================


print("=" * 100)
print("Semantics @ Suren — Module 4")
print("EXPERIMENT 2 — MEAN VECTOR / MEAN-CENTERING")
print("=" * 100)


# ==========================================================================================
# PATHS
# ==========================================================================================

BASE_DIR = Path(
    "/home/surendran/LangChain/Semantics@Suren/"
    "S@S_Module 4  Experiments"
)

INPUT_FILE = BASE_DIR / "Output_Files" / "synthetic_raw_embeddings.npy"

OUTPUT_FILE = (
    BASE_DIR
    / "Output_Files"
    / "mean_centered_synthetic_50x1024.npy"
)


# ==========================================================================================
# LOAD RAW EMBEDDINGS
# ==========================================================================================

print("\nINPUT")
print("-" * 100)
print(f"Loading raw embeddings:")
print(INPUT_FILE)

embeddings = np.load(INPUT_FILE).astype(np.float64)

print(f"\nEmbedding shape : {embeddings.shape}")


# ==========================================================================================
# VALIDATION
# ==========================================================================================

if embeddings.ndim != 2:
    raise ValueError(
        f"Expected a 2-dimensional embedding matrix. "
        f"Found {embeddings.ndim} dimensions."
    )

N, D = embeddings.shape


# ==========================================================================================
# RAW EMBEDDING STATISTICS
# ==========================================================================================

raw_mean = np.mean(embeddings)
raw_std = np.std(embeddings)
raw_min = np.min(embeddings)
raw_max = np.max(embeddings)

raw_norms = np.linalg.norm(embeddings, axis=1)


print("\n" + "=" * 100)
print("RAW EMBEDDING POPULATION")
print("=" * 100)

print(f"\nNumber of vectors       : {N}")
print(f"Dimensions              : {D}")

print("\nEmbedding value statistics")
print("-" * 100)

print(f"Mean                    : {raw_mean:.8f}")
print(f"Std Deviation           : {raw_std:.8f}")
print(f"Minimum                 : {raw_min:.8f}")
print(f"Maximum                 : {raw_max:.8f}")

print("\nVector L2 norm statistics")
print("-" * 100)

print(f"Mean L2 norm             : {np.mean(raw_norms):.8f}")
print(f"Median L2 norm           : {np.median(raw_norms):.8f}")
print(f"Std L2 norm              : {np.std(raw_norms):.8f}")
print(f"Minimum L2 norm          : {np.min(raw_norms):.8f}")
print(f"Maximum L2 norm          : {np.max(raw_norms):.8f}")


# ==========================================================================================
# MEAN VECTOR
# ==========================================================================================

mean_vector = np.mean(embeddings, axis=0)

mean_vector_norm = np.linalg.norm(mean_vector)

print("\n" + "=" * 100)
print("MEAN VECTOR")
print("=" * 100)

print(f"\nMean vector shape        : {mean_vector.shape}")
print(f"Mean vector L2 norm      : {mean_vector_norm:.8f}")

print("\nMean-vector value statistics")
print("-" * 100)

print(f"Mean of mean vector      : {np.mean(mean_vector):.8f}")
print(f"Std of mean vector       : {np.std(mean_vector):.8f}")
print(f"Minimum mean-vector value: {np.min(mean_vector):.8f}")
print(f"Maximum mean-vector value: {np.max(mean_vector):.8f}")


# ==========================================================================================
# MEAN-CENTERING
# ==========================================================================================

print("\n" + "=" * 100)
print("MEAN-CENTERING")
print("=" * 100)

print("\nApplying:")
print("x_centered = x - mean_vector")

mean_centered = embeddings - mean_vector


# ==========================================================================================
# VERIFY MEAN-CENTERING
# ==========================================================================================

centered_mean_vector = np.mean(mean_centered, axis=0)

centered_mean = np.mean(mean_centered)
centered_std = np.std(mean_centered)
centered_min = np.min(mean_centered)
centered_max = np.max(mean_centered)

centered_norms = np.linalg.norm(mean_centered, axis=1)

print("\n" + "=" * 100)
print("MEAN-CENTERED EMBEDDING POPULATION")
print("=" * 100)

print(f"\nShape                    : {mean_centered.shape}")

print("\nEmbedding value statistics")
print("-" * 100)

print(f"Mean                    : {centered_mean:.12f}")
print(f"Std Deviation           : {centered_std:.8f}")
print(f"Minimum                 : {centered_min:.8f}")
print(f"Maximum                 : {centered_max:.8f}")

print("\nVector L2 norm statistics")
print("-" * 100)

print(f"Mean L2 norm             : {np.mean(centered_norms):.8f}")
print(f"Median L2 norm           : {np.median(centered_norms):.8f}")
print(f"Std L2 norm              : {np.std(centered_norms):.8f}")
print(f"Minimum L2 norm          : {np.min(centered_norms):.8f}")
print(f"Maximum L2 norm          : {np.max(centered_norms):.8f}")

print("\nVerification of mean-centering")
print("-" * 100)

print(
    f"Maximum absolute value of "
    f"centered mean vector     : "
    f"{np.max(np.abs(centered_mean_vector)):.12f}"
)


# ==========================================================================================
# CHANGE IN VECTOR NORMS
# ==========================================================================================

print("\n" + "=" * 100)
print("CHANGE IN VECTOR NORMS")
print("=" * 100)

norm_change = centered_norms - raw_norms

print(f"\nMean norm change         : {np.mean(norm_change):.8f}")
print(f"Median norm change       : {np.median(norm_change):.8f}")
print(f"Minimum norm change      : {np.min(norm_change):.8f}")
print(f"Maximum norm change      : {np.max(norm_change):.8f}")


# ==========================================================================================
# SAVE MEAN-CENTERED EMBEDDINGS
# ==========================================================================================

np.save(OUTPUT_FILE, mean_centered)

print("\n" + "=" * 100)
print("OUTPUT")
print("=" * 100)

print(f"\nMean-centered embeddings saved to:")

print(OUTPUT_FILE)

print(f"\nSaved shape              : {mean_centered.shape}")


# ==========================================================================================
# COMPLETE
# ==========================================================================================

print("\n" + "=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)

print("\nOriginal embeddings were NOT modified.")
print("No PCA was applied.")
print("No whitening was applied.")
print("No dimensionality reduction was applied.")
print("Mean-centering was the only transformation.")
print("=" * 100)