import numpy as np
from pathlib import Path

# ============================================================
# Semantics @ Suren — Mean Vector & Mean Centering
# ============================================================

# This script is inside:
# S@S_Module 4  Experiments/Codes_Module_4/

# Therefore this points to:
# S@S_Module 4  Experiments/
BASE_DIR = Path(__file__).resolve().parent.parent

# Input embeddings
EMBEDDING_FILE = BASE_DIR / "20260424_B01" / "embeddings.npy"

# Output files
MEAN_VECTOR_FILE = BASE_DIR / "mean_vector_160.npy"
CENTERED_VECTOR_FILE = BASE_DIR / "mean_centered_160x1024.npy"


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("=" * 100)
print("MEAN VECTOR AND MEAN-CENTERING ANALYSIS")
print("=" * 100)

print("\nLoading embeddings:")
print(EMBEDDING_FILE)

embeddings = np.load(EMBEDDING_FILE).astype(np.float64)

print(f"\nOriginal embedding shape : {embeddings.shape}")


# ============================================================
# CALCULATE MEAN VECTOR
# ============================================================

mean_vector = np.mean(embeddings, axis=0)

print("\nMean vector calculated.")

print(f"Mean vector shape        : {mean_vector.shape}")


# ============================================================
# MEAN CENTERING
# ============================================================

# Each embedding is shifted by subtracting the global mean vector.
#
# centered[i] = embedding[i] - mean_vector

mean_centered = embeddings - mean_vector

print(f"Mean-centered shape       : {mean_centered.shape}")


# ============================================================
# VERIFY MEAN CENTERING
# ============================================================

new_mean = np.mean(mean_centered, axis=0)

print("\nVerification")
print("-" * 100)

print(f"Maximum absolute value of new mean : "
      f"{np.max(np.abs(new_mean)):.12e}")


# ============================================================
# SAVE RESULTS
# ============================================================

np.save(MEAN_VECTOR_FILE, mean_vector)

np.save(CENTERED_VECTOR_FILE, mean_centered)


# ============================================================
# OUTPUT INFORMATION
# ============================================================

print("\nSaved files")
print("-" * 100)

print(f"Mean vector       : {MEAN_VECTOR_FILE}")
print(f"Mean-centered     : {CENTERED_VECTOR_FILE}")


# ============================================================
# SHOW FIRST FEW VALUES
# ============================================================

print("\nFirst 10 dimensions of mean vector:")
print(mean_vector[:10])

print("\nFirst 10 dimensions of mean-centered vector 0:")
print(mean_centered[0, :10])


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)