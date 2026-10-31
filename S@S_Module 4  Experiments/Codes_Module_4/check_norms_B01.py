import numpy as np
from pathlib import Path

# ============================================================
# Semantics @ Suren — Embedding L2 Norm Analysis
# ============================================================

# Project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Actual embeddings location
EMBEDDING_FILE = BASE_DIR / "20260424_B01" / "embeddings.npy"

print("=" * 80)
print("Semantics @ Suren — Embedding L2 Norm Analysis")
print("=" * 80)

print(f"\nLoading: {EMBEDDING_FILE}")

# Load embeddings
embeddings = np.load(EMBEDDING_FILE)

print(f"Embedding shape: {embeddings.shape}")

# Calculate L2 norm of every vector
norms = np.linalg.norm(embeddings, axis=1)

print("\n" + "=" * 80)
print("L2 NORM STATISTICS")
print("=" * 80)

print(f"Minimum L2 norm : {norms.min():.10f}")
print(f"Maximum L2 norm : {norms.max():.10f}")
print(f"Mean L2 norm    : {norms.mean():.10f}")

print("\n" + "=" * 80)

if np.allclose(norms, 1.0, atol=1e-5):
    print("Conclusion: Embeddings appear to be L2-normalized.")
else:
    print("Conclusion: Embeddings are NOT all L2-normalized.")

print("=" * 80)