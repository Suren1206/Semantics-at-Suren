import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

embedding_path = (
    BASE_DIR
    / "Data"
    / "experiment2_newsgroup_embeddings.npy"
)

metadata_path = (
    BASE_DIR
    / "Data"
    / "experiment2_newsgroup_metadata.json"
)


# ---------------------------------------------------------
# Load frozen embeddings
# ---------------------------------------------------------

embeddings = np.load(embedding_path)

with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)


# ---------------------------------------------------------
# Basic shape
# ---------------------------------------------------------

print("=== EMBEDDING BASELINE ===")
print(f"Embedding shape: {embeddings.shape}")
print(f"Number of metadata records: {len(metadata)}")


# ---------------------------------------------------------
# Value statistics
# ---------------------------------------------------------

print("\n=== VALUE STATISTICS ===")
print(f"Minimum value: {embeddings.min():.6f}")
print(f"Maximum value: {embeddings.max():.6f}")
print(f"Mean value:    {embeddings.mean():.6f}")
print(f"Std deviation: {embeddings.std():.6f}")


# ---------------------------------------------------------
# L2 norm statistics
# ---------------------------------------------------------

norms = np.linalg.norm(embeddings, axis=1)

print("\n=== L2 NORM STATISTICS ===")
print(f"Minimum norm: {norms.min():.6f}")
print(f"Maximum norm: {norms.max():.6f}")
print(f"Mean norm:    {norms.mean():.6f}")
print(f"Std norm:     {norms.std():.6f}")


# ---------------------------------------------------------
# Numerical validity
# ---------------------------------------------------------

nan_count = np.isnan(embeddings).sum()
inf_count = np.isinf(embeddings).sum()

print("\n=== NUMERICAL VALIDITY ===")
print(f"NaN values: {nan_count}")
print(f"Infinite values: {inf_count}")


# ---------------------------------------------------------
# Metadata alignment
# ---------------------------------------------------------

if embeddings.shape[0] != len(metadata):
    raise ValueError(
        "Embedding count does not match metadata count."
    )

print("\nMetadata alignment: OK")


# ---------------------------------------------------------
# Cosine similarity sample
# ---------------------------------------------------------

rng = np.random.default_rng(42)

sample_size = 1000

document_count = embeddings.shape[0]

indices_a = rng.integers(
    0,
    document_count,
    size=sample_size,
)

indices_b = rng.integers(
    0,
    document_count,
    size=sample_size,
)

vectors_a = embeddings[indices_a]
vectors_b = embeddings[indices_b]

norm_a = np.linalg.norm(vectors_a, axis=1)
norm_b = np.linalg.norm(vectors_b, axis=1)

cosine_similarities = np.sum(
    vectors_a * vectors_b,
    axis=1,
) / (norm_a * norm_b)


# ---------------------------------------------------------
# Cosine similarity statistics
# ---------------------------------------------------------

print("\n=== COSINE SIMILARITY SAMPLE ===")
print(f"Number of sampled pairs: {sample_size}")
print(
    f"Minimum cosine similarity: "
    f"{cosine_similarities.min():.6f}"
)
print(
    f"Maximum cosine similarity: "
    f"{cosine_similarities.max():.6f}"
)
print(
    f"Mean cosine similarity:    "
    f"{cosine_similarities.mean():.6f}"
)
print(
    f"Std cosine similarity:     "
    f"{cosine_similarities.std():.6f}"
)

print("\nBaseline inspection complete.")