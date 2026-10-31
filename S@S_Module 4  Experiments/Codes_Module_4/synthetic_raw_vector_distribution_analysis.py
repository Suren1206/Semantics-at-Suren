from pathlib import Path
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

# Project directory:
# /home/surendran/LangChain/Semantics@Suren/S@S_Module 4  Experiments

PROJECT_DIR = Path(__file__).resolve().parent.parent

CORPUS_FILE = PROJECT_DIR / "Synthetic_Corpus.ods"

OUTPUT_DIR = PROJECT_DIR / "Output_Files"
OUTPUT_DIR.mkdir(exist_ok=True)

EMBEDDING_FILE = OUTPUT_DIR / "synthetic_raw_embeddings.npy"

OLLAMA_URL = "http://localhost:11434/api/embed"

EMBEDDING_MODEL = "mxbai-embed-large:335m"


# =============================================================================
# HEADER
# =============================================================================

print("=" * 100)
print("Semantics @ Suren — Raw Vector Distribution Analysis")
print("=" * 100)

print()
print(f"Corpus file       : {CORPUS_FILE}")
print(f"Embedding model   : {EMBEDDING_MODEL}")
print("Embedding column  : Column 4 ONLY")
print()


# =============================================================================
# LOAD CORPUS
# =============================================================================

print("Loading corpus...")

if not CORPUS_FILE.exists():
    raise FileNotFoundError(
        f"\nCorpus file not found:\n{CORPUS_FILE}"
    )

# IMPORTANT:
# The first row is DATA, not a header.
# Therefore header=None is required.
df = pd.read_excel(
    CORPUS_FILE,
    engine="odf",
    header=None
)

print(f"Corpus shape      : {df.shape}")


# =============================================================================
# EXTRACT ONLY COLUMN 4
# =============================================================================

print()
print("Extracting column 4 for embedding...")

# Column 4 = index 3 because Python uses zero-based indexing.
texts = df.iloc[:, 3].astype(str).tolist()

# Remove completely empty rows
texts = [
    text.strip()
    for text in texts
    if text.strip() and text.strip().lower() != "nan"
]

print(f"Texts to embed   : {len(texts)}")

print()
print("First 5 texts:")
print("-" * 100)

for i, text in enumerate(texts[:5], start=1):
    print(f"{i}. {text}")


# =============================================================================
# EMBEDDING FUNCTION
# =============================================================================

def get_embedding(text):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["embeddings"][0]


# =============================================================================
# GENERATE EMBEDDINGS
# =============================================================================

print()
print("=" * 100)
print("GENERATING EMBEDDINGS")
print("=" * 100)

embeddings = []

for i, text in enumerate(texts, start=1):

    print(f"Embedding {i}/{len(texts)}...", end="\r")

    vector = get_embedding(text)

    embeddings.append(vector)

print()
print("Embedding generation complete.")


# =============================================================================
# CONVERT TO NUMPY ARRAY
# =============================================================================

embeddings = np.asarray(
    embeddings,
    dtype=np.float64
)

print()
print(f"Embedding matrix shape : {embeddings.shape}")
print(f"Number of vectors      : {embeddings.shape[0]}")
print(f"Dimensions             : {embeddings.shape[1]}")


# =============================================================================
# SAVE RAW EMBEDDINGS
# =============================================================================

np.save(
    EMBEDDING_FILE,
    embeddings
)

print()
print(f"Raw embeddings saved to:")
print(EMBEDDING_FILE)


# =============================================================================
# VECTOR NORM ANALYSIS
# =============================================================================

norms = np.linalg.norm(
    embeddings,
    axis=1
)

print()
print("=" * 100)
print("VECTOR NORM DISTRIBUTION")
print("=" * 100)

print(f"Minimum L2 norm : {norms.min():.6f}")
print(f"Maximum L2 norm : {norms.max():.6f}")
print(f"Mean L2 norm    : {norms.mean():.6f}")
print(f"Median L2 norm  : {np.median(norms):.6f}")
print(f"Std deviation   : {norms.std():.6f}")


# =============================================================================
# RAW EMBEDDING VALUE DISTRIBUTION
# =============================================================================

print()
print("=" * 100)
print("RAW EMBEDDING VALUE DISTRIBUTION")
print("=" * 100)

print(f"Mean embedding value       : {embeddings.mean():.8f}")
print(f"Std deviation              : {embeddings.std():.8f}")
print(f"Minimum embedding value    : {embeddings.min():.8f}")
print(f"Maximum embedding value    : {embeddings.max():.8f}")
print(f"Median embedding value     : {np.median(embeddings):.8f}")


# =============================================================================
# PER-DIMENSION STATISTICS
# =============================================================================

dimension_means = embeddings.mean(axis=0)
dimension_stds = embeddings.std(axis=0)
dimension_mins = embeddings.min(axis=0)
dimension_maxs = embeddings.max(axis=0)

print()
print("=" * 100)
print("PER-DIMENSION STATISTICS")
print("=" * 100)

print()
print(f"Mean of dimension means : {dimension_means.mean():.8f}")
print(f"Min dimension mean      : {dimension_means.min():.8f}")
print(f"Max dimension mean      : {dimension_means.max():.8f}")

print()

print(f"Mean dimension std      : {dimension_stds.mean():.8f}")
print(f"Min dimension std       : {dimension_stds.min():.8f}")
print(f"Max dimension std       : {dimension_stds.max():.8f}")


# =============================================================================
# PAIRWISE COSINE SIMILARITY
# =============================================================================

print()
print("=" * 100)
print("GLOBAL PAIRWISE COSINE SIMILARITY")
print("=" * 100)

# Normalize a COPY.
# Original embeddings remain unchanged.

normalized = embeddings / np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

cosine_matrix = normalized @ normalized.T

n = len(embeddings)

# Extract upper triangle only.
# This gives every unique pair exactly once.

pairwise_cosines = cosine_matrix[
    np.triu_indices(n, k=1)
]

print()
print(f"Embeddings        : {n}")
print(f"Dimensions        : {embeddings.shape[1]}")
print(f"Unique pairs      : {len(pairwise_cosines):,}")


# =============================================================================
# COSINE DISTRIBUTION
# =============================================================================

print()
print("COSINE DISTRIBUTION")
print("-" * 100)

print(f"Mean              : {pairwise_cosines.mean():.8f}")
print(f"Median            : {np.median(pairwise_cosines):.8f}")
print(f"Std Deviation     : {pairwise_cosines.std():.8f}")
print(f"Minimum           : {pairwise_cosines.min():.8f}")
print(f"Maximum           : {pairwise_cosines.max():.8f}")

print()
print("Percentiles")
print("-" * 100)

percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

for p in percentiles:
    value = np.percentile(pairwise_cosines, p)
    print(f"P{p:02d}               : {value:.8f}")


# =============================================================================
# ADDITIONAL DISTRIBUTION STATISTICS
# =============================================================================

print()
print("ADDITIONAL SUMMARY STATISTICS")
print("-" * 100)

print(f"Cosine < 0        : {(pairwise_cosines < 0).sum():,}")
print(f"Cosine = 0        : {(pairwise_cosines == 0).sum():,}")
print(f"Cosine > 0        : {(pairwise_cosines > 0).sum():,}")

print(
    f"Cosine > 0.50     : {(pairwise_cosines > 0.50).sum():,}"
)

print(
    f"Cosine > 0.75     : {(pairwise_cosines > 0.75).sum():,}"
)

print(
    f"Cosine > 0.90     : {(pairwise_cosines > 0.90).sum():,}"
)


# =============================================================================
# HISTOGRAM
# =============================================================================

print()
print("=" * 100)
print("HISTOGRAM — PAIRWISE COSINE SIMILARITY")
print("=" * 100)

plt.figure(figsize=(10, 6))

plt.hist(
    pairwise_cosines,
    bins=50
)

plt.xlabel("Pairwise Cosine Similarity")
plt.ylabel("Frequency")
plt.title("Raw Pairwise Cosine Similarity Distribution")

plt.tight_layout()

plt.show()


# =============================================================================
# COMPLETE
# =============================================================================

print()
print("=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)

print()
print("Original corpus was not modified.")
print("Original embeddings were not modified.")
print("No mean-centering, PCA, whitening, or other transformation was applied.")