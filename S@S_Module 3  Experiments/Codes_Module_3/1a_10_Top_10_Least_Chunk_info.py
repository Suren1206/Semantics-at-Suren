"""
===============================================================================
Semantics @ Suren — C0 Inter-Chunk Similarity Analysis
===============================================================================

Objective
---------
Compare chunk_0 against all other 160 chunks in the existing B01 vector store.

Output
------
1. TOP 10  : highest cosine similarity to chunk_0
2. BOTTOM 10: lowest cosine similarity to chunk_0

The output contains:
    Rank
    Chunk ID
    Chunk Index
    Cosine similarity

The existing vector store is READ ONLY.
===============================================================================
"""

import json
import numpy as np
import faiss
from pathlib import Path


# =============================================================================
# BLOCK 1 — CONFIGURATION
# =============================================================================

VERSION_ID = "20260424_B01"
TARGET_CHUNK = "chunk_0"
TOP_K = 10


# =============================================================================
# BLOCK 2 — VECTOR STORE LOCATION
# =============================================================================

PROJECT_ROOT = Path(
    "/home/surendran/LangChain/Semantics@Suren"
)

VECTOR_STORE_DIR = (
    PROJECT_ROOT
    / "S@S_Module 1 Experiments "
    / "raGROW_S@S"
    / "data"
    / "vector_store"
    / VERSION_ID
)

INDEX_PATH = VECTOR_STORE_DIR / f"{VERSION_ID}.index"
INDEX_MAP_PATH = VECTOR_STORE_DIR / "index_to_chunk_id.json"


# =============================================================================
# BLOCK 3 — LOAD EXISTING FAISS INDEX
# =============================================================================

print("=" * 100)
print("Semantics @ Suren — C0 Inter-Chunk Similarity")
print("=" * 100)

print()
print("Loading existing B01 MXBAI vector store...")

index = faiss.read_index(str(INDEX_PATH))

index_to_chunk_id = json.loads(
    INDEX_MAP_PATH.read_text(encoding="utf-8")
)

print(f"Total vectors: {index.ntotal}")


# =============================================================================
# BLOCK 4 — CREATE CHUNK ID → FAISS INDEX MAPPING
# =============================================================================

chunk_to_index = {
    chunk_id: int(index_position)
    for index_position, chunk_id
    in index_to_chunk_id.items()
}


# =============================================================================
# BLOCK 5 — EXTRACT CHUNK_0
# =============================================================================

if TARGET_CHUNK not in chunk_to_index:
    raise ValueError(
        f"{TARGET_CHUNK} not found in index mapping."
    )

c0_index = chunk_to_index[TARGET_CHUNK]

C0 = np.asarray(
    index.reconstruct(c0_index),
    dtype=np.float32
)

C0_norm = np.linalg.norm(C0)

if C0_norm == 0:
    raise ValueError(
        "chunk_0 has a zero-norm embedding."
    )


# =============================================================================
# BLOCK 6 — CALCULATE COSINE SIMILARITY AGAINST ALL OTHER CHUNKS
# =============================================================================

results = []

for chunk_id, chunk_index in chunk_to_index.items():

    # Exclude chunk_0 itself
    if chunk_id == TARGET_CHUNK:
        continue

    C = np.asarray(
        index.reconstruct(chunk_index),
        dtype=np.float32
    )

    C_norm = np.linalg.norm(C)

    if C_norm == 0:
        continue

    cosine = np.dot(C0, C) / (C0_norm * C_norm)

    results.append(
        {
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "cosine": float(cosine)
        }
    )


# =============================================================================
# BLOCK 7 — SORT ALL 160 COMPARISONS
# =============================================================================

# Highest → lowest
results_sorted = sorted(
    results,
    key=lambda x: x["cosine"],
    reverse=True
)


# =============================================================================
# BLOCK 8 — TOP 10 CLOSEST TO CHUNK_0
# =============================================================================

print()
print("=" * 100)
print("TOP 10 CHUNKS — HIGHEST COSINE SIMILARITY TO CHUNK_0")
print("=" * 100)

print()

for rank, result in enumerate(
    results_sorted[:TOP_K],
    start=1
):

    print(
        f"Rank {rank:2d} | "
        f"Chunk ID   : {result['chunk_id']} | "
        f"Chunk Index: {result['chunk_index']} | "
        f"Cosine     : {result['cosine']:.6f}"
    )


# =============================================================================
# BLOCK 9 — BOTTOM 10 LEAST SIMILAR TO CHUNK_0
# =============================================================================
#
# IMPORTANT:
# These are the SAME global ranks from the complete 160-chunk comparison.
#
# Therefore, if there are 160 comparisons:
#
#     Bottom 10 = global ranks 151 → 160
#
# =============================================================================

print()
print("=" * 100)
print("BOTTOM 10 CHUNKS — LOWEST COSINE SIMILARITY TO CHUNK_0")
print("=" * 100)

print()

total = len(results_sorted)

for rank, result in enumerate(
    results_sorted[-TOP_K:],
    start=total - TOP_K + 1
):

    print(
        f"Rank {rank:3d} | "
        f"Chunk ID   : {result['chunk_id']} | "
        f"Chunk Index: {result['chunk_index']} | "
        f"Cosine     : {result['cosine']:.6f}"
    )


# =============================================================================
# BLOCK 10 — SUMMARY
# =============================================================================

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print()
print(f"Target chunk          : {TARGET_CHUNK}")
print(f"Total vectors         : {index.ntotal}")
print(f"Chunks compared       : {len(results)}")
print(f"Highest 10 displayed  : YES")
print(f"Lowest 10 displayed   : YES")
print()
print("Vector store modified : NO")
print("=" * 100)