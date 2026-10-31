"""
===============================================================================
Semantics @ Suren — Inter-Chunk Similarity
===============================================================================

Purpose:
    Calculate the cosine similarity between two existing chunks:

        C50 = chunk_50
        C48 = chunk_48

    using their ORIGINAL MXBAI embeddings stored in the existing
    20260424_B01 FAISS vector store.

IMPORTANT:
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

VECTOR_STORE_DIR = (
    Path(
        "/home/surendran/LangChain/Semantics@Suren/"
        "S@S_Module 3  Experiments"
    )
    / VERSION_ID
)

INDEX_PATH = (
    VECTOR_STORE_DIR
    / f"{VERSION_ID}.index"
)

INDEX_MAP_PATH = (
    VECTOR_STORE_DIR
    / "index_to_chunk_id.json"
)

CHUNK_A = "chunk_50"
CHUNK_B = "chunk_48"


# =============================================================================
# BLOCK 2 — LOAD EXISTING FAISS INDEX
# =============================================================================

print("=" * 80)
print("Semantics @ Suren — Inter-Chunk Similarity")
print("=" * 80)

print()
print("Loading existing B01 MXBAI vector store...")

index = faiss.read_index(
    str(INDEX_PATH)
)

index_to_chunk_id = json.loads(
    INDEX_MAP_PATH.read_text(
        encoding="utf-8"
    )
)


# =============================================================================
# BLOCK 3 — CREATE CHUNK → FAISS INDEX MAPPING
# =============================================================================

chunk_to_index = {
    chunk_id: int(faiss_index)
    for faiss_index, chunk_id
    in index_to_chunk_id.items()
}


# =============================================================================
# BLOCK 4 — LOCATE C50 AND C48
# =============================================================================

idx_50 = chunk_to_index[CHUNK_A]
idx_48 = chunk_to_index[CHUNK_B]

print()
print(f"{CHUNK_A} → FAISS index {idx_48}")
print(f"{CHUNK_B} → FAISS index {idx_50}")


# =============================================================================
# BLOCK 5 — EXTRACT THE ORIGINAL MXBAI EMBEDDINGS
# =============================================================================

C50 = index.reconstruct(idx_50)
C48 = index.reconstruct(idx_48)

C50 = np.asarray(
    C50,
    dtype=np.float32
)

C48 = np.asarray(
    C48,
    dtype=np.float32
)


# =============================================================================
# BLOCK 6 — CALCULATE VECTOR MAGNITUDES
# =============================================================================

norm_C50 = np.linalg.norm(C50)
norm_C48 = np.linalg.norm(C48)


# =============================================================================
# BLOCK 7 — CALCULATE COSINE SIMILARITY
# =============================================================================

cosine_C50_C48 = (
    np.dot(C50, C48)
    / (norm_C50 * norm_C48)
)


# =============================================================================
# BLOCK 8 — DISPLAY RESULTS
# =============================================================================

print()
print("=" * 80)
print("RESULT")
print("=" * 80)

print()
print(f"C50 dimension : {C50.shape[0]}")
print(f"C48 dimension : {C48.shape[0]}")

print()
print(f"||C50||       : {norm_C50:.6f}")
print(f"||C48||       : {norm_C48:.6f}")

print()
print("Cosine(C50, C48)")
print("-----------------")
print(f"{cosine_C50_C48:.6f}")

print()
print("=" * 80)
print("COMPLETE")
print("=" * 80)