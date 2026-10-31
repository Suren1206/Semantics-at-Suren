"""
===============================================================================
Semantics @ Suren — C0 Inter-Chunk Similarity Analysis
===============================================================================

Objective:
    Compare chunk_0 against all other 160 chunks in the existing B01
    MXBAI vector store.

Output:
    TOP 10    -> highest cosine similarity to chunk_0
    BOTTOM 10 -> lowest cosine similarity to chunk_0

For every result:
    Rank
    Chunk ID
    Chunk Index
    Cosine
    Header
    Full Chunk Text

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
    / "S@S_Module 3  Experiments"
    / VERSION_ID
)

INDEX_PATH = VECTOR_STORE_DIR / f"{VERSION_ID}.index"
INDEX_MAP_PATH = VECTOR_STORE_DIR / "index_to_chunk_id.json"
METADATA_PATH = VECTOR_STORE_DIR / "chunk_id_to_metadata.json"


# =============================================================================
# BLOCK 3 — LOAD EXISTING VECTOR STORE AND METADATA
# =============================================================================

print("=" * 100)
print("Semantics @ Suren — C0 Inter-Chunk Similarity Analysis")
print("=" * 100)

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

chunk_id_to_metadata = json.loads(
    METADATA_PATH.read_text(
        encoding="utf-8"
    )
)
print(f"Total vectors loaded : {index.ntotal}")


# =============================================================================
# BLOCK 4 — CREATE CHUNK ID → FAISS INDEX MAPPING
# =============================================================================

chunk_to_index = {
    chunk_id: int(index_position)
    for index_position, chunk_id
    in index_to_chunk_id.items()
}


# =============================================================================
# BLOCK 5 — EXTRACT CHUNK_0 EMBEDDING
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
# BLOCK 6 — CALCULATE COSINE AGAINST ALL OTHER CHUNKS
# =============================================================================

results = []

for chunk_id, chunk_index in chunk_to_index.items():

    if chunk_id == TARGET_CHUNK:
        continue

    chunk_vector = np.asarray(
        index.reconstruct(chunk_index),
        dtype=np.float32
    )

    chunk_norm = np.linalg.norm(chunk_vector)

    if chunk_norm == 0:
        continue

    cosine = (
        np.dot(C0, chunk_vector)
        / (C0_norm * chunk_norm)
    )

    # Retrieve metadata for the actual chunk text.
    metadata = chunk_id_to_metadata.get(
        chunk_id,
        {}
    )

    # Support the common metadata field names.
    header = metadata.get(
        "header",
        metadata.get("Header", "")
    )

    text = metadata.get(
        "text",
        metadata.get(
            "Text",
            metadata.get("content", "")
        )
    )

    results.append(
        {
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "cosine": float(cosine),
            "header": header,
            "text": text
        }
    )


# =============================================================================
# BLOCK 7 — SORT RESULTS
# =============================================================================

results_sorted = sorted(
    results,
    key=lambda x: x["cosine"],
    reverse=True
)


# =============================================================================
# BLOCK 8 — TOP 10 CHUNKS
# =============================================================================

print()
print("=" * 100)
print("TOP 10 CHUNKS — HIGHEST COSINE SIMILARITY TO CHUNK_0")
print("=" * 100)

for rank, result in enumerate(
    results_sorted[:TOP_K],
    start=1
):

    print()
    print(f"Rank       : {rank}")
    print(f"Chunk ID    : {result['chunk_id']}")
    print(f"Chunk Index : {result['chunk_index']}")
    print(f"Cosine     : {result['cosine']:.6f}")
    print(f"Header     : {result['header']}")
    print(f"Text       : {result['text']}")


# =============================================================================
# BLOCK 9 — BOTTOM 10 CHUNKS
# =============================================================================

print()
print("=" * 100)
print("BOTTOM 10 CHUNKS — LOWEST COSINE SIMILARITY TO CHUNK_0")
print("=" * 100)

total = len(results_sorted)

for rank, result in enumerate(
    results_sorted[-TOP_K:],
    start=total - TOP_K + 1
):

    print()
    print(f"Rank       : {rank}")
    print(f"Chunk ID    : {result['chunk_id']}")
    print(f"Chunk Index : {result['chunk_index']}")
    print(f"Cosine     : {result['cosine']:.6f}")
    print(f"Header     : {result['header']}")
    print(f"Text       : {result['text']}")


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
print(f"Top 10 displayed      : YES")
print(f"Bottom 10 displayed   : YES")
print()
print("Embedding model       : MXBAI")
print("Vector store modified : NO")
print("=" * 100)