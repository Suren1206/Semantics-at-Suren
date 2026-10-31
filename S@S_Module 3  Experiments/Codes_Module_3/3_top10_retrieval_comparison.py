"""
===============================================================================
Semantics @ Suren — MXBAI Top-10 Chunk Embeddings
===============================================================================

Purpose:
    Retrieve Top-10 chunks using the existing MXBAI B01 FAISS index and save
    their complete stored embeddings into a NumPy .npy file.

Query:
    When can self-certification be used for illness absence?

Output:
    mxbai_top10_embeddings.npy

Expected shape:
    (10, 1024)

IMPORTANT:
    The existing FAISS vector store is READ ONLY.
    No existing vector store is modified.
===============================================================================
"""

import json
import requests
import numpy as np
import faiss
from pathlib import Path


# =============================================================================
# BLOCK 1 — CONFIGURATION
# =============================================================================

QUERY = "When can self-certification be used for illness absence?"

TOP_K = 10

MODEL = "mxbai-embed-large:335m"

OLLAMA_URL = "http://localhost:11434/api/embeddings"

VERSION_ID = "20260424_B01"


# =============================================================================
# BLOCK 2 — LOCATE EXISTING B01 VECTOR STORE
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

VECTOR_STORE_DIR = (
    BASE_DIR
    / "raGROW_S@S"
    / "data"
    / "vector_store"
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

METADATA_PATH = (
    VECTOR_STORE_DIR
    / "chunk_id_to_metadata.json"
)


# =============================================================================
# BLOCK 3 — OUTPUT FILE
# =============================================================================
#
# IMPORTANT:
# The file is deliberately saved beside this Python script.
#
# If you run:
#
#     python top10_mxbai_embeddings.py
#
# from:
#
#     S@S_Module 1 Experiments
#
# you will get:
#
#     S@S_Module 1 Experiments/
#         mxbai_top10_embeddings.npy
#
# =============================================================================

OUTPUT_PATH = (
    BASE_DIR
    / "mxbai_top10_embeddings.npy"
)


# =============================================================================
# BLOCK 4 — LOAD EXISTING MXBAI FAISS INDEX
# =============================================================================

print("=" * 100)
print("Semantics @ Suren — MXBAI Top-10 Chunk Embeddings")
print("=" * 100)

print()
print("Query :", QUERY)
print("Model :", MODEL)
print()

print("Loading existing MXBAI FAISS index...")

if not INDEX_PATH.exists():
    raise FileNotFoundError(
        f"\nFAISS index not found:\n{INDEX_PATH}"
    )

if not INDEX_MAP_PATH.exists():
    raise FileNotFoundError(
        f"\nIndex mapping not found:\n{INDEX_MAP_PATH}"
    )

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"\nMetadata file not found:\n{METADATA_PATH}"
    )


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

print(
    f"Loaded {index.ntotal} vectors."
)


# =============================================================================
# BLOCK 5 — CREATE MXBAI QUERY EMBEDDING
# =============================================================================

print()
print("Embedding query with MXBAI...")

response = requests.post(
    OLLAMA_URL,
    json={
        "model": MODEL,
        "prompt": QUERY
    },
    timeout=120
)

response.raise_for_status()

query_vector = np.asarray(
    response.json()["embedding"],
    dtype=np.float32
)

query_norm = np.linalg.norm(
    query_vector
)

if query_norm == 0:
    raise ValueError(
        "Query embedding has zero norm."
    )

query_vector = (
    query_vector / query_norm
).reshape(1, -1)


# =============================================================================
# BLOCK 6 — RETRIEVE TOP-10
# =============================================================================

print(
    "Searching existing MXBAI index..."
)

scores, indices = index.search(
    query_vector,
    TOP_K
)


# =============================================================================
# BLOCK 7 — EXTRACT THE ACTUAL STORED EMBEDDINGS
# =============================================================================
#
# FAISS IndexFlatIP stores the complete vectors.
#
# index.reconstruct(idx)
#
# therefore gives us the exact stored MXBAI vector associated with the
# retrieved chunk.
#
# =============================================================================

top10_embeddings = []

top10_chunk_ids = []

for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):

    if idx == -1:
        continue

    chunk_id = index_to_chunk_id.get(
        str(int(idx))
    )

    if chunk_id is None:
        continue

    metadata = chunk_id_to_metadata.get(
        chunk_id
    )

    if metadata is None:
        continue

    embedding = index.reconstruct(
        int(idx)
    )

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    top10_embeddings.append(
        embedding
    )

    top10_chunk_ids.append(
        chunk_id
    )

    print(
        f"Rank {rank:2d} | "
        f"{chunk_id:10s} | "
        f"Cosine = {float(score):.6f} | "
        f"Dimension = {embedding.shape[0]}"
    )


# =============================================================================
# BLOCK 8 — BUILD (10, 1024) NUMPY MATRIX
# =============================================================================

if len(top10_embeddings) != TOP_K:
    raise RuntimeError(
        f"Expected {TOP_K} embeddings, "
        f"but extracted {len(top10_embeddings)}."
    )

top10_embeddings = np.vstack(
    top10_embeddings
).astype(
    np.float32
)

print()
print("Embedding matrix created.")
print(
    f"Shape : {top10_embeddings.shape}"
)

print(
    f"Dtype : {top10_embeddings.dtype}"
)


# =============================================================================
# BLOCK 9 — SAVE THE .NPY FILE
# =============================================================================

np.save(
    str(OUTPUT_PATH),
    top10_embeddings
)


# =============================================================================
# BLOCK 10 — VERIFY THAT THE FILE ACTUALLY EXISTS
# =============================================================================

if not OUTPUT_PATH.exists():
    raise RuntimeError(
        "ERROR: .npy file was not created."
    )

file_size = OUTPUT_PATH.stat().st_size

print()
print("=" * 100)
print("NPY FILE CREATED SUCCESSFULLY")
print("=" * 100)

print()
print(
    f"File : {OUTPUT_PATH}"
)

print(
    f"Size : {file_size:,} bytes"
)

print(
    f"Shape: {top10_embeddings.shape}"
)

print()
print("Top-10 chunks stored in row order:")

for row, chunk_id in enumerate(
    top10_chunk_ids
):
    print(
        f"Row {row:2d} -> {chunk_id}"
    )

print()
print("=" * 100)
print("COMPLETE")
print("=" * 100)