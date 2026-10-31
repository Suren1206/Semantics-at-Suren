"""
===============================================================================
Semantics @ Suren — Query / Chunk Similarity Experiment
===============================================================================

OBJECTIVE
---------
Compare the semantic geometry of three text relationships:

    1. Query  <-> Chunk 1
    2. Chunk 0  <-> Chunk 2
    3. Chunk new <-> Chunk 3

EMBEDDING MODEL
---------------
MXBAI

IMPORTANT
---------
This experiment creates embeddings temporarily in memory only.

No existing FAISS index is loaded.
No existing vector store is modified.
No new vector store is persisted.

The embedding mechanism uses the same OllamaEmbeddings interface used
by the existing raGROW/Semantics experiments.

===============================================================================
"""


# =============================================================================
# BLOCK 1 — IMPORTS
# =============================================================================

import numpy as np

from langchain_ollama import OllamaEmbeddings


# =============================================================================
# BLOCK 2 — EMBEDDING MODEL CONFIGURATION
# =============================================================================

MODEL_NAME = "mxbai-embed-large:335m"

embeddings = OllamaEmbeddings(
    model=MODEL_NAME
)


# =============================================================================
# BLOCK 3 — INPUT TEXT
# =============================================================================

QUERY = (
    "How should I notify my manager when I am unable to attend work?"
)


CHUNK_1 = (
    "Employees must notify their direct manager at least 30 minutes before "
    "their scheduled start time unless physically incapacit…5 consecutive "
    "working days must be supported by medical certification issued no later " 
    "than the 7th calendar day of absence. "
)


CHUNK_2 = (
    "if you are unable to attend work, you should reach out your manager "
    "with your phone to explain him the reason. In case you are not able "
    "to reach him, then you can give a whatsapp message explaining him "
    "the background of absence"
)


# =============================================================================
# BLOCK 4 — GENERATE EMBEDDINGS
# =============================================================================

print("=" * 100)
print("MXBAI — QUERY / CHUNK TRIANGULAR SIMILARITY")
print("=" * 100)

print()
print("Generating embeddings...")

texts = [
    QUERY,
    CHUNK_1,
    CHUNK_2
]

query_embedding, chunk1_embedding, chunk2_embedding = (
    embeddings.embed_documents(texts)
)

query_embedding = np.asarray(
    query_embedding,
    dtype=np.float32
)

chunk1_embedding = np.asarray(
    chunk1_embedding,
    dtype=np.float32
)

chunk2_embedding = np.asarray(
    chunk2_embedding,
    dtype=np.float32
)

print()
print(f"Embedding model     : {MODEL_NAME}")
print(f"Embedding dimension : {len(query_embedding)}")


# =============================================================================
# BLOCK 5 — COSINE SIMILARITY FUNCTION
# =============================================================================

def cosine_similarity(vector_a, vector_b):

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    return np.dot(vector_a, vector_b) / (
        norm_a * norm_b
    )


# =============================================================================
# BLOCK 6 — CALCULATE ALL THREE RELATIONSHIPS
# =============================================================================

query_chunk1 = cosine_similarity(
    query_embedding,
    chunk1_embedding
)

query_chunk2 = cosine_similarity(
    query_embedding,
    chunk2_embedding
)

chunk1_chunk2 = cosine_similarity(
    chunk1_embedding,
    chunk2_embedding
)


# =============================================================================
# BLOCK 7 — PRINT COSINE SIMILARITY RESULTS
# =============================================================================

print()
print("=" * 100)
print("COSINE SIMILARITY RESULTS")
print("=" * 100)

print()

print("1. QUERY <-> CHUNK 1")
print(
    f"   Cosine similarity : {query_chunk1:.6f}"
)

print()

print("2. QUERY <-> CHUNK 2")
print(
    f"   Cosine similarity : {query_chunk2:.6f}"
)

print()

print("3. CHUNK 1 <-> CHUNK 2")
print(
    f"   Cosine similarity : {chunk1_chunk2:.6f}"
)


# =============================================================================
# BLOCK 8 — PRINT INPUT REFERENCE
# =============================================================================

print()
print("=" * 100)
print("INPUT REFERENCE")
print("=" * 100)

print()
print("QUERY:")
print(QUERY)

print()
print("CHUNK 1:")
print(CHUNK_1)

print()
print("CHUNK 2:")
print(CHUNK_2)


# =============================================================================
# BLOCK 9 — EXPERIMENT SUMMARY
# =============================================================================

print()
print("=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)

print()
print("Relationships calculated : 3")
print("Vector store modified    : NO")
print("Embeddings persisted     : NO")
print("=" * 100)