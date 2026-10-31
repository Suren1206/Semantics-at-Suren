"""
===============================================================================
Semantics @ Suren — Module 1
Experiment: Chunk Perturbation Across Embedding Models
File: chunk_perturbation.py
===============================================================================

EXPERIMENTAL OBJECTIVE
----------------------
Measure the effect of perturbing ONE corpus chunk on dense retrieval.

The user query remains CONSTANT:

    "When can self-certification be used for illness absence?"

Only chunk_0 is changed.

BEFORE
------
chunk_0 contains the original policy text.

AFTER
-----
chunk_0 is replaced by the perturbed policy text supplied for this experiment.

All other 160 chunks remain unchanged.

-------------------------------------------------------------------------------
EXPERIMENTAL CONTROL
-------------------------------------------------------------------------------

                    BEFORE                  AFTER

Corpus             B01 corpus              B01 corpus
Chunk 0             Original text           Perturbed text
Chunks 1-160        Unchanged              Unchanged
Query               Same                   Same

The experiment is therefore a controlled CORPUS perturbation.

-------------------------------------------------------------------------------
EMBEDDING MODELS
-------------------------------------------------------------------------------

Two independent embedding spaces are created temporarily in memory:

    1. mxbai-embed-large:335m
    2. BAAI/bge-m3

No existing vector store is modified.

No new vector-store files are persisted.

-------------------------------------------------------------------------------
RETRIEVAL PROCESS
-------------------------------------------------------------------------------

For each model:

    BEFORE corpus
        ↓
    Embed all 161 chunks
        ↓
    Normalize vectors
        ↓
    Temporary FAISS IndexFlatIP
        ↓
    Original query
        ↓
    Top-5

    AFTER corpus
        ↓
    Embed perturbed chunk corpus
        ↓
    Normalize vectors
        ↓
    Temporary FAISS IndexFlatIP
        ↓
    Same original query
        ↓
    Top-5

Because all vectors are L2-normalized:

    Inner Product = Cosine Similarity

-------------------------------------------------------------------------------
IMPORTANT DESIGN DECISION
-------------------------------------------------------------------------------

The existing B01 FAISS index and embeddings.npy are NOT used for the
perturbation retrieval.

This is deliberate.

The existing B01 store represents the ORIGINAL corpus. Once chunk_0 is
perturbed, its stored embedding is no longer valid for the AFTER condition.

Therefore both BEFORE and AFTER representations are constructed temporarily
from the common JSON corpus.

-------------------------------------------------------------------------------
OUTPUT
-------------------------------------------------------------------------------

Each retrieval result occupies exactly ONE physical line:

    Rank | Chunk ID | Chunk Index | Cosine | Header | Text

Therefore every Top-5 retrieval produces exactly five lines suitable for
copy/paste into the existing Excel analysis template.

===============================================================================
"""

import json
import sys
from pathlib import Path

import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer


# =============================================================================
# BLOCK 1 — PROJECT PATH CONFIGURATION
# =============================================================================
#
# PURPOSE
# -------
# Define the locations of the experimental corpus and the original raGROW
# project.
#
# The directory name contains a REAL trailing space:
#
#     "S@S_Module 1 Experiments "
#
# This must be preserved exactly.
#
# The script itself remains outside raGROW_S@S so that the original project
# code and vector stores remain untouched.
# =============================================================================

PROJECT_DIR = Path(
    "/home/surendran/LangChain/Semantics@Suren/"
    "S@S_Module 1 Experiments "
)

RAGROW_DIR = PROJECT_DIR / "raGROW_S@S"

B01_DIR = PROJECT_DIR / "20260424_B01"

CHUNK_METADATA_PATH = B01_DIR / "chunk_id_to_metadata.json"


# =============================================================================
# BLOCK 2 — EXPERIMENT PARAMETERS
# =============================================================================
#
# PURPOSE
# -------
# Define the fixed query, the chunk being perturbed, and the exact replacement
# text.
#
# ONLY chunk_0 is modified.
#
# The query itself is identical for BEFORE and AFTER.
# =============================================================================

QUERY = (
    "When can self-certification be used for illness absence?"
)

VERSION_ID = "20260424_B01"

PERTURBED_CHUNK_ID = "chunk_0"

TOP_K = 5

MXBAI_MODEL = "mxbai-embed-large:335m"

BGE_MODEL = "BAAI/bge-m3"

OLLAMA_URL = "http://localhost:11434/api/embeddings"


# =============================================================================
# BLOCK 3 — PERTURBED CHUNK TEXT
# =============================================================================
#
# PURPOSE
# -------
# Store the AFTER version of chunk_0 exactly as supplied for this experiment.
#
# No other chunk is changed.
#
# IMPORTANT
# ---------
# This text is deliberately kept as supplied. The experiment should measure
# the retrieval consequence of THIS perturbation, not a corrected or rewritten
# version of it.
# =============================================================================

PERTURBED_CHUNK_TEXT = (

    "Absence exceeding 2 consecutive working days requires submission of a "
    "self-declaration form within 24 hours of return. "
)


# =============================================================================
# BLOCK 4 — LOAD COMMON B01 CHUNK CORPUS
# =============================================================================
#
# PURPOSE
# -------
# Load the 161 chunks from the existing JSON metadata.
#
# The JSON file is the source of truth for:
#
#     chunk_id
#     chunk_index
#     header
#     text
#
# The existing B01 FAISS index is intentionally NOT used.
#
# This gives us a clean corpus from which both BEFORE and AFTER temporary
# vector spaces can be constructed.
# =============================================================================

def load_chunks():

    if not CHUNK_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Chunk metadata file not found:\n{CHUNK_METADATA_PATH}"
        )

    metadata = json.loads(
        CHUNK_METADATA_PATH.read_text(encoding="utf-8")
    )

    chunks = []

    for chunk_id, item in metadata.items():

        chunks.append(
            {
                "chunk_id": chunk_id,
                "chunk_index": item.get("chunk_index"),
                "header": item.get("header", ""),
                "text": item.get("text", ""),
            }
        )

    chunks.sort(key=lambda x: x["chunk_index"])

    if len(chunks) != 161:
        raise ValueError(
            f"Expected 161 chunks, but found {len(chunks)}."
        )

    return chunks


# =============================================================================
# BLOCK 5 — CREATE BEFORE / AFTER CORPORA
# =============================================================================
#
# PURPOSE
# -------
# Create two independent in-memory representations of the corpus.
#
# BEFORE:
#     Exact text from chunk_id_to_metadata.json.
#
# AFTER:
#     Identical to BEFORE except chunk_0.text is replaced with the supplied
#     perturbation.
#
# The metadata itself is not written back to disk.
# =============================================================================

def create_corpora(chunks):

    before_chunks = [
        {
            "chunk_id": chunk["chunk_id"],
            "chunk_index": chunk["chunk_index"],
            "header": chunk["header"],
            "text": chunk["text"],
        }
        for chunk in chunks
    ]

    after_chunks = [
        {
            "chunk_id": chunk["chunk_id"],
            "chunk_index": chunk["chunk_index"],
            "header": chunk["header"],
            "text": chunk["text"],
        }
        for chunk in chunks
    ]

    perturbed = False

    for chunk in after_chunks:

        if chunk["chunk_id"] == PERTURBED_CHUNK_ID:

            chunk["text"] = PERTURBED_CHUNK_TEXT
            perturbed = True
            break

    if not perturbed:
        raise ValueError(
            f"{PERTURBED_CHUNK_ID} was not found in the corpus."
        )

    return before_chunks, after_chunks


# =============================================================================
# BLOCK 6 — NORMALIZE EMBEDDINGS
# =============================================================================
#
# PURPOSE
# -------
# Apply L2 normalization to every embedding.
#
# After normalization:
#
#     cosine_similarity(q, d)
#
# is equivalent to:
#
#     dot(q, d)
#
# This permits the use of FAISS IndexFlatIP while retaining cosine similarity
# semantics.
# =============================================================================

def normalize_embeddings(embeddings):

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    if np.any(norms == 0):

        raise ValueError(
            "One or more embeddings has zero norm."
        )

    return embeddings / norms


# =============================================================================
# BLOCK 7 — MXBAI CHUNK EMBEDDING
# =============================================================================
#
# PURPOSE
# -------
# Generate mxbai embeddings using the SAME Ollama embedding endpoint used by
# the existing Retriever.
#
# This preserves the model and embedding mechanism of the validated B01
# environment while avoiding the existing stored vector representation.
#
# Each chunk is embedded independently.
# =============================================================================

def embed_mxbai_text(text):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MXBAI_MODEL,
            "prompt": text
        },
        timeout=60
    )

    response.raise_for_status()

    vector = response.json()["embedding"]

    vector = np.asarray(
        vector,
        dtype=np.float32
    )

    return vector


def build_mxbai_embeddings(chunks):

    embeddings = []

    for position, chunk in enumerate(chunks, start=1):

        print(
            f"mxbai embedding chunk "
            f"{position}/{len(chunks)}",
            end="\r"
        )

        vector = embed_mxbai_text(
            chunk["text"]
        )

        embeddings.append(vector)

    print()

    embeddings = np.vstack(embeddings)

    return normalize_embeddings(embeddings)


# =============================================================================
# BLOCK 8 — BGE-M3 CHUNK EMBEDDING
# =============================================================================
#
# PURPOSE
# -------
# Generate BGE-M3 embeddings for the same corpus.
#
# BGE-M3 has its OWN embedding space and therefore cannot use embeddings.npy
# from B01.
#
# The resulting matrix exists only in memory.
# =============================================================================

def build_bge_embeddings(model, chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True
    )

    return normalize_embeddings(embeddings)


# =============================================================================
# BLOCK 9 — BUILD TEMPORARY FAISS VECTOR STORE
# =============================================================================
#
# PURPOSE
# -------
# Create an in-memory FAISS IndexFlatIP from the supplied embedding matrix.
#
# This is a TEMPORARY vector store.
#
# Nothing is written to:
#
#     20260424_B01.index
#     embeddings.npy
#     index_to_chunk_id.json
#     chunk_id_to_metadata.json
#
# The index disappears when the script terminates.
# =============================================================================

def build_temporary_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(
        embeddings.astype(np.float32)
    )

    return index


# =============================================================================
# BLOCK 10 — MXBAI QUERY EMBEDDING
# =============================================================================
#
# PURPOSE
# -------
# Generate the mxbai representation of the fixed experimental query.
#
# This uses the same embedding mechanism as the existing Retriever.
# =============================================================================

def embed_mxbai_query(query):

    vector = embed_mxbai_text(query)

    vector = vector.reshape(1, -1)

    return normalize_embeddings(vector)


# =============================================================================
# BLOCK 11 — BGE-M3 QUERY EMBEDDING
# =============================================================================
#
# PURPOSE
# -------
# Generate the BGE-M3 representation of the fixed experimental query.
#
# The query is deliberately unchanged between BEFORE and AFTER.
# =============================================================================

def embed_bge_query(model, query):

    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False
    )

    return normalize_embeddings(vector)


# =============================================================================
# BLOCK 12 — RETRIEVE TOP-K FROM TEMPORARY VECTOR STORE
# =============================================================================
#
# PURPOSE
# -------
# Perform Top-K retrieval against an in-memory FAISS index.
#
# Since the index contains normalized vectors and uses inner product:
#
#     FAISS score = cosine similarity
#
# The FAISS result index is mapped back to the corresponding chunk using the
# same positional ordering used when the temporary index was constructed.
# =============================================================================

def retrieve_from_index(
    index,
    query_vector,
    chunks,
    top_k=5
):

    top_k = min(
        top_k,
        index.ntotal
    )

    scores, indices = index.search(
        query_vector.astype(np.float32),
        top_k
    )

    results = []

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if idx == -1:
            continue

        chunk = chunks[int(idx)]

        results.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "header": chunk["header"],
                "text": " ".join(
                    chunk["text"].split()
                ),
            }
        )

    return results


# =============================================================================
# BLOCK 13 — COPY/PASTE-FRIENDLY OUTPUT
# =============================================================================
#
# PURPOSE
# -------
# Print exactly ONE physical line per retrieved chunk.
#
# This format is designed specifically for the existing Excel analysis
# template.
#
# Five retrieved chunks therefore produce exactly five lines.
#
# No additional line breaks are introduced inside the chunk text.
# =============================================================================

def print_results(title, results):

    print("\n")
    print("=" * 100)
    print(title)
    print("=" * 100)

    for result in results:

        print(
            f"Rank {result['rank']} | "
            f"Chunk ID: {result['chunk_id']} | "
            f"Chunk Index: {result['chunk_index']} | "
            f"Cosine: {result['score']:.6f} | "
            f"Header: {result['header']} | "
            f"Text: {result['text']}"
        )

    print("-" * 100)


# =============================================================================
# BLOCK 14 — MAIN EXPERIMENT
# =============================================================================
#
# PURPOSE
# -------
# Execute the complete controlled experiment.
#
# The experiment proceeds model-by-model:
#
# MXBAI
# -----
#     BEFORE corpus → temporary vector store → query → Top-5
#     AFTER corpus  → temporary vector store → query → Top-5
#
# BGE-M3
# ------
#     BEFORE corpus → temporary vector store → query → Top-5
#     AFTER corpus  → temporary vector store → query → Top-5
#
# The same query is used in every condition.
# =============================================================================

def main():

    print("=" * 100)
    print("Semantics @ Suren — Chunk Perturbation Experiment")
    print("=" * 100)

    print(f"\nVersion : {VERSION_ID}")
    print(f"Query   : {QUERY}")
    print(f"Perturb : {PERTURBED_CHUNK_ID}")
    print(f"Top-K   : {TOP_K}")

    # -------------------------------------------------------------------------
    # STEP 1 — Load common corpus
    # -------------------------------------------------------------------------

    print("\nLoading B01 chunk metadata...")

    chunks = load_chunks()

    print(
        f"Loaded {len(chunks)} chunks."
    )

    # -------------------------------------------------------------------------
    # STEP 2 — Construct BEFORE and AFTER corpora in memory
    # -------------------------------------------------------------------------

    before_chunks, after_chunks = create_corpora(
        chunks
    )

    print(
        f"Created BEFORE and AFTER corpus representations in memory."
    )

    # -------------------------------------------------------------------------
    # STEP 3 — Load BGE-M3
    # -------------------------------------------------------------------------

    print("\nLoading BGE-M3 model...")

    bge_model = SentenceTransformer(
        BGE_MODEL
    )

    # -------------------------------------------------------------------------
    # STEP 4 — Build MXBAI BEFORE embeddings
    # -------------------------------------------------------------------------

    print("\nBuilding temporary MXBAI BEFORE vector store...")

    mxbai_before_embeddings = build_mxbai_embeddings(
        before_chunks
    )

    mxbai_before_index = build_temporary_index(
        mxbai_before_embeddings
    )

    # -------------------------------------------------------------------------
    # STEP 5 — Build MXBAI AFTER embeddings
    # -------------------------------------------------------------------------
    #
    # Only chunk_0 has changed, but we construct the complete AFTER embedding
    # matrix so that BEFORE and AFTER represent two independent corpus states.
    # -------------------------------------------------------------------------

    print("\nBuilding temporary MXBAI AFTER vector store...")

    mxbai_after_embeddings = build_mxbai_embeddings(
        after_chunks
    )

    mxbai_after_index = build_temporary_index(
        mxbai_after_embeddings
    )

    # -------------------------------------------------------------------------
    # STEP 6 — MXBAI query embedding
    # -------------------------------------------------------------------------

    print("\nEmbedding query with MXBAI...")

    mxbai_query_vector = embed_mxbai_query(
        QUERY
    )

    # -------------------------------------------------------------------------
    # STEP 7 — MXBAI BEFORE retrieval
    # -------------------------------------------------------------------------

    mxbai_before_results = retrieve_from_index(
        mxbai_before_index,
        mxbai_query_vector,
        before_chunks,
        TOP_K
    )

    print_results(
        "MXBAI — BEFORE CHUNK PERTURBATION",
        mxbai_before_results
    )

    # -------------------------------------------------------------------------
    # STEP 8 — MXBAI AFTER retrieval
    # -------------------------------------------------------------------------

    mxbai_after_results = retrieve_from_index(
        mxbai_after_index,
        mxbai_query_vector,
        after_chunks,
        TOP_K
    )

    print_results(
        "MXBAI — AFTER CHUNK PERTURBATION",
        mxbai_after_results
    )

    # -------------------------------------------------------------------------
    # STEP 9 — Build BGE-M3 BEFORE vector store
    # -------------------------------------------------------------------------

    print("\nBuilding temporary BGE-M3 BEFORE vector store...")

    bge_before_embeddings = build_bge_embeddings(
        bge_model,
        before_chunks
    )

    bge_before_index = build_temporary_index(
        bge_before_embeddings
    )

    # -------------------------------------------------------------------------
    # STEP 10 — Build BGE-M3 AFTER vector store
    # -------------------------------------------------------------------------

    print("\nBuilding temporary BGE-M3 AFTER vector store...")

    bge_after_embeddings = build_bge_embeddings(
        bge_model,
        after_chunks
    )

    bge_after_index = build_temporary_index(
        bge_after_embeddings
    )

    # -------------------------------------------------------------------------
    # STEP 11 — BGE-M3 query embedding
    # -------------------------------------------------------------------------

    print("\nEmbedding query with BGE-M3...")

    bge_query_vector = embed_bge_query(
        bge_model,
        QUERY
    )

    # -------------------------------------------------------------------------
    # STEP 12 — BGE-M3 BEFORE retrieval
    # -------------------------------------------------------------------------

    bge_before_results = retrieve_from_index(
        bge_before_index,
        bge_query_vector,
        before_chunks,
        TOP_K
    )

    print_results(
        "BGE-M3 — BEFORE CHUNK PERTURBATION",
        bge_before_results
    )

    # -------------------------------------------------------------------------
    # STEP 13 — BGE-M3 AFTER retrieval
    # -------------------------------------------------------------------------

    bge_after_results = retrieve_from_index(
        bge_after_index,
        bge_query_vector,
        after_chunks,
        TOP_K
    )

    print_results(
        "BGE-M3 — AFTER CHUNK PERTURBATION",
        bge_after_results
    )

    # -------------------------------------------------------------------------
    # STEP 14 — Completion message
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("EXPERIMENT COMPLETE")
    print("=" * 100)

    print(
        "\nFour controlled retrieval conditions completed:"
    )

    print("  1. MXBAI  — BEFORE")
    print("  2. MXBAI  — AFTER")
    print("  3. BGE-M3 — BEFORE")
    print("  4. BGE-M3 — AFTER")

    print(
        "\nNo existing B01 vector-store files were modified."
    )

    print(
        "All experimental vector stores existed only in memory."
    )

    print("=" * 100)


# =============================================================================
# BLOCK 15 — SCRIPT ENTRY POINT
# =============================================================================
#
# PURPOSE
# -------
# Execute the experiment only when this file is run directly.
# =============================================================================

if __name__ == "__main__":
    main()