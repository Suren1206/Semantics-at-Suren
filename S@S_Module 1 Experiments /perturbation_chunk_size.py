"""
===============================================================================
Semantics @ Suren — Module 1
Experiment: Chunk Boundary Perturbation — Chunk Size
File: perturbation_chunk_size.py
===============================================================================

EXPERIMENTAL OBJECTIVE
----------------------
Study the effect of chunk size on dense retrieval while keeping overlap
constant at zero.

Fixed query:

    "When can self-certification be used for illness absence?"

Two chunking conditions are compared:

    CONDITION A
        Chunk size = 200 characters
        Overlap    = 0 characters

    CONDITION B
        Chunk size = 400 characters
        Overlap    = 0 characters

Each condition is evaluated independently using:

    1. mxbai-embed-large:335m
    2. BGE-M3

Therefore four retrieval experiments are performed:

    MXBAI  — 200 characters / 0 overlap
    MXBAI  — 400 characters / 0 overlap

    BGE-M3 — 200 characters / 0 overlap
    BGE-M3 — 400 characters / 0 overlap

-------------------------------------------------------------------------------
EXPERIMENTAL CONTROL
-------------------------------------------------------------------------------

The existing 20260424_B01 vector store is NOT modified.

No experimental FAISS indexes or embedding files are written to disk.

All experimental vector stores exist only in memory for the duration of
this script.

The same source corpus is used for all four conditions.

The same fixed query is used for all four conditions.

-------------------------------------------------------------------------------
CHUNKING
-------------------------------------------------------------------------------

This experiment deliberately uses character-based chunking.

    200 / 0
        [0 ---------------- 199]
        [200 -------------- 399]
        ...

    400 / 0
        [0 ---------------- 399]
        [400 -------------- 799]
        ...

Overlap is fixed at zero.

Therefore the isolated variable is:

    CHUNK SIZE

and NOT overlap.

-------------------------------------------------------------------------------
EMBEDDING
-------------------------------------------------------------------------------

MXBAI:

    mxbai-embed-large:335m

    Generated through the local Ollama embedding endpoint.

BGE-M3:

    BAAI/bge-m3

    Generated using SentenceTransformers.

Each model has its own independent embedding space.

-------------------------------------------------------------------------------
RETRIEVAL
-------------------------------------------------------------------------------

FAISS IndexFlatIP is used.

All embeddings are L2 normalized.

Therefore:

    Inner Product = Cosine Similarity

For every condition:

    Chunk corpus
        ↓
    Model-specific embeddings
        ↓
    L2 normalization
        ↓
    Temporary FAISS index
        ↓
    Fixed query embedding
        ↓
    Top-5 retrieval
        ↓
    Cosine similarity scores

-------------------------------------------------------------------------------
OUTPUT
-------------------------------------------------------------------------------

The script first displays the generated chunk structure for both chunk sizes.

It then produces four Top-5 retrieval sections.

Each retrieved chunk is printed as ONE physical line:

    Rank | Chunk ID | Chunk Index | Cosine | Header | Text

This format is designed for direct copy/paste into the existing Excel
analysis template.

===============================================================================
"""

import json
from pathlib import Path

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


# =============================================================================
# BLOCK 1 — PROJECT PATH CONFIGURATION
# =============================================================================
#
# PURPOSE
# -------
# Define the location of the experiment folder and the source corpus.
#
# IMPORTANT
# ---------
# The existing B01 vector store is deliberately NOT loaded by this script.
# We only use the source corpus to construct the experimental chunks.
# =============================================================================

PROJECT_DIR = Path(
    "/home/surendran/LangChain/Semantics@Suren/"
    "S@S_Module 1 Experiments "
)

MASTER_CORPUS_PATH = (
    PROJECT_DIR
    / "raGROW_S@S"
    / "data"
    / "master_corpus.txt"
)


# =============================================================================
# BLOCK 2 — FIXED EXPERIMENT PARAMETERS
# =============================================================================
#
# PURPOSE
# -------
# Define the parameters that remain constant throughout the experiment.
#
# The only experimental variable is CHUNK_SIZE.
# =============================================================================

QUERY = (
    "When can self-certification be used for illness absence?"
)

CHUNK_SIZE_SMALL = 200

CHUNK_SIZE_LARGE = 400

NO_OVERLAP = 0

TOP_K = 5

MXBAI_MODEL = "mxbai-embed-large:335m"

BGE_MODEL = "BAAI/bge-m3"

OLLAMA_URL = "http://localhost:11434/api/embeddings"


# =============================================================================
# BLOCK 3 — LOAD SOURCE CORPUS
# =============================================================================
#
# PURPOSE
# -------
# Load the original corpus without modifying it.
#
# This corpus is the common source for both chunk-size conditions.
# =============================================================================

def load_source_corpus():

    if not MASTER_CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Master corpus not found:\n{MASTER_CORPUS_PATH}"
        )

    text = MASTER_CORPUS_PATH.read_text(
        encoding="utf-8"
    )

    if not text.strip():
        raise ValueError(
            "Source corpus is empty."
        )

    return text


# =============================================================================
# BLOCK 4 — CREATE CHARACTER-BASED CHUNKS
# =============================================================================
#
# PURPOSE
# -------
# Construct chunks using a fixed character count.
#
# For this experiment:
#
#     200-character condition → chunk_size = 200, overlap = 0
#     400-character condition → chunk_size = 400, overlap = 0
#
# The function is generic so that the experiment can easily be extended
# later without changing the retrieval mechanism.
# =============================================================================

def create_chunks(
    text,
    chunk_size,
    overlap
):

    if chunk_size <= 0:
        raise ValueError(
            "Chunk size must be positive."
        )

    if overlap < 0:
        raise ValueError(
            "Overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "Overlap must be smaller than chunk size."
        )

    step = chunk_size - overlap

    chunks = []

    start = 0
    chunk_number = 0

    while start < len(text):

        end = start + chunk_size

        chunk_text = text[start:end]

        if not chunk_text.strip():
            break

        chunks.append(
            {
                "chunk_id": f"experiment_chunk_{chunk_number}",
                "chunk_index": chunk_number,
                "start": start,
                "end": min(end, len(text)),
                "text": chunk_text,
            }
        )

        chunk_number += 1
        start += step

    return chunks


# =============================================================================
# BLOCK 5 — NORMALIZE EMBEDDINGS
# =============================================================================
#
# PURPOSE
# -------
# Apply L2 normalization to all embeddings.
#
# This allows FAISS IndexFlatIP to provide cosine similarity.
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
# BLOCK 6 — MXBAI TEXT EMBEDDING
# =============================================================================
#
# PURPOSE
# -------
# Generate one MXBAI embedding through the local Ollama server.
#
# This follows the same Ollama endpoint used by the existing retrieval
# implementation.
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

    return np.asarray(
        vector,
        dtype=np.float32
    )


# =============================================================================
# BLOCK 7 — BUILD MXBAI CORPUS EMBEDDINGS
# =============================================================================
#
# PURPOSE
# -------
# Generate MXBAI embeddings for every experimental chunk.
# =============================================================================

def build_mxbai_embeddings(chunks):

    embeddings = []

    for position, chunk in enumerate(
        chunks,
        start=1
    ):

        print(
            f"MXBAI embedding chunk "
            f"{position}/{len(chunks)}",
            end="\r"
        )

        vector = embed_mxbai_text(
            chunk["text"]
        )

        embeddings.append(vector)

    print()

    embeddings = np.vstack(
        embeddings
    )

    return normalize_embeddings(
        embeddings
    )


# =============================================================================
# BLOCK 8 — BUILD BGE-M3 CORPUS EMBEDDINGS
# =============================================================================
#
# PURPOSE
# -------
# Generate BGE-M3 embeddings for every experimental chunk.
#
# BGE-M3 uses a completely separate embedding space from MXBAI.
# =============================================================================

def build_bge_embeddings(
    model,
    chunks
):

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

    return normalize_embeddings(
        embeddings
    )


# =============================================================================
# BLOCK 9 — BUILD TEMPORARY FAISS INDEX
# =============================================================================
#
# PURPOSE
# -------
# Create an in-memory FAISS IndexFlatIP.
#
# Nothing is written to disk.
# =============================================================================

def build_temporary_index(
    embeddings
):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings.astype(
            np.float32
        )
    )

    return index


# =============================================================================
# BLOCK 10 — EMBED MXBAI QUERY
# =============================================================================
#
# PURPOSE
# -------
# Create the fixed MXBAI query vector.
#
# The same query vector is reused for both chunk-size conditions.
# =============================================================================

def embed_mxbai_query(query):

    vector = embed_mxbai_text(
        query
    )

    vector = vector.reshape(
        1, -1
    )

    return normalize_embeddings(
        vector
    )


# =============================================================================
# BLOCK 11 — EMBED BGE-M3 QUERY
# =============================================================================
#
# PURPOSE
# -------
# Create the fixed BGE-M3 query vector.
#
# The same query vector is reused for both chunk-size conditions.
# =============================================================================

def embed_bge_query(
    model,
    query
):

    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False
    )

    return normalize_embeddings(
        vector
    )


# =============================================================================
# BLOCK 12 — RETRIEVE TOP-5 RESULTS
# =============================================================================
#
# PURPOSE
# -------
# Search the temporary FAISS index and return the five highest scoring
# chunks.
#
# Because both query and corpus vectors are normalized:
#
#     FAISS Inner Product = Cosine Similarity
# =============================================================================

def retrieve(
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
        query_vector.astype(
            np.float32
        ),
        top_k
    )

    results = []

    for rank, (score, idx) in enumerate(
        zip(
            scores[0],
            indices[0]
        ),
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
                "header": chunk.get(
                    "header",
                    ""
                ),
                "text": " ".join(
                    chunk["text"].split()
                ),
            }
        )

    return results


# =============================================================================
# BLOCK 13 — DISPLAY CHUNK STRUCTURE
# =============================================================================
#
# PURPOSE
# -------
# Display the chunks generated under both chunk-size conditions.
#
# This lets us visually inspect how changing the chunk size changes the
# boundaries before looking at retrieval behaviour.
#
# Only chunks containing "absence" are displayed to keep the inspection
# focused on the query-relevant region.
# =============================================================================

def print_chunk_structure(
    chunks_200,
    chunks_400
):

    print("\n")
    print("=" * 100)
    print("CHUNK STRUCTURE INSPECTION")
    print("=" * 100)

    print("\n200 CHARACTER / 0 OVERLAP")
    print("-" * 100)

    for chunk in chunks_200:

        if "absence" in chunk["text"].lower():

            print(
                f"\nChunk ID    : {chunk['chunk_id']}"
            )

            print(
                f"Chunk Index : {chunk['chunk_index']}"
            )

            print(
                f"Characters  : "
                f"{chunk['start']} - {chunk['end']}"
            )

            print(
                f"Text        : "
                f"{' '.join(chunk['text'].split())}"
            )

    print("\n\n400 CHARACTER / 0 OVERLAP")
    print("-" * 100)

    for chunk in chunks_400:

        if "absence" in chunk["text"].lower():

            print(
                f"\nChunk ID    : {chunk['chunk_id']}"
            )

            print(
                f"Chunk Index : {chunk['chunk_index']}"
            )

            print(
                f"Characters  : "
                f"{chunk['start']} - {chunk['end']}"
            )

            print(
                f"Text        : "
                f"{' '.join(chunk['text'].split())}"
            )


# =============================================================================
# BLOCK 14 — COPY/PASTE-FRIENDLY RETRIEVAL OUTPUT
# =============================================================================
#
# PURPOSE
# -------
# Print each retrieval result as one physical line.
#
# This preserves the established format used in the previous perturbation
# experiments and allows direct copy/paste into the Excel analysis template.
# =============================================================================

def print_results(
    title,
    results
):

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
# BLOCK 15 — MAIN EXPERIMENT
# =============================================================================
#
# PURPOSE
# -------
# Execute all four controlled retrieval experiments.
#
#                    200 / 0             400 / 0
#
# MXBAI              CONDITION 1         CONDITION 2
# BGE-M3             CONDITION 3         CONDITION 4
#
# The chunking variable is isolated while the query remains fixed.
# =============================================================================

def main():

    print("=" * 100)
    print(
        "Semantics @ Suren — "
        "Chunk Boundary Perturbation — Chunk Size Experiment"
    )
    print("=" * 100)

    print(
        f"\nQuery   : {QUERY}"
    )

    print(
        f"Top-K   : {TOP_K}"
    )

    print(
        "Conditions: 200 characters / 0 overlap "
        "vs. 400 characters / 0 overlap"
    )

    print(
        "\nExisting 20260424_B01 vector store will NOT be modified."
    )

    # =========================================================================
    # STEP 1 — LOAD SOURCE CORPUS
    # =========================================================================

    print(
        "\nLoading source corpus..."
    )

    source_text = load_source_corpus()

    print(
        f"Loaded source corpus: "
        f"{len(source_text)} characters."
    )

    # =========================================================================
    # STEP 2 — CREATE 200-CHARACTER CORPUS
    # =========================================================================

    print(
        "\nCreating 200-character / 0-overlap chunks..."
    )

    chunks_200 = create_chunks(
        source_text,
        CHUNK_SIZE_SMALL,
        NO_OVERLAP
    )

    print(
        f"Created {len(chunks_200)} chunks."
    )

    # =========================================================================
    # STEP 3 — CREATE 400-CHARACTER CORPUS
    # =========================================================================

    print(
        "\nCreating 400-character / 0-overlap chunks..."
    )

    chunks_400 = create_chunks(
        source_text,
        CHUNK_SIZE_LARGE,
        NO_OVERLAP
    )

    print(
        f"Created {len(chunks_400)} chunks."
    )

    # =========================================================================
    # STEP 4 — DISPLAY CHUNK STRUCTURE
    # =========================================================================

    print_chunk_structure(
        chunks_200,
        chunks_400
    )

    # =========================================================================
    # STEP 5 — LOAD BGE-M3
    # =========================================================================

    print(
        "\n\nLoading BGE-M3..."
    )

    bge_model = SentenceTransformer(
        BGE_MODEL
    )

    # =========================================================================
    # STEP 6 — MXBAI QUERY EMBEDDING
    # =========================================================================

    print(
        "\nEmbedding query with MXBAI..."
    )

    mxbai_query = embed_mxbai_query(
        QUERY
    )

    # =========================================================================
    # STEP 7 — MXBAI / 200 CHARACTERS
    # =========================================================================

    print(
        "\nBuilding temporary MXBAI "
        "200-character / 0-overlap vector store..."
    )

    mxbai_200_embeddings = (
        build_mxbai_embeddings(
            chunks_200
        )
    )

    mxbai_200_index = (
        build_temporary_index(
            mxbai_200_embeddings
        )
    )

    mxbai_200_results = retrieve(
        mxbai_200_index,
        mxbai_query,
        chunks_200,
        TOP_K
    )

    print_results(
        "MXBAI — 200 CHARACTERS / 0 OVERLAP",
        mxbai_200_results
    )

    # =========================================================================
    # STEP 8 — MXBAI / 400 CHARACTERS
    # =========================================================================

    print(
        "\nBuilding temporary MXBAI "
        "400-character / 0-overlap vector store..."
    )

    mxbai_400_embeddings = (
        build_mxbai_embeddings(
            chunks_400
        )
    )

    mxbai_400_index = (
        build_temporary_index(
            mxbai_400_embeddings
        )
    )

    mxbai_400_results = retrieve(
        mxbai_400_index,
        mxbai_query,
        chunks_400,
        TOP_K
    )

    print_results(
        "MXBAI — 400 CHARACTERS / 0 OVERLAP",
        mxbai_400_results
    )

    # =========================================================================
    # STEP 9 — BGE-M3 QUERY EMBEDDING
    # =========================================================================

    print(
        "\nEmbedding query with BGE-M3..."
    )

    bge_query = embed_bge_query(
        bge_model,
        QUERY
    )

    # =========================================================================
    # STEP 10 — BGE-M3 / 200 CHARACTERS
    # =========================================================================

    print(
        "\nBuilding temporary BGE-M3 "
        "200-character / 0-overlap vector store..."
    )

    bge_200_embeddings = (
        build_bge_embeddings(
            bge_model,
            chunks_200
        )
    )

    bge_200_index = (
        build_temporary_index(
            bge_200_embeddings
        )
    )

    bge_200_results = retrieve(
        bge_200_index,
        bge_query,
        chunks_200,
        TOP_K
    )

    print_results(
        "BGE-M3 — 200 CHARACTERS / 0 OVERLAP",
        bge_200_results
    )

    # =========================================================================
    # STEP 11 — BGE-M3 / 400 CHARACTERS
    # =========================================================================

    print(
        "\nBuilding temporary BGE-M3 "
        "400-character / 0-overlap vector store..."
    )

    bge_400_embeddings = (
        build_bge_embeddings(
            bge_model,
            chunks_400
        )
    )

    bge_400_index = (
        build_temporary_index(
            bge_400_embeddings
        )
    )

    bge_400_results = retrieve(
        bge_400_index,
        bge_query,
        chunks_400,
        TOP_K
    )

    print_results(
        "BGE-M3 — 400 CHARACTERS / 0 OVERLAP",
        bge_400_results
    )

    # =========================================================================
    # STEP 12 — EXPERIMENT COMPLETION
    # =========================================================================

    print("\n")
    print("=" * 100)
    print("EXPERIMENT COMPLETE")
    print("=" * 100)

    print(
        "\nFour retrieval conditions completed:"
    )

    print(
        "  1. MXBAI  — 200 characters / 0 overlap"
    )

    print(
        "  2. MXBAI  — 400 characters / 0 overlap"
    )

    print(
        "  3. BGE-M3 — 200 characters / 0 overlap"
    )

    print(
        "  4. BGE-M3 — 400 characters / 0 overlap"
    )

    print(
        "\nExisting B01 vector store was not modified."
    )

    print(
        "All experimental vector stores were temporary and in-memory."
    )

    print("=" * 100)


# =============================================================================
# BLOCK 16 — SCRIPT ENTRY POINT
# =============================================================================
#
# PURPOSE
# -------
# Execute the experiment when the file is run directly.
# =============================================================================

if __name__ == "__main__":
    main()