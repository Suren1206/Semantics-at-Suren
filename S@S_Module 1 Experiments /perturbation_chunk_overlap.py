"""
===============================================================================
Semantics @ Suren — Module 1
Experiment: Chunk Boundary Perturbation — Overlap
File: chunk_overlap_perturbation.py
===============================================================================

EXPERIMENTAL OBJECTIVE
----------------------
Study the effect of chunk overlap on dense retrieval.

Fixed query:

    "When can self-certification be used for illness absence?"

Two chunking conditions are compared:

    CONDITION A
        Chunk size = 300 characters
        Overlap    = 0 characters

    CONDITION B
        Chunk size = 300 characters
        Overlap    = 50 characters

Both conditions are tested independently with:

    1. mxbai-embed-large:335m
    2. BGE-M3

The experiment therefore produces four retrieval results:

    MXBAI  — 300 chars / 0 overlap
    MXBAI  — 300 chars / 50 overlap

    BGE-M3 — 300 chars / 0 overlap
    BGE-M3 — 300 chars / 50 overlap

-------------------------------------------------------------------------------
IMPORTANT EXPERIMENTAL CONTROL
-------------------------------------------------------------------------------

The existing vector store:

    20260424_B01

is NOT modified, rebuilt, overwritten, or otherwise changed.

The experiment creates all required embeddings and FAISS indexes temporarily
in memory.

No experimental .index or .npy files are written to disk.

-------------------------------------------------------------------------------
CHUNKING RULE
-------------------------------------------------------------------------------

For this experiment the baseline chunking condition is:

    Maximum chunk size = 300 characters
    Overlap = 0 characters

The overlap condition uses:

    Chunk size = 300 characters
    Overlap = 50 characters

The source corpus is the same in both cases.

-------------------------------------------------------------------------------
RETRIEVAL
-------------------------------------------------------------------------------

For every condition:

    Corpus
       ↓
    Chunking
       ↓
    Embedding
       ↓
    L2 normalization
       ↓
    FAISS IndexFlatIP
       ↓
    Fixed query
       ↓
    Top-5 retrieval

Because vectors are normalized:

    Inner Product = Cosine Similarity

-------------------------------------------------------------------------------
OUTPUT
-------------------------------------------------------------------------------

The script first displays the chunks generated around the relevant
ABSENCE MANAGEMENT content so the boundary/overlap effect can be inspected.

After that, each retrieval result is printed as exactly ONE physical line:

    Rank | Chunk ID | Chunk Index | Cosine | Header | Text

This makes the Top-5 output directly copy/pasteable into the existing
Excel analysis template.

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
# Locate the existing B01 metadata and corpus.
#
# The existing FAISS index is deliberately NOT loaded.
#
# The metadata JSON provides the original chunk text from which this
# experiment creates its own temporary chunking structures.
# =============================================================================

PROJECT_DIR = Path(
    "/home/surendran/LangChain/Semantics@Suren/"
    "S@S_Module 1 Experiments "
)

B01_DIR = PROJECT_DIR / "20260424_B01"

CHUNK_METADATA_PATH = (
    B01_DIR / "chunk_id_to_metadata.json"
)

MASTER_CORPUS_PATH = (
    PROJECT_DIR / "raGROW_S@S" / "data" /
    "master_corpus.txt"
)


# =============================================================================
# BLOCK 2 — EXPERIMENT PARAMETERS
# =============================================================================
#
# PURPOSE
# -------
# Define the fixed experimental parameters.
#
# These values should remain unchanged when comparing the two chunking
# conditions.
# =============================================================================

QUERY = (
    "When can self-certification be used for illness absence?"
)

CHUNK_SIZE = 300

NO_OVERLAP = 0

OVERLAP_50 = 50

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
# Load the source corpus used to construct the experimental chunks.
#
# Preference is given to the master corpus because the experiment concerns
# CHUNK BOUNDARIES, not the existing B01 chunk boundaries.
#
# The corpus itself is never modified.
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
# BLOCK 4 — CREATE 300-CHARACTER CHUNKS
# =============================================================================
#
# PURPOSE
# -------
# Construct fixed-size character chunks.
#
# NO OVERLAP CONDITION
# --------------------
#
#     [0 ---------------- 299]
#     [300 -------------- 599]
#     [600 -------------- 899]
#
# OVERLAP CONDITION
# -----------------
#
#     [0 ---------------- 299]
#                   [250 ---------------- 549]
#                                     [500 ---------------- 799]
#
# Therefore consecutive chunks overlap by exactly 50 characters.
#
# NOTE
# ----
# Chunking is character-based for this controlled experiment. We are not
# changing the existing production sentence-based chunking strategy.
# =============================================================================

def create_chunks(
    text,
    chunk_size=300,
    overlap=0
):

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
                "end": end,
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
# L2-normalize every embedding so that FAISS inner product represents cosine
# similarity.
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
# BLOCK 6 — EMBED TEXT WITH MXBAI
# =============================================================================
#
# PURPOSE
# -------
# Generate MXBAI embeddings through the existing local Ollama endpoint.
#
# This follows the same embedding mechanism used by the existing Retriever.
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
# BLOCK 7 — BUILD MXBAI EMBEDDINGS
# =============================================================================
#
# PURPOSE
# -------
# Embed all chunks generated by one of the two chunking conditions.
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
# BLOCK 8 — BUILD BGE-M3 EMBEDDINGS
# =============================================================================
#
# PURPOSE
# -------
# Generate BGE-M3 embeddings for the same experimentally generated chunks.
#
# BGE-M3 uses its own embedding space. Therefore its embeddings are generated
# independently and are never mixed with MXBAI vectors.
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
# NOTHING IS SAVED TO DISK.
#
# The index exists only for the duration of this Python process.
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
# BLOCK 10 — EMBED QUERY WITH MXBAI
# =============================================================================
#
# PURPOSE
# -------
# Generate the fixed query embedding for MXBAI.
#
# The same query vector is reused for:
#
#     300 / 0
#     300 / 50
#
# This keeps the query side of the experiment constant.
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
# BLOCK 11 — EMBED QUERY WITH BGE-M3
# =============================================================================
#
# PURPOSE
# -------
# Generate the fixed query embedding for BGE-M3.
#
# The same BGE-M3 query vector is reused for both chunking conditions.
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
# BLOCK 12 — RETRIEVE TOP-5
# =============================================================================
#
# PURPOSE
# -------
# Search a temporary FAISS index and return the five highest-scoring chunks.
#
# FAISS uses IndexFlatIP and all vectors are normalized, therefore:
#
#     score = cosine similarity
#
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
        zip(scores[0], indices[0]),
        start=1
    ):

        if idx == -1:
            continue

        chunk = chunks[
            int(idx)
        ]

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
# Show the actual chunks generated around the relevant "ABSENCE MANAGEMENT"
# content.
#
# This is an inspection aid for the perturbation experiment.
#
# It allows you to visually compare:
#
#     Original corpus text
#     300-character / 0-overlap chunks
#     300-character / 50-overlap chunks
#
# before looking at retrieval behaviour.
# =============================================================================

def print_chunk_structure(
    no_overlap_chunks,
    overlap_chunks
):

    print("\n")
    print("=" * 100)
    print("CHUNK STRUCTURE INSPECTION")
    print("=" * 100)

    print(
        "\n300 CHARACTER / 0 OVERLAP"
    )

    print("-" * 100)

    for chunk in no_overlap_chunks:

        if "absence" in chunk["text"].lower():

            print(
                f"\nChunk ID    : {chunk['chunk_id']}"
            )
            print(
                f"Chunk Index : {chunk['chunk_index']}"
            )
            print(
                f"Characters  : {chunk['start']} - {chunk['end']}"
            )
            print(
                f"Text        : "
                f"{' '.join(chunk['text'].split())}"
            )

    print(
        "\n\n300 CHARACTER / 50 OVERLAP"
    )

    print("-" * 100)

    for chunk in overlap_chunks:

        if "absence" in chunk["text"].lower():

            print(
                f"\nChunk ID    : {chunk['chunk_id']}"
            )
            print(
                f"Chunk Index : {chunk['chunk_index']}"
            )
            print(
                f"Characters  : {chunk['start']} - {chunk['end']}"
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
# Produce exactly five physical lines for every retrieval condition.
#
# This format is intentionally identical to the format established for the
# previous perturbation experiments.
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
# Execute the complete four-condition experiment.
#
#                    300 / 0          300 / 50
#
# MXBAI              BEFORE           OVERLAP
# BGE-M3             BEFORE           OVERLAP
#
# Each model receives the SAME query in both conditions.
# =============================================================================

def main():

    print("=" * 100)
    print(
        "Semantics @ Suren — "
        "Chunk Boundary Perturbation — Overlap Experiment"
    )
    print("=" * 100)

    print(
        f"\nQuery : {QUERY}"
    )

    print(
        f"Chunk size : {CHUNK_SIZE} characters"
    )

    print(
        f"Conditions : 0 overlap vs 50 character overlap"
    )

    # -------------------------------------------------------------------------
    # STEP 1 — Load original source corpus
    # -------------------------------------------------------------------------

    print(
        "\nLoading source corpus..."
    )

    source_text = load_source_corpus()

    print(
        f"Source corpus loaded: "
        f"{len(source_text)} characters."
    )

    # -------------------------------------------------------------------------
    # STEP 2 — Create 300 / 0 corpus
    # -------------------------------------------------------------------------

    print(
        "\nCreating 300-character / 0-overlap chunks..."
    )

    chunks_no_overlap = create_chunks(
        source_text,
        CHUNK_SIZE,
        NO_OVERLAP
    )

    print(
        f"Created {len(chunks_no_overlap)} chunks."
    )

    # -------------------------------------------------------------------------
    # STEP 3 — Create 300 / 50 corpus
    # -------------------------------------------------------------------------

    print(
        "\nCreating 300-character / 50-overlap chunks..."
    )

    chunks_overlap = create_chunks(
        source_text,
        CHUNK_SIZE,
        OVERLAP_50
    )

    print(
        f"Created {len(chunks_overlap)} chunks."
    )

    # -------------------------------------------------------------------------
    # STEP 4 — Inspect chunk boundaries
    # -------------------------------------------------------------------------

    print_chunk_structure(
        chunks_no_overlap,
        chunks_overlap
    )

    # -------------------------------------------------------------------------
    # STEP 5 — Load BGE-M3
    # -------------------------------------------------------------------------

    print(
        "\n\nLoading BGE-M3..."
    )

    bge_model = SentenceTransformer(
        BGE_MODEL
    )

    # -------------------------------------------------------------------------
    # STEP 6 — MXBAI query embedding
    # -------------------------------------------------------------------------

    print(
        "\nEmbedding fixed query with MXBAI..."
    )

    mxbai_query = embed_mxbai_query(
        QUERY
    )

    # -------------------------------------------------------------------------
    # STEP 7 — MXBAI 300 / 0
    # -------------------------------------------------------------------------

    print(
        "\nBuilding temporary MXBAI "
        "300-character / 0-overlap vector store..."
    )

    mxbai_no_overlap_embeddings = (
        build_mxbai_embeddings(
            chunks_no_overlap
        )
    )

    mxbai_no_overlap_index = (
        build_temporary_index(
            mxbai_no_overlap_embeddings
        )
    )

    mxbai_no_overlap_results = retrieve(
        mxbai_no_overlap_index,
        mxbai_query,
        chunks_no_overlap,
        TOP_K
    )

    print_results(
        "MXBAI — 300 CHARACTERS / 0 OVERLAP",
        mxbai_no_overlap_results
    )

    # -------------------------------------------------------------------------
    # STEP 8 — MXBAI 300 / 50
    # -------------------------------------------------------------------------

    print(
        "\nBuilding temporary MXBAI "
        "300-character / 50-overlap vector store..."
    )

    mxbai_overlap_embeddings = (
        build_mxbai_embeddings(
            chunks_overlap
        )
    )

    mxbai_overlap_index = (
        build_temporary_index(
            mxbai_overlap_embeddings
        )
    )

    mxbai_overlap_results = retrieve(
        mxbai_overlap_index,
        mxbai_query,
        chunks_overlap,
        TOP_K
    )

    print_results(
        "MXBAI — 300 CHARACTERS / 50 OVERLAP",
        mxbai_overlap_results
    )

    # -------------------------------------------------------------------------
    # STEP 9 — BGE-M3 query embedding
    # -------------------------------------------------------------------------

    print(
        "\nEmbedding fixed query with BGE-M3..."
    )

    bge_query = embed_bge_query(
        bge_model,
        QUERY
    )

    # -------------------------------------------------------------------------
    # STEP 10 — BGE-M3 300 / 0
    # -------------------------------------------------------------------------

    print(
        "\nBuilding temporary BGE-M3 "
        "300-character / 0-overlap vector store..."
    )

    bge_no_overlap_embeddings = (
        build_bge_embeddings(
            bge_model,
            chunks_no_overlap
        )
    )

    bge_no_overlap_index = (
        build_temporary_index(
            bge_no_overlap_embeddings
        )
    )

    bge_no_overlap_results = retrieve(
        bge_no_overlap_index,
        bge_query,
        chunks_no_overlap,
        TOP_K
    )

    print_results(
        "BGE-M3 — 300 CHARACTERS / 0 OVERLAP",
        bge_no_overlap_results
    )

    # -------------------------------------------------------------------------
    # STEP 11 — BGE-M3 300 / 50
    # -------------------------------------------------------------------------

    print(
        "\nBuilding temporary BGE-M3 "
        "300-character / 50-overlap vector store..."
    )

    bge_overlap_embeddings = (
        build_bge_embeddings(
            bge_model,
            chunks_overlap
        )
    )

    bge_overlap_index = (
        build_temporary_index(
            bge_overlap_embeddings
        )
    )

    bge_overlap_results = retrieve(
        bge_overlap_index,
        bge_query,
        chunks_overlap,
        TOP_K
    )

    print_results(
        "BGE-M3 — 300 CHARACTERS / 50 OVERLAP",
        bge_overlap_results
    )

    # -------------------------------------------------------------------------
    # STEP 12 — Completion
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("EXPERIMENT COMPLETE")
    print("=" * 100)

    print(
        "\nFour retrieval conditions completed:"
    )

    print(
        "  1. MXBAI  — 300 characters / 0 overlap"
    )

    print(
        "  2. MXBAI  — 300 characters / 50 overlap"
    )

    print(
        "  3. BGE-M3 — 300 characters / 0 overlap"
    )

    print(
        "  4. BGE-M3 — 300 characters / 50 overlap"
    )

    print(
        "\nExisting 20260424_B01 vector store was not modified."
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
# Run the experiment when this file is executed directly.
# =============================================================================

if __name__ == "__main__":
    main()