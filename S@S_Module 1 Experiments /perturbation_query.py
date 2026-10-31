"""
===============================================================================
Semantics @ Suren — Module 1
Experiment: Query Perturbation Across Embedding Models
File: perturbation_query.py
===============================================================================

EXPERIMENTAL OBJECTIVE
----------------------
Compare the retrieval behaviour of two embedding models when a semantically
important term in the user query is substituted:

    Original:
    "When can self-certification be used for illness absence?"

    Perturbed:
    "When can self-declaration form be used for illness absence?"

The experiment compares:

    1. mxbai-embed-large : original query
    2. mxbai-embed-large : perturbed query
    3. BGE-M3            : original query
    4. BGE-M3            : perturbed query

The purpose is to observe whether the two embedding models respond similarly
or differently to the semantic substitution.

-------------------------------------------------------------------------------
EXPERIMENTAL CONTROL
-------------------------------------------------------------------------------

The corpus and chunking are held CONSTANT.

Both models operate over the SAME 161 chunks represented in:

    20260424_B01/chunk_id_to_metadata.json

Therefore:

    Corpus            = CONSTANT
    Chunk boundaries  = CONSTANT
    Chunk IDs         = CONSTANT
    Query             = VARIABLE
    Embedding model   = VARIABLE

This allows observed retrieval differences to be attributed to the query
perturbation and/or embedding model rather than to differences in corpus
construction.

-------------------------------------------------------------------------------
MXBAI RETRIEVAL PATH
-------------------------------------------------------------------------------

mxbai-embed-large uses the EXISTING B01 FAISS vector store.

The existing Retriever implementation is deliberately reused unchanged.

Therefore:

    Query
      ↓
    mxbai-embed-large:335m
      ↓
    normalized query vector
      ↓
    existing B01 FAISS IndexFlatIP
      ↓
    Top-5 chunks

The existing B01 index was previously validated against the benchmark and
matched the benchmark cosine scores to the last digit.

-------------------------------------------------------------------------------
BGE-M3 RETRIEVAL PATH
-------------------------------------------------------------------------------

There is NO BGE-M3 vector store.

For this experiment only, BGE-M3 embeddings are generated dynamically
in memory from the SAME chunk texts contained in:

    chunk_id_to_metadata.json

The process is:

    chunk text
       ↓
    BGE-M3 embedding
       ↓
    L2 normalization
       ↓
    temporary in-memory embedding matrix

For each query:

    query
       ↓
    BGE-M3 query embedding
       ↓
    L2 normalization
       ↓
    cosine similarity against all chunk embeddings
       ↓
    Top-5 chunks

Nothing is written to or added to the B01 vector store.

-------------------------------------------------------------------------------
SIMILARITY MEASURE
-------------------------------------------------------------------------------

All vectors are L2-normalized before comparison.

For normalized vectors:

    cosine_similarity = dot(query_vector, chunk_vector)

This is equivalent to cosine similarity.

The mxbai path uses FAISS IndexFlatIP, whose inner product therefore represents
cosine similarity after normalization.

The BGE-M3 path explicitly calculates the same normalized dot-product measure.

-------------------------------------------------------------------------------
OUTPUT
-------------------------------------------------------------------------------

Each retrieval condition reports:

    Rank
    Chunk ID
    Chunk Index
    Cosine Score
    Header
    Chunk Text

The output is deliberately verbose because this is an exploratory research
experiment rather than a production retrieval component.

===============================================================================
"""

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# =============================================================================
# BLOCK 1 — PROJECT PATH CONFIGURATION
# =============================================================================
#
# PURPOSE
# -------
# Define the location of the existing Semantics @ Suren project.
#
# IMPORTANT
# ---------
# The parent directory contains a trailing space:
#
#     "S@S_Module 1 Experiments "
#
# This is the ACTUAL directory name on the filesystem and must therefore be
# preserved exactly.
#
# The experimental script itself remains OUTSIDE the original raGROW_S@S code
# directory so that this experiment does not modify or contaminate production
# retrieval code.
# =============================================================================

PROJECT_DIR = Path(
    "/home/surendran/LangChain/Semantics@Suren/"
    "S@S_Module 1 Experiments "
)

RAGROW_DIR = PROJECT_DIR / "raGROW_S@S"

B01_DIR = PROJECT_DIR / "20260424_B01"

CHUNK_METADATA_PATH = B01_DIR / "chunk_id_to_metadata.json"


# =============================================================================
# BLOCK 2 — MAKE THE EXISTING raGROW RETRIEVAL CODE IMPORTABLE
# =============================================================================
#
# PURPOSE
# -------
# Allow this standalone experimental script to reuse the existing Retriever
# without moving the experimental code into the original raGROW_S@S codebase.
#
# We deliberately reuse Retriever for the mxbai experiment because the B01
# baseline has already been validated against the benchmark.
# =============================================================================

sys.path.insert(0, str(RAGROW_DIR))

from rag_engine.retriever import Retriever


# =============================================================================
# BLOCK 3 — EXPERIMENT PARAMETERS
# =============================================================================
#
# PURPOSE
# -------
# Define the four controlled experimental conditions.
#
# QUERY 1
# -------
# The original benchmark query.
#
# QUERY 2
# -------
# The perturbation changes ONLY:
#
#     self-certification
#
# to:
#
#     self-declaration form
#
# Everything else in the query remains unchanged.
#
# This is important because the experiment is intended to isolate the effect
# of this semantic substitution.
# =============================================================================

VERSION_ID = "20260424_B01"

ORIGINAL_QUERY = (
    "When can self-certification be used for illness absence?"
    #"How should I notify my manager when I am unable to attend work?"    
)


PERTURBED_QUERY = (
    #"For illness absence, when can self-certification be used?"
    "When can self-certification be used for illness absence and what is the difference between self certification & medical certificate?"
    #"What is the method of notification to my manager when I am unable to attend work?"
)

TOP_K = 5

BGE_MODEL_NAME = "BAAI/bge-m3"


# =============================================================================
# BLOCK 4 — LOAD THE COMMON CHUNK CORPUS
# =============================================================================
#
# PURPOSE
# -------
# Load the chunk texts that will be used to construct the temporary BGE-M3
# embedding space.
#
# The JSON file is the common semantic corpus for the experiment.
#
# Each entry has the structure:
#
#     chunk_id
#         ├── chunk_index
#         ├── header
#         └── text
#
# The ordering is reconstructed from chunk_index so that the resulting BGE-M3
# embedding matrix has a deterministic correspondence with chunk IDs.
#
# IMPORTANT
# ---------
# We do NOT use embeddings.npy for BGE-M3 because that file belongs to the
# existing mxbai embedding space.
#
# Mixing embeddings from different models would make the similarity calculation
# invalid.
# =============================================================================

def load_chunks():

    if not CHUNK_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Chunk metadata file not found:\n{CHUNK_METADATA_PATH}"
        )

    chunk_metadata = json.loads(
        CHUNK_METADATA_PATH.read_text(encoding="utf-8")
    )

    chunks = []

    for chunk_id, metadata in chunk_metadata.items():

        chunks.append(
            {
                "chunk_id": chunk_id,
                "chunk_index": metadata.get("chunk_index"),
                "header": metadata.get("header", ""),
                "text": metadata.get("text", ""),
            }
        )

    chunks.sort(key=lambda x: x["chunk_index"])

    return chunks


# =============================================================================
# BLOCK 5 — NORMALIZE VECTORS
# =============================================================================
#
# PURPOSE
# -------
# Apply L2 normalization to embeddings before similarity calculation.
#
# After normalization:
#
#     cosine(query, chunk)
#
# becomes equivalent to:
#
#     dot(query, chunk)
#
# This mirrors the mathematical basis of the existing mxbai + FAISS
# IndexFlatIP retrieval path.
#
# A zero-norm vector is explicitly guarded against because normalization of
# such a vector would be undefined.
# =============================================================================

def normalize_vectors(vectors):

    vectors = np.asarray(vectors, dtype=np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise ValueError(
            "One or more embeddings has zero norm."
        )

    return vectors / norms


# =============================================================================
# BLOCK 6 — CREATE TEMPORARY BGE-M3 EMBEDDING SPACE
# =============================================================================
#
# PURPOSE
# -------
# Generate BGE-M3 embeddings for every B01 chunk.
#
# This is deliberately performed at runtime and held only in memory.
#
# NO FILE IS CREATED.
# NO FAISS INDEX IS CREATED.
# NO EXISTING B01 ARTIFACT IS MODIFIED.
#
# The returned embedding matrix has the same row ordering as the `chunks`
# list, allowing:
#
#     embedding row 0 → chunk_0
#     embedding row 1 → chunk_1
#     ...
#
# assuming the supplied B01 chunk metadata follows the known chunk indexing.
# =============================================================================

def build_bge_chunk_embeddings(model, chunks):

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

    embeddings = normalize_vectors(embeddings)

    return embeddings


# =============================================================================
# BLOCK 7 — BGE-M3 QUERY RETRIEVAL
# =============================================================================
#
# PURPOSE
# -------
# Retrieve the Top-5 chunks for a query using the temporary BGE-M3 embedding
# space.
#
# PROCESS
# -------
#
#     Query
#       ↓
#     BGE-M3 query embedding
#       ↓
#     L2 normalization
#       ↓
#     dot product against normalized chunk embeddings
#       ↓
#     descending similarity order
#       ↓
#     Top-5
#
# Because both query and chunk vectors are normalized, the resulting dot
# products are cosine similarities.
# =============================================================================

def retrieve_bge(
    model,
    query,
    chunks,
    chunk_embeddings,
    top_k=5
):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False
    )

    query_embedding = normalize_vectors(query_embedding)[0]

    scores = np.dot(
        chunk_embeddings,
        query_embedding
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, idx in enumerate(top_indices, start=1):

        chunk = chunks[int(idx)]

        results.append(
            {
                "rank": rank,
                "score": float(scores[idx]),
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "header": chunk["header"],
                "text": " ".join(chunk["text"].split()),
            }
        )

    return results

# =============================================================================
# BLOCK 8 — PRINT RESULTS IN COPY/PASTE-FRIENDLY FORMAT
# =============================================================================
#
# PURPOSE
# -------
# Present each retrieved chunk as ONE physical line.
#
# This is intentional: the experimental results are being copied directly into
# the existing Excel analysis template, where each Top-5 retrieval occupies
# one row.
#
# Therefore:
#
#     Rank 1 → one line
#     Rank 2 → one line
#     Rank 3 → one line
#     Rank 4 → one line
#     Rank 5 → one line
#
# The retrieval logic and scores are NOT changed here. This block only controls
# how the already-calculated results are displayed.
# =============================================================================

def print_results(title, results):

    print("\n")
    print("=" * 90)
    print(title)
    print("=" * 90)

    for result in results:

        print(
            f"Rank {result['rank']} | "
            f"Chunk ID: {result['chunk_id']} | "
            f"Chunk Index: {result['chunk_index']} | "
            f"Cosine: {result['score']:.6f} | "
            f"Header: {result.get('header', '')} | "
            f"Text: {result['text']}"
        )

    print("\n" + "-" * 90)



# =============================================================================
# BLOCK 9 — CONVERT EXISTING MXBAI RETRIEVER OUTPUT
# =============================================================================
#
# PURPOSE
# -------
# Adapt the existing Retriever's output into the same result structure used
# by the temporary BGE-M3 retrieval path.
#
# No mxbai retrieval logic is reproduced here.
#
# The existing Retriever remains the authoritative implementation for the
# mxbai side of this experiment.
# =============================================================================

def retrieve_mxbai(retriever, query):

    result = retriever.retrieve(
        query=query,
        top_k=TOP_K,
        score_threshold=0.0,
        return_metadata=True
    )

    if "results" not in result:
        return []

    converted_results = []

    for item in result["results"]:

        metadata = item.get("metadata", {})

        converted_results.append(
            {
                "rank": item["rank"],
                "score": item["score"],
                "chunk_id": item["chunk_id"],
                "chunk_index": item["chunk_index"],
                "header": metadata.get("header", ""),
                "text": item["text"],
            }
        )

    return converted_results


# =============================================================================
# BLOCK 10 — MAIN EXPERIMENT DRIVER
# =============================================================================
#
# PURPOSE
# -------
# Execute the four controlled retrieval conditions in a single run.
#
# EXPERIMENT MATRIX
# -----------------
#
#                     Original Query       Perturbed Query
#
# mxbai                   RUN                    RUN
# BGE-M3                  RUN                    RUN
#
# The mxbai Retriever is loaded from the existing B01 implementation.
#
# BGE-M3 is loaded separately and creates a temporary in-memory embedding
# representation of the same 161 chunks.
# =============================================================================

def main():

    print("=" * 90)
    print("Semantics @ Suren — Query Perturbation Experiment")
    print("=" * 90)

    print(f"\nVersion        : {VERSION_ID}")
    print(f"Original Query : {ORIGINAL_QUERY}")
    print(f"Perturbed Query: {PERTURBED_QUERY}")
    print(f"BGE-M3 Model   : {BGE_MODEL_NAME}")
    print(f"Top-K          : {TOP_K}")

    # -------------------------------------------------------------------------
    # STEP 1 — Existing mxbai Retriever
    # -------------------------------------------------------------------------
    #
    # The existing B01 FAISS index and Retriever implementation are reused
    # without modification.
    # -------------------------------------------------------------------------

    print("\nLoading existing mxbai B01 Retriever...")

    mxbai_retriever = Retriever(
        version_id=VERSION_ID
    )

    # -------------------------------------------------------------------------
    # STEP 2 — mxbai / Original Query
    # -------------------------------------------------------------------------

    mxbai_original = retrieve_mxbai(
        mxbai_retriever,
        ORIGINAL_QUERY
    )

    print_results(
        "MXBAI — ORIGINAL QUERY",
        mxbai_original
    )

    # -------------------------------------------------------------------------
    # STEP 3 — mxbai / Perturbed Query
    # -------------------------------------------------------------------------

    mxbai_perturbed = retrieve_mxbai(
        mxbai_retriever,
        PERTURBED_QUERY
    )

    print_results(
        "MXBAI — PERTURBED QUERY",
        mxbai_perturbed
    )

    # -------------------------------------------------------------------------
    # STEP 4 — Load BGE-M3
    # -------------------------------------------------------------------------
    #
    # This model is used only for the experimental comparison.
    # No BGE-M3 vector store is created.
    # -------------------------------------------------------------------------

    print("\nLoading BGE-M3 model...")
    print(f"Model: {BGE_MODEL_NAME}")

    bge_model = SentenceTransformer(
        BGE_MODEL_NAME
    )

    # -------------------------------------------------------------------------
    # STEP 5 — Load the SAME B01 chunks
    # -------------------------------------------------------------------------

    print("\nLoading common B01 chunk corpus...")

    chunks = load_chunks()

    print(f"Number of chunks loaded: {len(chunks)}")

    # -------------------------------------------------------------------------
    # STEP 6 — Build temporary BGE-M3 chunk embeddings
    # -------------------------------------------------------------------------

    print("\nGenerating temporary BGE-M3 embeddings for B01 chunks...")

    bge_chunk_embeddings = build_bge_chunk_embeddings(
        bge_model,
        chunks
    )

    print(
        f"BGE-M3 embedding matrix shape: "
        f"{bge_chunk_embeddings.shape}"
    )

    # -------------------------------------------------------------------------
    # STEP 7 — BGE-M3 / Original Query
    # -------------------------------------------------------------------------

    bge_original = retrieve_bge(
        model=bge_model,
        query=ORIGINAL_QUERY,
        chunks=chunks,
        chunk_embeddings=bge_chunk_embeddings,
        top_k=TOP_K
    )

    print_results(
        "BGE-M3 — ORIGINAL QUERY",
        bge_original
    )

    # -------------------------------------------------------------------------
    # STEP 8 — BGE-M3 / Perturbed Query
    # -------------------------------------------------------------------------

    bge_perturbed = retrieve_bge(
        model=bge_model,
        query=PERTURBED_QUERY,
        chunks=chunks,
        chunk_embeddings=bge_chunk_embeddings,
        top_k=TOP_K
    )

    print_results(
        "BGE-M3 — PERTURBED QUERY",
        bge_perturbed
    )

    # -------------------------------------------------------------------------
    # STEP 9 — Experimental Completion Summary
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("EXPERIMENT COMPLETE")
    print("=" * 90)

    print("\nFour retrieval conditions completed:")
    print("  1. mxbai  — original query")
    print("  2. mxbai  — perturbed query")
    print("  3. BGE-M3 — original query")
    print("  4. BGE-M3 — perturbed query")

    print("\nExisting B01 vector store was not modified.")
    print("BGE-M3 embeddings were generated only in memory.")
    print("=" * 90)


# =============================================================================
# BLOCK 11 — SCRIPT ENTRY POINT
# =============================================================================
#
# PURPOSE
# -------
# Ensure that the experiment runs only when this file is executed directly.
#
# This also allows the functions above to be imported later into another
# exploratory analysis script without automatically executing the experiment.
# =============================================================================

if __name__ == "__main__":
    main()