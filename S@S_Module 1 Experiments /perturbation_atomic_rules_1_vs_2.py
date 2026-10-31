"""
===============================================================================
Semantics @ Suren — Module 1
Experiment: Atomic Rule vs Two-Rule Retrieval
File: perturbation_atomic_rules_1_vs_2.py
===============================================================================

EXPERIMENTAL OBJECTIVE
----------------------
Study the effect of chunk granularity on dense retrieval.

Two corpus representations are created from the SAME source data:

    CONDITION A
    -----------
    One complete rule/sentence per chunk.

    CONDITION B
    -----------
    Two consecutive rules/sentences per chunk.

Both representations are tested with:

    1. MXBAI
    2. BGE-M3

Therefore, four independent retrieval experiments are performed:

    MXBAI  + Single Rule
    MXBAI  + Two Rules

    BGE-M3 + Single Rule
    BGE-M3 + Two Rules

-------------------------------------------------------------------------------
SOURCE
-------------------------------------------------------------------------------

ODS file:

    HR_corpus_1.ods

Column C contains the actual rule/sentence text.

Column C is therefore treated as the SOURCE OF TRUTH for this experiment.

Headers and other metadata are not used to construct the embedding text.

-------------------------------------------------------------------------------
FIXED QUERY
-------------------------------------------------------------------------------

"When can self-certification be used for illness absence?"

-------------------------------------------------------------------------------
CONDITION A — SINGLE ATOMIC RULE
-------------------------------------------------------------------------------

Each complete sentence/rule from Column C becomes one retrieval chunk.

Example:

    Rule A.
    Rule B.
    Rule C.

becomes:

    atomic_0 = Rule A.
    atomic_1 = Rule B.
    atomic_2 = Rule C.

-------------------------------------------------------------------------------
CONDITION B — TWO RULES CLUBBED
-------------------------------------------------------------------------------

Two consecutive rules are combined into one retrieval chunk.

Example:

    Rule A.
    Rule B.
    Rule C.
    Rule D.

becomes:

    two_rule_0 = Rule A. Rule B.
    two_rule_1 = Rule C. Rule D.

There is NO overlap.

-------------------------------------------------------------------------------
EMBEDDING
-------------------------------------------------------------------------------

MXBAI:

    mxbai-embed-large:335m

    Generated through Ollama.

BGE-M3:

    BAAI/bge-m3

    Generated through SentenceTransformers.

-------------------------------------------------------------------------------
VECTOR STORE
-------------------------------------------------------------------------------

Temporary in-memory FAISS IndexFlatIP stores are created.

Nothing is written to disk.

The existing 20260424_B01 vector store is NOT modified.

-------------------------------------------------------------------------------
SIMILARITY
-------------------------------------------------------------------------------

All vectors are L2 normalized.

FAISS Inner Product therefore represents cosine similarity.

-------------------------------------------------------------------------------
OUTPUT
-------------------------------------------------------------------------------

Four Top-5 result sets are printed.

Every result is deliberately printed on one line:

    Rank | Chunk ID | Chunk Index | Cosine | Text

This makes the output easy to copy into the Excel analysis template.

===============================================================================
"""


# =============================================================================
# BLOCK 1 — IMPORTS
# =============================================================================
#
# PURPOSE
# -------
# Import the libraries required for:
#
#     - reading the ODS file
#     - MXBAI embedding through Ollama
#     - BGE-M3 embedding
#     - FAISS vector search
#     - vector normalization
#
# =============================================================================

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import requests

from sentence_transformers import SentenceTransformer


# =============================================================================
# BLOCK 2 — EXPERIMENT CONFIGURATION
# =============================================================================
#
# PURPOSE
# -------
# Define all fixed parameters for the experiment.
#
# IMPORTANT
# ---------
# Nothing here points to the existing B01 vector store.
#
# The ODS is the only corpus source.
# =============================================================================

PROJECT_DIR = Path(
    "/home/surendran/LangChain/Semantics@Suren/"
    "S@S_Module 1 Experiments "
)

CORPUS_PATH = (
    PROJECT_DIR / "HR_corpus_1.ods"
)

QUERY = (
    "When can self-certification be used for illness absence?"
)

TOP_K = 5

# MXBAI through Ollama
MXBAI_MODEL = "mxbai-embed-large:335m"

OLLAMA_URL = (
    "http://localhost:11434/api/embeddings"
)

# BGE-M3 through SentenceTransformers
BGE_MODEL = "BAAI/bge-m3"


# =============================================================================
# BLOCK 3 — LOAD COLUMN C FROM ODS
# =============================================================================
#
# PURPOSE
# -------
# Read HR_corpus_1.ods and extract ONLY Column C.
#
# Pandas uses zero-based column numbering:
#
#     Column A → index 0
#     Column B → index 1
#     Column C → index 2
#
# Therefore df.iloc[:, 2] is Column C.
#
# No other column is used for constructing retrieval text.
# =============================================================================

def load_column_c():

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus file not found:\n{CORPUS_PATH}"
        )

    print(
        f"Loading corpus from:\n{CORPUS_PATH}"
    )

    df = pd.read_excel(
        CORPUS_PATH,
        engine="odf",
        header=None
    )

    if df.shape[1] < 3:
        raise ValueError(
            "The ODS file does not contain Column C."
        )

    # -------------------------------------------------------------------------
    # Column C
    # -------------------------------------------------------------------------

    column_c = df.iloc[:, 2]

    rules = []

    for value in column_c:

        if pd.isna(value):
            continue

        text = str(value).strip()

        if not text:
            continue

        rules.append(text)

    if not rules:
        raise ValueError(
            "No rule/sentence text found in Column C."
        )

    print(
        f"Loaded {len(rules)} non-empty entries from Column C."
    )

    return rules


# =============================================================================
# BLOCK 4 — CREATE SINGLE-RULE REPRESENTATION
# =============================================================================
#
# PURPOSE
# -------
# Convert each complete sentence/rule from Column C into one retrieval unit.
#
# IMPORTANT
# ---------
# The source text is NOT augmented with headers or metadata.
#
# Each rule remains exactly as supplied in Column C, except for whitespace
# normalization.
# =============================================================================

def create_atomic_rules(rules):

    atomic_rules = []

    for index, rule in enumerate(rules):

        text = " ".join(
            rule.split()
        )

        atomic_rules.append(
            {
                "chunk_id": f"atomic_{index}",
                "chunk_index": index,
                "text": text,
            }
        )

    return atomic_rules


# =============================================================================
# BLOCK 5 — CREATE TWO-RULE REPRESENTATION
# =============================================================================
#
# PURPOSE
# -------
# Combine two consecutive Column-C rules into one retrieval unit.
#
# NO OVERLAP
# ----------
#
#     Rule 0 + Rule 1
#     Rule 2 + Rule 3
#     Rule 4 + Rule 5
#     ...
#
# If the corpus contains an odd number of rules, the final rule is retained
# as a single final chunk.
# =============================================================================

def create_two_rule_chunks(rules):

    chunks = []

    chunk_index = 0

    for i in range(
        0,
        len(rules),
        2
    ):

        first_rule = " ".join(
            rules[i].split()
        )

        if i + 1 < len(rules):

            second_rule = " ".join(
                rules[i + 1].split()
            )

            combined_text = (
                first_rule
                + " "
                + second_rule
            )

        else:

            combined_text = first_rule

        chunks.append(
            {
                "chunk_id": f"two_rule_{chunk_index}",
                "chunk_index": chunk_index,
                "text": combined_text,
            }
        )

        chunk_index += 1

    return chunks


# =============================================================================
# BLOCK 6 — NORMALIZE VECTORS
# =============================================================================
#
# PURPOSE
# -------
# L2-normalize embeddings.
#
# This allows FAISS IndexFlatIP to produce cosine similarity:
#
#     cosine(q, x) = normalized(q) · normalized(x)
#
# =============================================================================

def normalize_embeddings(
    embeddings
):

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
            "Zero-norm embedding encountered."
        )

    return embeddings / norms


# =============================================================================
# BLOCK 7 — MXBAI EMBEDDING FUNCTION
# =============================================================================
#
# PURPOSE
# -------
# Generate an embedding using the locally running Ollama MXBAI model.
# =============================================================================

def embed_mxbai(text):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MXBAI_MODEL,
            "prompt": text
        },
        timeout=60
    )

    response.raise_for_status()

    embedding = response.json()["embedding"]

    return np.asarray(
        embedding,
        dtype=np.float32
    )


# =============================================================================
# BLOCK 8 — BUILD MXBAI CORPUS EMBEDDINGS
# =============================================================================
#
# PURPOSE
# -------
# Generate MXBAI embeddings for every retrieval chunk.
#
# The input can be either:
#
#     - atomic_rules
#     - two_rule_chunks
#
# =============================================================================

def build_mxbai_embeddings(chunks):

    embeddings = []

    total = len(chunks)

    for position, chunk in enumerate(
        chunks,
        start=1
    ):

        print(
            f"MXBAI embedding {position}/{total}",
            end="\r"
        )

        vector = embed_mxbai(
            chunk["text"]
        )

        embeddings.append(
            vector
        )

    print()

    embeddings = np.vstack(
        embeddings
    )

    return normalize_embeddings(
        embeddings
    )


# =============================================================================
# BLOCK 9 — LOAD BGE-M3
# =============================================================================
#
# PURPOSE
# -------
# Load BGE-M3 once.
#
# The same model instance is reused for all BGE-M3 experiments.
# =============================================================================

def load_bge_model():

    print(
        "\nLoading BGE-M3..."
    )

    model = SentenceTransformer(
        BGE_MODEL
    )

    return model


# =============================================================================
# BLOCK 10 — BUILD BGE-M3 CORPUS EMBEDDINGS
# =============================================================================
#
# PURPOSE
# -------
# Generate BGE-M3 embeddings for the supplied retrieval representation.
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
# BLOCK 11 — BUILD TEMPORARY FAISS STORE
# =============================================================================
#
# PURPOSE
# -------
# Create an in-memory FAISS vector index.
#
# IMPORTANT
# ---------
# No .index file is created.
#
# The index disappears when the script terminates.
# =============================================================================

def build_temporary_vector_store(
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
# BLOCK 12 — EMBED QUERY WITH MXBAI
# =============================================================================

def embed_mxbai_query(query):

    vector = embed_mxbai(
        query
    )

    vector = vector.reshape(
        1,
        -1
    )

    return normalize_embeddings(
        vector
    )


# =============================================================================
# BLOCK 13 — EMBED QUERY WITH BGE-M3
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
# BLOCK 14 — RETRIEVE TOP-K
# =============================================================================
#
# PURPOSE
# -------
# Search the temporary FAISS index and return the Top-K chunks.
# =============================================================================

def retrieve(
    index,
    query_vector,
    chunks,
    top_k
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

        chunk = chunks[
            int(idx)
        ]

        results.append(
            {
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "score": float(score),
                "text": " ".join(
                    chunk["text"].split()
                )
            }
        )

    return results


# =============================================================================
# BLOCK 15 — PRINT COPY-PASTE OUTPUT
# =============================================================================
#
# PURPOSE
# -------
# Print each retrieval result as a single line.
#
# This is intentionally compatible with the Excel analysis workflow.
#
# NO HEADER IS PRINTED.
# =============================================================================

def print_results(
    title,
    results
):

    print("\n")
    print("=" * 120)
    print(title)
    print("=" * 120)

    print(
        f"Query: {QUERY}"
    )

    print("-" * 120)

    for result in results:

        print(
            f"Rank {result['rank']} | "
            f"Chunk ID: {result['chunk_id']} | "
            f"Chunk Index: {result['chunk_index']} | "
            f"Cosine: {result['score']:.6f} | "
            f"Text: {result['text']}"
        )

    print("-" * 120)


# =============================================================================
# BLOCK 16 — MAIN EXPERIMENT
# =============================================================================
#
# PURPOSE
# -------
# Execute all four combinations:
#
#     1. MXBAI  + Atomic
#     2. MXBAI  + Two Rules
#     3. BGE-M3 + Atomic
#     4. BGE-M3 + Two Rules
#
# Every vector store is temporary and exists only in memory.
# =============================================================================

def main():

    print("=" * 120)

    print(
        "Semantics @ Suren — "
        "Atomic Rule vs Two-Rule Retrieval Experiment"
    )

    print("=" * 120)

    print(
        f"\nQuery   : {QUERY}"
    )

    print(
        f"Top-K   : {TOP_K}"
    )

    print(
        f"Source  : {CORPUS_PATH}"
    )

    print(
        "\nExisting vector stores will NOT be modified."
    )

    print(
        "Temporary vector stores are created only in memory."
    )

    # =========================================================================
    # LOAD SOURCE
    # =========================================================================

    rules = load_column_c()

    # =========================================================================
    # CREATE REPRESENTATION A
    # =========================================================================

    print(
        "\nCreating single-rule representation..."
    )

    atomic_rules = create_atomic_rules(
        rules
    )

    print(
        f"Single-rule chunks: {len(atomic_rules)}"
    )

    # =========================================================================
    # CREATE REPRESENTATION B
    # =========================================================================

    print(
        "\nCreating two-rule representation..."
    )

    two_rule_chunks = create_two_rule_chunks(
        rules
    )

    print(
        f"Two-rule chunks: {len(two_rule_chunks)}"
    )

    # =========================================================================
    # LOAD BGE-M3
    # =========================================================================

    bge_model = load_bge_model()

    # =========================================================================
    # MXBAI QUERY
    # =========================================================================

    print(
        "\nEmbedding query with MXBAI..."
    )

    mxbai_query_vector = embed_mxbai_query(
        QUERY
    )

    # =========================================================================
    # MXBAI — SINGLE RULE
    # =========================================================================

    print(
        "\nBuilding temporary MXBAI "
        "single-rule vector store..."
    )

    mxbai_atomic_embeddings = (
        build_mxbai_embeddings(
            atomic_rules
        )
    )

    mxbai_atomic_index = (
        build_temporary_vector_store(
            mxbai_atomic_embeddings
        )
    )

    mxbai_atomic_results = retrieve(
        mxbai_atomic_index,
        mxbai_query_vector,
        atomic_rules,
        TOP_K
    )

    print_results(
        "MXBAI — SINGLE ATOMIC RULE",
        mxbai_atomic_results
    )

    # =========================================================================
    # MXBAI — TWO RULES
    # =========================================================================

    print(
        "\nBuilding temporary MXBAI "
        "two-rule vector store..."
    )

    mxbai_two_rule_embeddings = (
        build_mxbai_embeddings(
            two_rule_chunks
        )
    )

    mxbai_two_rule_index = (
        build_temporary_vector_store(
            mxbai_two_rule_embeddings
        )
    )

    mxbai_two_rule_results = retrieve(
        mxbai_two_rule_index,
        mxbai_query_vector,
        two_rule_chunks,
        TOP_K
    )

    print_results(
        "MXBAI — TWO CONSECUTIVE RULES",
        mxbai_two_rule_results
    )

    # =========================================================================
    # BGE-M3 QUERY
    # =========================================================================

    print(
        "\nEmbedding query with BGE-M3..."
    )

    bge_query_vector = embed_bge_query(
        bge_model,
        QUERY
    )

    # =========================================================================
    # BGE-M3 — SINGLE RULE
    # =========================================================================

    print(
        "\nBuilding temporary BGE-M3 "
        "single-rule vector store..."
    )

    bge_atomic_embeddings = (
        build_bge_embeddings(
            bge_model,
            atomic_rules
        )
    )

    bge_atomic_index = (
        build_temporary_vector_store(
            bge_atomic_embeddings
        )
    )

    bge_atomic_results = retrieve(
        bge_atomic_index,
        bge_query_vector,
        atomic_rules,
        TOP_K
    )

    print_results(
        "BGE-M3 — SINGLE ATOMIC RULE",
        bge_atomic_results
    )

    # =========================================================================
    # BGE-M3 — TWO RULES
    # =========================================================================

    print(
        "\nBuilding temporary BGE-M3 "
        "two-rule vector store..."
    )

    bge_two_rule_embeddings = (
        build_bge_embeddings(
            bge_model,
            two_rule_chunks
        )
    )

    bge_two_rule_index = (
        build_temporary_vector_store(
            bge_two_rule_embeddings
        )
    )

    bge_two_rule_results = retrieve(
        bge_two_rule_index,
        bge_query_vector,
        two_rule_chunks,
        TOP_K
    )

    print_results(
        "BGE-M3 — TWO CONSECUTIVE RULES",
        bge_two_rule_results
    )

    # =========================================================================
    # COMPLETION
    # =========================================================================

    print("\n")
    print("=" * 120)
    print("EXPERIMENT COMPLETE")
    print("=" * 120)

    print(
        "\nFour retrieval conditions completed:"
    )

    print(
        "  1. MXBAI  — Single Atomic Rule"
    )

    print(
        "  2. MXBAI  — Two Consecutive Rules"
    )

    print(
        "  3. BGE-M3 — Single Atomic Rule"
    )

    print(
        "  4. BGE-M3 — Two Consecutive Rules"
    )

    print(
        "\nNo existing vector store was modified."
    )

    print(
        "No temporary vector store was persisted."
    )

    print("=" * 120)


# =============================================================================
# BLOCK 17 — SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()