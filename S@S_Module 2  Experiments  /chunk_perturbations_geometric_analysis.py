"""
===============================================================================
Semantics @ Suren
Experiment: Chunk Perturbation — Geometry Analysis of Chunk 0
===============================================================================

EXPERIMENT DESIGN
-----------------

QUERY IS CONSTANT:

    When can self-certification be used for illness absence?

ONLY CHUNK 0 IS PERTURBED.

ORIGINAL CHUNK 0:
    a           
REVISED CHUNK 0:
    Absence exceeding 2 consecutive working days requires submission of a
    self-declaration form within 24 hours of return.

The objective is to isolate the effect of removing the surrounding
information ("noise") from the chunk while keeping the query unchanged.

MODELS
------
1. mxbai-embed-large:335m
2. BGE-M3

IMPORTANT
---------
This script does NOT modify the existing FAISS vector store.

It creates no persistent vector store.

It simply obtains fresh embeddings temporarily in memory and performs
the requested geometric calculations.

===============================================================================
"""

import math
import requests
import numpy as np


# =============================================================================
# BLOCK 1 — EXPERIMENT INPUT
# =============================================================================

QUERY = (
    "When can self-certification be used for illness absence?"
)


# -----------------------------------------------------------------------------
# Original chunk 0
# -----------------------------------------------------------------------------

ORIGINAL_CHUNK = (
    "Employees must notify their direct manager at least 30 minutes before "
    "their scheduled start time unless physically incapacitated. "
    "Absence exceeding 2 consecutive working days requires submission of a "
    "self-declaration form within 24 hours of return. "
    "Any absence extending beyond 5 consecutive working days must be "
    "supported by medical certification issued no later than the 7th "
    "calendar day of absence."
)


# -----------------------------------------------------------------------------
# Revised chunk 0
# Noise removed — only the relevant rule retained
# -----------------------------------------------------------------------------

REVISED_CHUNK = (
    "Absence exceeding 2 consecutive working days requires submission of a "
    "self-declaration form within 24 hours of return."
)


# =============================================================================
# BLOCK 2 — MODEL CONFIGURATION
# =============================================================================

OLLAMA_URL = "http://localhost:11434/api/embeddings"

MODELS = {
    "MXBAI": "mxbai-embed-large:335m",
    "BGE-M3": "bge-m3",
}


# =============================================================================
# BLOCK 3 — EMBEDDING FUNCTION
# =============================================================================

def get_embedding(text, model):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": text
        },
        timeout=120
    )

    response.raise_for_status()

    embedding = response.json()["embedding"]

    return np.asarray(
        embedding,
        dtype=np.float32
    )


# =============================================================================
# BLOCK 4 — NORMALIZATION
# =============================================================================

def normalize(vector):

    magnitude = np.linalg.norm(vector)

    if magnitude == 0:

        raise ValueError(
            "Zero-norm embedding encountered."
        )

    return vector / magnitude


# =============================================================================
# BLOCK 5 — COSINE CALCULATION
# =============================================================================

def cosine_similarity(vector_a, vector_b):

    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    return float(
        np.dot(vector_a, vector_b)
        / (magnitude_a * magnitude_b)
    )


# =============================================================================
# BLOCK 6 — EUCLIDEAN DISTANCE
# =============================================================================

def euclidean_distance(vector_a, vector_b):

    return float(
        np.linalg.norm(vector_a - vector_b)
    )


# =============================================================================
# BLOCK 7 — MODEL ANALYSIS
# =============================================================================

def analyze_model(model_name, model_id):

    print("\n")
    print("=" * 100)
    print(f"{model_name} — CHUNK PERTURBATION ANALYSIS")
    print("=" * 100)

    print("\nQUERY")
    print("-" * 100)
    print(QUERY)

    print("\nMODEL")
    print("-" * 100)
    print(model_id)


    # -------------------------------------------------------------------------
    # Obtain embeddings
    # -------------------------------------------------------------------------

    print("\nGenerating embeddings...")

    query_vector = get_embedding(
        QUERY,
        model_id
    )

    original_vector = get_embedding(
        ORIGINAL_CHUNK,
        model_id
    )

    revised_vector = get_embedding(
        REVISED_CHUNK,
        model_id
    )


    # -------------------------------------------------------------------------
    # 0. ORIGINAL CHUNK MAGNITUDE
    # -------------------------------------------------------------------------

    original_magnitude = np.linalg.norm(
        original_vector
    )


    # -------------------------------------------------------------------------
    # 1. REVISED CHUNK MAGNITUDE
    # -------------------------------------------------------------------------

    revised_magnitude = np.linalg.norm(
        revised_vector
    )


    # -------------------------------------------------------------------------
    # 2. MAGNITUDE CHANGE
    # -------------------------------------------------------------------------

    magnitude_change = (
        revised_magnitude
        - original_magnitude
    )


    # -------------------------------------------------------------------------
    # 3. COSINE BETWEEN ORIGINAL AND REVISED CHUNKS
    # -------------------------------------------------------------------------

    chunk_cosine = cosine_similarity(
        original_vector,
        revised_vector
    )


    # -------------------------------------------------------------------------
    # 4. EUCLIDEAN DISTANCE BETWEEN ORIGINAL AND REVISED CHUNKS
    #
    # IMPORTANT:
    # This is calculated on the ORIGINAL RAW embeddings.
    # It therefore measures the actual embedding-space displacement.
    # -------------------------------------------------------------------------

    chunk_euclidean_distance = euclidean_distance(
        original_vector,
        revised_vector
    )


    # -------------------------------------------------------------------------
    # Normalized versions of the two chunks
    # -------------------------------------------------------------------------

    original_normalized = normalize(
        original_vector
    )

    revised_normalized = normalize(
        revised_vector
    )

    query_normalized = normalize(
        query_vector
    )


    # -------------------------------------------------------------------------
    # 5. QUERY → ORIGINAL CHUNK COSINE
    # -------------------------------------------------------------------------

    query_original_cosine = cosine_similarity(
        query_normalized,
        original_normalized
    )


    # -------------------------------------------------------------------------
    # 5A. QUERY → REVISED CHUNK COSINE
    # -------------------------------------------------------------------------

    query_revised_cosine = cosine_similarity(
        query_normalized,
        revised_normalized
    )


    # -------------------------------------------------------------------------
    # Additional useful measurement:
    #
    # Change in query-to-chunk cosine
    # -------------------------------------------------------------------------

    query_chunk_cosine_change = (
        query_revised_cosine
        - query_original_cosine
    )


    # -------------------------------------------------------------------------
    # Normalized Euclidean distance between Q and original C
    # -------------------------------------------------------------------------

    query_original_normalized_ed = math.sqrt(
        2 - 2 * query_original_cosine
    )


    # -------------------------------------------------------------------------
    # Normalized Euclidean distance between Q and revised C'
    # -------------------------------------------------------------------------

    query_revised_normalized_ed = math.sqrt(
        2 - 2 * query_revised_cosine
    )


    # =============================================================================
    # BLOCK 8 — PRINT GEOMETRIC RESULTS
    # =============================================================================

    print("\n")
    print("GEOMETRIC RESULTS")
    print("=" * 100)


    print("\n0. ORIGINAL CHUNK MAGNITUDE")
    print(
        f"   ||C|| = {original_magnitude:.6f}"
    )


    print("\n1. REVISED CHUNK MAGNITUDE")
    print(
        f"   ||C'|| = {revised_magnitude:.6f}"
    )


    print("\n2. MAGNITUDE CHANGE")
    print(
        f"   ||C'|| - ||C||"
        f" = {revised_magnitude:.6f}"
        f" - {original_magnitude:.6f}"
        f" = {magnitude_change:.6f}"
    )


    print("\n3. COSINE BETWEEN ORIGINAL AND REVISED CHUNKS")
    print(
        f"   cos(C, C') = {chunk_cosine:.6f}"
    )


    print("\n4. EUCLIDEAN DISTANCE BETWEEN ORIGINAL AND REVISED CHUNKS")
    print(
        f"   ED(C, C') = {chunk_euclidean_distance:.6f}"
    )


    print("\n5. QUERY → ORIGINAL CHUNK COSINE")
    print(
        f"   cos(Q, C) = {query_original_cosine:.6f}"
    )


    print("\n5A. QUERY → REVISED CHUNK COSINE")
    print(
        f"   cos(Q, C') = {query_revised_cosine:.6f}"
    )


    print("\n6. CHANGE IN QUERY → CHUNK COSINE")
    print(
        f"   cos(Q, C') - cos(Q, C)"
        f" = {query_revised_cosine:.6f}"
        f" - {query_original_cosine:.6f}"
        f" = {query_chunk_cosine_change:.6f}"
    )


    print("\n7. NORMALIZED EUCLIDEAN DISTANCE — Q → ORIGINAL C")
    print(
        f"   ED(Q, C) = {query_original_normalized_ed:.6f}"
    )


    print("\n8. NORMALIZED EUCLIDEAN DISTANCE — Q → REVISED C'")
    print(
        f"   ED(Q, C') = {query_revised_normalized_ed:.6f}"
    )


    # =============================================================================
    # BLOCK 9 — COMPACT SUMMARY
    # =============================================================================

    print("\n")
    print("=" * 100)
    print(f"COMPACT SUMMARY — {model_name}")
    print("=" * 100)

    print(
        f"||C||              : {original_magnitude:.6f}"
    )

    print(
        f"||C'||             : {revised_magnitude:.6f}"
    )

    print(
        f"Magnitude Change   : {magnitude_change:.6f}"
    )

    print(
        f"cos(C,C')          : {chunk_cosine:.6f}"
    )

    print(
        f"ED(C,C')           : {chunk_euclidean_distance:.6f}"
    )

    print(
        f"cos(Q,C)           : {query_original_cosine:.6f}"
    )

    print(
        f"cos(Q,C')          : {query_revised_cosine:.6f}"
    )

    print(
        f"Query Cosine Δ     : {query_chunk_cosine_change:.6f}"
    )

    print(
        f"ED(Q,C) normalized : {query_original_normalized_ed:.6f}"
    )

    print(
        f"ED(Q,C') normalized: {query_revised_normalized_ed:.6f}"
    )

    print("=" * 100)


# =============================================================================
# BLOCK 10 — RUN BOTH MODELS
# =============================================================================

def main():

    for model_name, model_id in MODELS.items():

        analyze_model(
            model_name,
            model_id
        )


# =============================================================================
# BLOCK 11 — SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()