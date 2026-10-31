# =============================================================================
# Semantics @ Suren — Module 3
# Baseline Top-5 Retrieval from B01 Embeddings
#
# Purpose:
#   Retrieve the top-5 most similar B01 chunks for selected questions.
#
# Data source:
#   20260424_B01/embeddings.npy
#
# No dependency on:
#   - baseline_150_results.txt
#   - raGROW retrieval code
#   - FAISS index
#
# The .npy file is used directly for similarity calculation.
# JSON files are used only to identify the chunks and display metadata.
# =============================================================================


from pathlib import Path
import json
import numpy as np
import requests


# =============================================================================
# BLOCK 1 — CONFIGURATION
# =============================================================================

# Questions to test.
# Change this list whenever you want to test different question numbers.

QUESTIONS_TO_TEST = [
    "Q1",
    "Q39",
]

TOP_K = 5

# Same Ollama embedding model used to create the B01 embeddings.
EMBEDDING_MODEL = "mxbai-embed-large:335m"

OLLAMA_URL = "http://localhost:11434/api/embeddings"


# =============================================================================
# BLOCK 2 — DIRECTORY STRUCTURE
# =============================================================================

# This script is located in:
#
# S@S_Module 3 Experiments/Codes_Module_3/
#
# Therefore:
#
# .parent       -> Codes_Module_3
# .parent.parent -> S@S_Module 3 Experiments

BASE_DIR = Path(__file__).resolve().parent.parent


# =============================================================================
# BLOCK 3 — B01 FILE LOCATIONS
# =============================================================================

B01_DIR = BASE_DIR / "20260424_B01"

EMBEDDINGS_FILE = B01_DIR / "embeddings.npy"

INDEX_MAP_FILE = B01_DIR / "index_to_chunk_id.json"

METADATA_FILE = B01_DIR / "chunk_id_to_metadata.json"


# =============================================================================
# BLOCK 4 — QUESTIONS
# =============================================================================

# These are the questions corresponding to the question numbers above.
#
# IMPORTANT:
#   The question text must be entered here because the embedding calculation
#   requires the actual query text.

QUESTIONS = {

    "Q1":
        "How should I notify my manager when I am unable to attend work?",

    "Q39":
        "What if my manager only tried to contact me once?",

}


# =============================================================================
# BLOCK 5 — VALIDATE INPUT FILES
# =============================================================================

print("=" * 100)
print("Semantics @ Suren — Module 3")
print("B01 TOP-5 RETRIEVAL USING EMBEDDINGS.NPY")
print("=" * 100)

print("\nB01 directory:")
print(B01_DIR)

print("\nEmbedding file:")
print(EMBEDDINGS_FILE)

print("\nIndex mapping:")
print(INDEX_MAP_FILE)

print("\nMetadata file:")
print(METADATA_FILE)


required_files = [
    EMBEDDINGS_FILE,
    INDEX_MAP_FILE,
    METADATA_FILE,
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file_path}"
        )


# =============================================================================
# BLOCK 6 — LOAD EMBEDDINGS
# =============================================================================

print("\n" + "=" * 100)
print("LOADING B01 EMBEDDINGS")
print("=" * 100)

embeddings = np.load(EMBEDDINGS_FILE)

print(f"\nEmbedding shape : {embeddings.shape}")
print(f"Number of vectors: {embeddings.shape[0]}")
print(f"Dimensions       : {embeddings.shape[1]}")


# =============================================================================
# BLOCK 7 — LOAD CHUNK MAPPING
# =============================================================================

print("\n" + "=" * 100)
print("LOADING CHUNK MAPPING")
print("=" * 100)

with open(INDEX_MAP_FILE, "r", encoding="utf-8") as f:
    index_to_chunk_id = json.load(f)

print(f"\nIndex mappings loaded : {len(index_to_chunk_id)}")


# =============================================================================
# BLOCK 8 — LOAD METADATA
# =============================================================================

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    chunk_metadata = json.load(f)

print(f"Metadata records loaded: {len(chunk_metadata)}")


# =============================================================================
# BLOCK 9 — EMBEDDING FUNCTION
# =============================================================================

def get_embedding(text):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": EMBEDDING_MODEL,
            "prompt": text
        },
        timeout=120
    )

    response.raise_for_status()

    return np.array(
        response.json()["embedding"],
        dtype=np.float32
    )


# =============================================================================
# BLOCK 10 — NORMALIZE B01 EMBEDDINGS
# =============================================================================

# Cosine similarity:
#
#       cosine(A,B) = A.B / (||A|| ||B||)
#
# We explicitly normalize the vectors here so that the dot product becomes
# cosine similarity.
#
# IMPORTANT:
#   The original .npy file is NOT modified.

embedding_norms = np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

normalized_embeddings = embeddings / embedding_norms


# =============================================================================
# BLOCK 11 — RETRIEVE TOP-K FOR EACH QUESTION
# =============================================================================

for question_id in QUESTIONS_TO_TEST:

    if question_id not in QUESTIONS:

        raise ValueError(
            f"\nQuestion '{question_id}' is not defined in QUESTIONS."
        )

    question_text = QUESTIONS[question_id]

    print("\n\n")
    print("=" * 100)
    print(f"QUESTION {question_id}")
    print("=" * 100)

    print(f"\nQuery:")
    print(question_text)

    # -------------------------------------------------------------------------
    # Generate query embedding
    # -------------------------------------------------------------------------

    print("\nGenerating query embedding...")

    query_embedding = get_embedding(question_text)

    # Normalize query vector
    query_norm = np.linalg.norm(query_embedding)

    query_embedding = (
        query_embedding / query_norm
    )

    # -------------------------------------------------------------------------
    # Calculate cosine similarity against ALL B01 vectors
    # -------------------------------------------------------------------------

    cosine_scores = np.dot(
        normalized_embeddings,
        query_embedding
    )

    # -------------------------------------------------------------------------
    # Rank descending
    # -------------------------------------------------------------------------

    ranked_indices = np.argsort(
        cosine_scores
    )[::-1]

    top_indices = ranked_indices[:TOP_K]

    # -------------------------------------------------------------------------
    # Display results
    # -------------------------------------------------------------------------

    print("\n" + "-" * 100)
    print("TOP 5 MOST SIMILAR CHUNKS")
    print("-" * 100)

    for rank, vector_index in enumerate(top_indices, start=1):

        cosine_score = cosine_scores[vector_index]

        # index_to_chunk_id may have string keys
        chunk_id = index_to_chunk_id.get(
            str(vector_index),
            index_to_chunk_id.get(
                vector_index,
                f"INDEX_{vector_index}"
            )
        )

        metadata = chunk_metadata.get(
            chunk_id,
            {}
        )

        print(f"\nRANK {rank}")
        print("-" * 100)

        print(f"Vector Index : {vector_index}")
        print(f"Chunk ID      : {chunk_id}")
        print(f"Cosine Score  : {cosine_score:.6f}")

        if isinstance(metadata, dict):

            for key, value in metadata.items():

                print(f"{key:<15}: {value}")

        else:

            print(f"Metadata      : {metadata}")


# =============================================================================
# BLOCK 12 — EXPERIMENT COMPLETE
# =============================================================================

print("\n")
print("=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)

print("\nOriginal B01 embeddings were not modified.")
print("No vector store was created.")
print("No mean-centering was applied.")
print("No PCA was applied.")
print("No whitening was applied.")
print("Similarity was calculated directly against the B01 embedding population.")