# =============================================================================
# Semantics @ Suren — Module 3
#
# TOP-5 RETRIEVAL GEOMETRY ANALYSIS
#
# Uses:
#   1. Existing Query -> Top-5 cosine scores supplied from B01 retrieval
#   2. B01 embeddings.npy for pairwise geometry
#   3. B01 metadata files for chunk identification / traceability
#
# No new retrieval is performed.
# No mean-centering.
# No PCA.
# No whitening.
# No dimensionality reduction.
#
# =============================================================================

from pathlib import Path
import json
import numpy as np


# =============================================================================
# BLOCK 1 — QUESTIONS AND EXISTING TOP-5 RETRIEVAL RESULTS
# =============================================================================

RETRIEVAL_RESULTS = {

    "98": {
        "question":
            "When must compliance training be completed? ",

        "chunk_ids": [
            "chunk_44",
            "chunk_95",
            "chunk_13",
            "chunk_96",
            "chunk_144"
        ],

        "scores": [
        

            0.765052, 
            0.745572, 
            0.740892, 
            0.72703, 
            0.685008
        ]
    },

    "186": {
        "question":
            "How many working days in advance must employees submit leave "
            "requests for up to five days of leave?",

        "chunk_ids": [
            "chunk_6",
            "chunk_8",
            "chunk_7",
            "chunk_0",
            "chunk_3"
        ],

        "scores": [
            0.901862,
            0.727107,
            0.712832,
            0.712246,
            0.709766
        ]
    },

    "246": {
        "question":
            "Which header contains the travel reimbursement rules?",

        "chunk_ids": [
            "chunk_46",
            "chunk_76",
            "chunk_141",
            "chunk_14",
            "chunk_160"
        ],

        "scores": [
            0.536907,
            0.535686,
            0.532903,
            0.532021,
            0.523386
        ]
    }
}


# =============================================================================
# BLOCK 2 — CORRECT CHUNK IDs
# =============================================================================
#
# IMPORTANT:
#
# Enter the known correct chunk ID for each question here.
#
# I have NOT guessed these values.
#
# Example:
#
# CORRECT_CHUNKS = {
#     "177": "chunk_2",
#     "186": "chunk_6",
#     "246": "chunk_46"
# }
#
# Replace the values below with the actual correct chunk IDs.
#
# =============================================================================

CORRECT_CHUNKS = {

    "177": None,
    "186": None,
    "246": None

}


# =============================================================================
# BLOCK 3 — B01 VECTOR STORE LOCATION
# =============================================================================

# This script is located in:
#
# S@S_Module 3 Experiments/Codes_Module_3/
#
# .parent.parent
#       ↓
# S@S_Module 3 Experiments/
#
# Therefore the B01 vector store is:
#
# S@S_Module 3 Experiments/20260424_B01/

BASE_DIR = Path(__file__).resolve().parent.parent

B01_DIR = BASE_DIR / "20260424_B01"

EMBEDDINGS_FILE = B01_DIR / "embeddings.npy"

INDEX_MAP_FILE = B01_DIR / "index_to_chunk_id.json"

METADATA_FILE = B01_DIR / "chunk_id_to_metadata.json"


# =============================================================================
# BLOCK 4 — LOAD VECTOR STORE FILES
# =============================================================================

print("=" * 100)
print("Semantics @ Suren — Module 3")
print("TOP-5 RETRIEVAL GEOMETRY ANALYSIS")
print("=" * 100)

print("\nB01 Vector Store")
print("-" * 100)

print(f"Directory  : {B01_DIR}")
print(f"Embeddings  : {EMBEDDINGS_FILE}")
print(f"Index map   : {INDEX_MAP_FILE}")
print(f"Metadata    : {METADATA_FILE}")


for file_path in [
    EMBEDDINGS_FILE,
    INDEX_MAP_FILE,
    METADATA_FILE
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nRequired B01 vector-store file not found:\n{file_path}"
        )


# =============================================================================
# BLOCK 5 — LOAD EMBEDDINGS
# =============================================================================

embeddings = np.load(EMBEDDINGS_FILE)

print("\n")
print("=" * 100)
print("EMBEDDING MATRIX")
print("=" * 100)

print(f"\nShape       : {embeddings.shape}")
print(f"Vectors     : {embeddings.shape[0]}")
print(f"Dimensions  : {embeddings.shape[1]}")


# =============================================================================
# BLOCK 6 — LOAD INDEX → CHUNK ID MAPPING
# =============================================================================

with open(INDEX_MAP_FILE, "r", encoding="utf-8") as f:

    index_to_chunk_id = json.load(f)


# =============================================================================
# BLOCK 7 — LOAD CHUNK METADATA
# =============================================================================

with open(METADATA_FILE, "r", encoding="utf-8") as f:

    chunk_id_to_metadata = json.load(f)


# =============================================================================
# BLOCK 8 — CREATE CHUNK ID → VECTOR INDEX MAPPING
# =============================================================================

chunk_to_index = {}

for index, chunk_id in index_to_chunk_id.items():

    chunk_to_index[str(chunk_id)] = int(index)


# =============================================================================
# BLOCK 9 — NORMALIZE EMBEDDINGS FOR COSINE CALCULATION
# =============================================================================
#
# The original embeddings.npy is NOT modified.
#
# A separate normalized copy is created in memory.
#
# =============================================================================

norms = np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

normalized_embeddings = embeddings / norms


# =============================================================================
# BLOCK 10 — COSINE MATRIX FUNCTION
# =============================================================================

def cosine_matrix(vectors):

    """
    Calculate the pairwise cosine similarity matrix
    for the supplied vectors.
    """

    vector_norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    normalized = vectors / vector_norms

    return np.matmul(
        normalized,
        normalized.T
    )


# =============================================================================
# BLOCK 11 — MAIN ANALYSIS
# =============================================================================

for question_id, data in RETRIEVAL_RESULTS.items():

    question = data["question"]

    chunk_ids = data["chunk_ids"]

    query_scores = np.array(
        data["scores"],
        dtype=float
    )


    # =========================================================================
    # BASIC VALIDATION
    # =========================================================================

    if len(chunk_ids) != 5:

        raise ValueError(
            f"Question {question_id} does not contain exactly 5 chunks."
        )

    if len(query_scores) != 5:

        raise ValueError(
            f"Question {question_id} does not contain exactly 5 scores."
        )


    # =========================================================================
    # GET VECTOR INDICES
    # =========================================================================

    vector_indices = []

    for chunk_id in chunk_ids:

        if chunk_id not in chunk_to_index:

            raise KeyError(
                f"\n{chunk_id} not found in B01 index mapping."
            )

        vector_indices.append(
            chunk_to_index[chunk_id]
        )


    vector_indices = np.array(
        vector_indices,
        dtype=int
    )


    # =========================================================================
    # EXTRACT TOP-5 VECTORS
    # =========================================================================

    top5_vectors = embeddings[
        vector_indices
    ]


    # =========================================================================
    # PAIRWISE COSINE MATRIX
    # =========================================================================

    pairwise_matrix = cosine_matrix(
        top5_vectors
    )


    # =========================================================================
    # CONSECUTIVE SCORE GAPS
    # =========================================================================

    gaps = []

    for i in range(4):

        gap = query_scores[i] - query_scores[i + 1]

        gaps.append(gap)


    # =========================================================================
    # CORRECT CHUNK ANALYSIS
    # =========================================================================

    correct_chunk = CORRECT_CHUNKS.get(
        question_id
    )

    correct_rank = None
    correct_query_score = None
    correct_vector_position = None

    if correct_chunk is not None:

        if correct_chunk in chunk_ids:

            correct_rank = chunk_ids.index(correct_chunk) + 1

            correct_query_score = query_scores[
                correct_rank - 1
            ]

            correct_vector_position = (
                correct_rank - 1
            )

        else:

            print(
                f"\nWARNING: Correct chunk {correct_chunk} "
                f"is not present in the Top-5 for Q{question_id}."
            )


    # =========================================================================
    # PRINT QUESTION HEADER
    # =========================================================================

    print("\n\n")

    print("=" * 100)

    print(
        f"QUESTION {question_id} — TOP-5 GEOMETRY ANALYSIS"
    )

    print("=" * 100)

    print("\nQuestion:")
    print(question)


    # =========================================================================
    # RETRIEVAL SCORES
    # =========================================================================

    print("\n")
    print("-" * 100)
    print("QUERY → TOP-5 COSINE SCORES")
    print("-" * 100)

    print(
        f"{'Rank':<8}"
        f"{'Chunk ID':<15}"
        f"{'Query Cosine':<20}"
    )

    print("-" * 100)

    for rank in range(5):

        print(
            f"{rank + 1:<8}"
            f"{chunk_ids[rank]:<15}"
            f"{query_scores[rank]:<20.6f}"
        )


    # =========================================================================
    # CONSECUTIVE GAPS
    # =========================================================================

    print("\n")
    print("-" * 100)
    print("CONSECUTIVE SCORE GAPS")
    print("-" * 100)

    print(
        f"Rank 1 − Rank 2 : {gaps[0]:.6f}"
    )

    print(
        f"Rank 2 − Rank 3 : {gaps[1]:.6f}"
    )

    print(
        f"Rank 3 − Rank 4 : {gaps[2]:.6f}"
    )

    print(
        f"Rank 4 − Rank 5 : {gaps[3]:.6f}"
    )


    # =========================================================================
    # CORRECT CHUNK
    # =========================================================================

    print("\n")
    print("-" * 100)
    print("CORRECT CHUNK ANALYSIS")
    print("-" * 100)

    if correct_chunk is None:

        print(
            "Correct chunk : NOT SPECIFIED"
        )

        print(
            "Set CORRECT_CHUNKS in Block 2."
        )

    elif correct_rank is None:

        print(
            f"Correct chunk : {correct_chunk}"
        )

        print(
            "Correct chunk rank : Outside Top-5"
        )

    else:

        print(
            f"Correct chunk       : {correct_chunk}"
        )

        print(
            f"Correct chunk rank  : {correct_rank}"
        )

        print(
            f"Query → correct cosine : "
            f"{correct_query_score:.6f}"
        )


    # =========================================================================
    # TOP-5 PAIRWISE COSINE MATRIX
    # =========================================================================

    print("\n")
    print("-" * 100)
    print("PAIRWISE COSINE SIMILARITY — TOP-5 CHUNKS")
    print("-" * 100)

    print(
        "\nRows and columns follow retrieval rank."
    )

    print(
        "\n"
        f"{'':<12}"
        + "".join(
            f"{chunk_id:<15}"
            for chunk_id in chunk_ids
        )
    )

    print("-" * 100)

    for i in range(5):

        print(
            f"{chunk_ids[i]:<12}"
            + "".join(
                f"{pairwise_matrix[i, j]:<15.6f}"
                for j in range(5)
            )
        )


    # =========================================================================
    # CORRECT CHUNK vs COMPETING CHUNKS
    # =========================================================================

    print("\n")
    print("-" * 100)
    print("CORRECT CHUNK vs COMPETING TOP-5 CHUNKS")
    print("-" * 100)

    if correct_rank is None:

        print(
            "Correct chunk is not available in the Top-5."
        )

    else:

        print(
            f"\nCorrect chunk: {correct_chunk}"
        )

        print(
            f"Correct chunk rank: {correct_rank}"
        )

        print(
            "\n"
            f"{'Competing Rank':<18}"
            f"{'Competing Chunk':<20}"
            f"{'Cosine vs Correct':<20}"
        )

        print("-" * 100)

        for rank in range(5):

            if rank == correct_vector_position:

                continue

            competing_chunk = chunk_ids[rank]

            score = pairwise_matrix[
                correct_vector_position,
                rank
            ]

            print(
                f"{rank + 1:<18}"
                f"{competing_chunk:<20}"
                f"{score:<20.6f}"
            )


    # =========================================================================
    # CHUNK METADATA / TRACEABILITY
    # =========================================================================

    print("\n")
    print("-" * 100)
    print("TOP-5 CHUNK TRACEABILITY")
    print("-" * 100)

    for rank, chunk_id in enumerate(
        chunk_ids,
        start=1
    ):

        vector_index = chunk_to_index[chunk_id]

        metadata = chunk_id_to_metadata.get(
            chunk_id,
            {}
        )

        print("\n")
        print(
            f"RANK {rank} — {chunk_id}"
        )

        print(
            f"Vector index : {vector_index}"
        )

        if isinstance(metadata, dict):

            for key, value in metadata.items():

                print(
                    f"{key:<20}: {value}"
                )

        else:

            print(
                f"Metadata     : {metadata}"
            )


# =============================================================================
# EXPERIMENT COMPLETE
# =============================================================================

print("\n\n")

print("=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)

print("\nUsed:")
print("  • Existing B01 Query → Top-5 cosine scores")
print("  • B01 embeddings.npy")
print("  • B01 index_to_chunk_id.json")
print("  • B01 chunk_id_to_metadata.json")

print("\nNo new retrieval was performed.")
print("No mean-centering was applied.")
print("No PCA was applied.")
print("No whitening was applied.")
print("No dimensionality reduction was applied.")