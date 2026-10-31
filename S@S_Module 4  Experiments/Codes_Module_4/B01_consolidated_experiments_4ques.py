# =============================================================================
# Semantics @ Suren — Module 4
# B01 — CONTROLLED RETRIEVAL SENSITIVITY EXPERIMENT
#
# Representations:
#   1. Raw
#   2. Mean-Centered
#   3. Whitened
#   4. Intrinsic-Dimensionality-Reduced
#
# Questions:
#   Q1, Q39, Q177, Q186
#
# Output:
#   - Detailed CSV
#   - Summary CSV
#
# IMPORTANT:
#   The original embeddings and corpus are never modified.
#   Retrieval methodology remains identical across representations.
# =============================================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# =============================================================================
# BLOCK 1 — BASE DIRECTORY
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "Output_Files"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# BLOCK 2 — FILE LOCATIONS
# =============================================================================

RAW_FILE = (
    BASE_DIR
    / "20260424_B01"
    / "embeddings.npy"
)

MEAN_CENTERED_FILE = (
    OUTPUT_DIR
    / "mean_centered_B01.npy"
)

WHITENED_FILE = (
    OUTPUT_DIR
    / "B01_whitened_embeddings.npy"
)

REDUCED_FILE = (
    OUTPUT_DIR
    / "B01_80pct_variance_reduced.npy"
)

METADATA_FILE = (
    BASE_DIR
    / "20260424_B01"
    / "chunk_id_to_metadata.json"
)

MEAN_VECTOR_FILE = (
    OUTPUT_DIR
    / "mean_vector_160.npy"
)

DETAILED_OUTPUT = (
    OUTPUT_DIR
    / "B01_4Q_retrieval_raw_centered_whitened_reduced_detailed.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "B01_4Q_retrieval_raw_centered_whitened_reduced_summary.csv"
)


# =============================================================================
# BLOCK 3 — QUESTIONS
# =============================================================================

QUESTIONS = [
    (
        "Q1",
        "Simple / directly answerable",
        "How should I notify my manager when I am unable to attend work?"
    ),
    (
        "Q39",
        "Difficult / distributed-answer",
        "What if my manager only tried to contact me once?"
    ),
    (
        "Q177",
        "Simple / directly answerable",
        "After returning from more than two consecutive working days of absence, "
        "what document must employees submit?"
    ),
    (
        "Q186",
        "Simple / directly answerable",
        "How many working days in advance must employees submit leave requests "
        "for up to five days of leave?"
    ),
]


# =============================================================================
# BLOCK 4 — LOAD METADATA
# =============================================================================

print("=" * 100)
print("Semantics @ Suren — Module 4")
print("B01 — CONTROLLED RETRIEVAL SENSITIVITY EXPERIMENT")
print("=" * 100)

print("\n")
print("INPUT / REPRESENTATION FILES")
print("-" * 100)

print(f"{'Raw':<40}: {RAW_FILE}")
print(f"{'Mean-Centered':<40}: {MEAN_CENTERED_FILE}")
print(f"{'Whitened':<40}: {WHITENED_FILE}")
print(f"{'Intrinsic-Dimensionality-Reduced':<40}: {REDUCED_FILE}")

required_files = [
    RAW_FILE,
    MEAN_CENTERED_FILE,
    WHITENED_FILE,
    REDUCED_FILE,
    METADATA_FILE,
    MEAN_VECTOR_FILE,
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{file_path}"
        )

print("\nAll required files found.")


# =============================================================================
# BLOCK 5 — LOAD METADATA
# =============================================================================

print("\n")
print("=" * 100)
print("LOADING B01 CHUNK METADATA")
print("=" * 100)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Metadata entries : {len(metadata)}")


# =============================================================================
# BLOCK 6 — LOAD EMBEDDINGS
# =============================================================================

print("\n")
print("=" * 100)
print("LOADING REPRESENTATIONS")
print("=" * 100)

raw_embeddings = np.load(RAW_FILE)
mean_centered_embeddings = np.load(MEAN_CENTERED_FILE)
whitened_embeddings = np.load(WHITENED_FILE)
reduced_embeddings = np.load(REDUCED_FILE)

print(
    f"{'Raw':<40}: "
    f"{raw_embeddings.shape[0]} vectors × "
    f"{raw_embeddings.shape[1]} dimensions"
)

print(
    f"{'Mean-Centered':<40}: "
    f"{mean_centered_embeddings.shape[0]} vectors × "
    f"{mean_centered_embeddings.shape[1]} dimensions"
)

print(
    f"{'Whitened':<40}: "
    f"{whitened_embeddings.shape[0]} vectors × "
    f"{whitened_embeddings.shape[1]} dimensions"
)

print(
    f"{'Intrinsic-Dimensionality-Reduced':<40}: "
    f"{reduced_embeddings.shape[0]} vectors × "
    f"{reduced_embeddings.shape[1]} dimensions"
)


# =============================================================================
# BLOCK 7 — BASIC CONSISTENCY CHECK
# =============================================================================

representation_counts = {
    raw_embeddings.shape[0],
    mean_centered_embeddings.shape[0],
    whitened_embeddings.shape[0],
    reduced_embeddings.shape[0],
}

if len(representation_counts) != 1:
    raise ValueError(
        "\nVector-count mismatch between representation files."
    )

n_vectors = raw_embeddings.shape[0]

print(
    f"\nVerified vector count across representations: "
    f"{n_vectors}"
)


# =============================================================================
# BLOCK 8 — CHUNK ID ORDER
# =============================================================================

# The raw B01 embedding matrix follows the vector index ordering.
#
# index_to_chunk_id.json maps vector index -> chunk ID.

INDEX_MAP_FILE = (
    BASE_DIR
    / "20260424_B01"
    / "index_to_chunk_id.json"
)

if not INDEX_MAP_FILE.exists():
    raise FileNotFoundError(
        f"\nRequired file not found:\n{INDEX_MAP_FILE}"
    )

with open(INDEX_MAP_FILE, "r", encoding="utf-8") as f:
    index_to_chunk_id = json.load(f)

chunk_ids = []

for i in range(n_vectors):

    key = str(i)

    if key in index_to_chunk_id:
        chunk_ids.append(index_to_chunk_id[key])

    elif i in index_to_chunk_id:
        chunk_ids.append(index_to_chunk_id[i])

    else:
        raise ValueError(
            f"Could not find chunk ID for vector index {i}."
        )


# =============================================================================
# BLOCK 9 — QUERY EMBEDDING
# =============================================================================

# IMPORTANT:
#
# The original query embedding is generated in the raw 1024-D embedding
# space.
#
# Raw:
#       query_raw
#
# Mean-centered:
#       query_raw - mean_vector
#
# Reduced:
#       PCA.transform(query_mean_centered)
#
# Whitened:
#       PCA whitening applied consistently to the mean-centered query.
#
# The query is NOT independently normalized here.
# Cosine similarity is calculated explicitly later.
# =============================================================================

print("\n")
print("=" * 100)
print("GENERATING QUERY EMBEDDINGS")
print("=" * 100)

# -------------------------------------------------------------------------
# Import the same embedding mechanism used in the project.
#
# Ollama is used here because the B01 embeddings were generated using
# the same local embedding model.
# -------------------------------------------------------------------------

import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "mxbai-embed-large:335m"


def get_query_embedding(text):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": EMBEDDING_MODEL,
            "prompt": text,
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return np.asarray(
        data["embedding"],
        dtype=np.float32,
    )


query_embeddings = {}

for question_id, question_type, question in QUESTIONS:

    query_vector = get_query_embedding(question)

    if query_vector.shape[0] != raw_embeddings.shape[1]:
        raise ValueError(
            f"\n{question_id}: Query dimension mismatch.\n"
            f"Query dimension : {query_vector.shape[0]}\n"
            f"Raw dimension   : {raw_embeddings.shape[1]}"
        )

    query_embeddings[question_id] = query_vector

    print(
        f"\n{question_id}: {question}"
    )

    print(
        f"Query embedding shape: {query_vector.shape}"
    )


# =============================================================================
# BLOCK 10 — LOAD MEAN VECTOR
# =============================================================================

mean_vector = np.load(MEAN_VECTOR_FILE)

print("\n")
print("=" * 100)
print("MEAN VECTOR")
print("=" * 100)

print(
    f"Mean vector shape: {mean_vector.shape}"
)

if mean_vector.shape[0] != raw_embeddings.shape[1]:
    raise ValueError(
        "\nMean vector dimension does not match raw embeddings."
    )


# =============================================================================
# BLOCK 11 — FIT PCA FOR THE 80% VARIANCE REPRESENTATION
# =============================================================================

print("\n")
print("=" * 100)
print("RECONSTRUCTING PCA FOR 80% VARIANCE")
print("=" * 100)

# The existing B01 reduced representation was created from the
# mean-centered embeddings using PCA.
#
# We reproduce the SAME procedure:
#
#   1. Full PCA
#   2. Determine minimum PCs reaching 80%
#   3. Fit PCA again using exactly that number of components
#
# This gives us the transformation required to convert a query
# from 1024-D mean-centered space into the same reduced space.

pca_full = PCA(
    n_components=min(
        mean_centered_embeddings.shape[0],
        mean_centered_embeddings.shape[1],
    )
)

pca_full.fit(mean_centered_embeddings)

cumulative_variance = np.cumsum(
    pca_full.explained_variance_ratio_
)

reduced_dimension = (
    np.searchsorted(
        cumulative_variance,
        0.80,
    )
    + 1
)

print(
    f"Required PCs for 80% variance : "
    f"{reduced_dimension}"
)

if reduced_dimension != reduced_embeddings.shape[1]:
    raise ValueError(
        "\nThe reconstructed PCA dimension does not match "
        "the stored reduced embedding file.\n"
        f"Expected : {reduced_embeddings.shape[1]}\n"
        f"Found    : {reduced_dimension}"
    )

pca_reduced = PCA(
    n_components=reduced_dimension
)

pca_reduced.fit(mean_centered_embeddings)

print(
    f"Stored reduced dimension       : "
    f"{reduced_embeddings.shape[1]}"
)

print(
    f"Retained variance               : "
    f"{np.sum(pca_reduced.explained_variance_ratio_):.6f}"
)


# =============================================================================
# BLOCK 12 — FIT WHITENING TRANSFORMATION
# =============================================================================

print("\n")
print("=" * 100)
print("RECONSTRUCTING WHITENING TRANSFORMATION")
print("=" * 100)

# The existing whitened representation is 20-D.
#
# We reconstruct the whitening transformation using PCA with
# whiten=True on the mean-centered embeddings.
#
# The number of components is taken directly from the stored
# whitened representation.

whitened_dimension = whitened_embeddings.shape[1]

pca_whiten = PCA(
    n_components=whitened_dimension,
    whiten=True,
)

pca_whiten.fit(mean_centered_embeddings)

print(
    f"Whitened dimension : {whitened_dimension}"
)


# =============================================================================
# BLOCK 13 — COSINE SIMILARITY FUNCTION
# =============================================================================

def cosine_similarity(query_vector, matrix):

    query_norm = np.linalg.norm(query_vector)

    matrix_norms = np.linalg.norm(
        matrix,
        axis=1,
    )

    if query_norm == 0:
        raise ValueError(
            "Query vector has zero norm."
        )

    if np.any(matrix_norms == 0):
        raise ValueError(
            "At least one document vector has zero norm."
        )

    return (
        matrix @ query_vector
    ) / (
        matrix_norms * query_norm
    )


# =============================================================================
# BLOCK 14 — PREPARE QUERY REPRESENTATIONS
# =============================================================================

print("\n")
print("=" * 100)
print("TRANSFORMING QUERIES INTO EACH REPRESENTATION SPACE")
print("=" * 100)

query_representations = {}

for question_id, question_type, question in QUESTIONS:

    query_raw = query_embeddings[question_id]

    # -------------------------------------------------------------------------
    # RAW
    # -------------------------------------------------------------------------

    query_raw_rep = query_raw

    # -------------------------------------------------------------------------
    # MEAN-CENTERED
    # -------------------------------------------------------------------------

    query_centered_rep = (
        query_raw
        - mean_vector
    )

    # -------------------------------------------------------------------------
    # WHITENED
    # -------------------------------------------------------------------------

    query_whitened_rep = (
        pca_whiten.transform(
            query_centered_rep.reshape(1, -1)
        )[0]
    )

    # -------------------------------------------------------------------------
    # REDUCED
    # -------------------------------------------------------------------------

    query_reduced_rep = (
        pca_reduced.transform(
            query_centered_rep.reshape(1, -1)
        )[0]
    )

    # -------------------------------------------------------------------------
    # Dimension checks
    # -------------------------------------------------------------------------

    if query_raw_rep.shape[0] != raw_embeddings.shape[1]:
        raise ValueError(
            f"{question_id}: Raw query dimension mismatch."
        )

    if query_centered_rep.shape[0] != mean_centered_embeddings.shape[1]:
        raise ValueError(
            f"{question_id}: Mean-centered query dimension mismatch."
        )

    if query_whitened_rep.shape[0] != whitened_embeddings.shape[1]:
        raise ValueError(
            f"{question_id}: Whitened query dimension mismatch."
        )

    if query_reduced_rep.shape[0] != reduced_embeddings.shape[1]:
        raise ValueError(
            f"{question_id}: Reduced query dimension mismatch."
        )

    query_representations[question_id] = {
        "Raw": query_raw_rep,
        "Mean-Centered": query_centered_rep,
        "Whitened": query_whitened_rep,
        "Intrinsic-Dimensionality-Reduced": query_reduced_rep,
    }

    print(
        f"{question_id}: "
        f"Raw={query_raw_rep.shape[0]}, "
        f"Centered={query_centered_rep.shape[0]}, "
        f"Whitened={query_whitened_rep.shape[0]}, "
        f"Reduced={query_reduced_rep.shape[0]}"
    )


# =============================================================================
# BLOCK 15 — REPRESENTATION DEFINITIONS
# =============================================================================

representations = {
    "Raw": raw_embeddings,
    "Mean-Centered": mean_centered_embeddings,
    "Whitened": whitened_embeddings,
    "Intrinsic-Dimensionality-Reduced": reduced_embeddings,
}


# =============================================================================
# BLOCK 16 — RETRIEVAL
# =============================================================================

print("\n")
print("=" * 100)
print("RUNNING TOP-5 RETRIEVAL")
print("=" * 100)

detailed_rows = []

summary_rows = []

for question_id, question_type, question in QUESTIONS:

    print("\n")
    print("-" * 100)
    print(
        f"{question_id}: {question}"
    )
    print("-" * 100)

    retrieval_results = {}

    for representation_name, matrix in representations.items():

        query_vector = query_representations[
            question_id
        ][representation_name]

        similarities = cosine_similarity(
            query_vector,
            matrix,
        )

        ranking = np.argsort(
            similarities
        )[::-1]

        top5_indices = ranking[:5]

        top5_chunk_ids = [
            chunk_ids[idx]
            for idx in top5_indices
        ]

        top5_scores = [
            float(similarities[idx])
            for idx in top5_indices
        ]

        top1_chunk = top5_chunk_ids[0]
        top1_score = top5_scores[0]

        rank1_rank2_gap = (
            top5_scores[0]
            - top5_scores[1]
        )

        retrieval_results[
            representation_name
        ] = {
            "indices": top5_indices,
            "chunks": top5_chunk_ids,
            "scores": top5_scores,
            "top1_chunk": top1_chunk,
            "top1_score": top1_score,
            "gap": rank1_rank2_gap,
        }

        print(
            f"\n{representation_name}"
        )

        print(
            f"Top-1 : {top1_chunk} "
            f"({top1_score:.6f})"
        )

        print(
            f"R1-R2 gap : "
            f"{rank1_rank2_gap:.6f}"
        )

        print(
            "Top-5 : "
            + ", ".join(
                f"{cid} ({score:.6f})"
                for cid, score
                in zip(
                    top5_chunk_ids,
                    top5_scores,
                )
            )
        )

        # ---------------------------------------------------------------------
        # Detailed rows
        # ---------------------------------------------------------------------

        for rank_number, idx in enumerate(
            top5_indices,
            start=1,
        ):

            detailed_rows.append(
                {
                    "Question_ID": question_id,
                    "Question_Type": question_type,
                    "Question": question,
                    "Representation": representation_name,
                    "Rank": rank_number,
                    "Chunk_ID": chunk_ids[idx],
                    "Cosine_Score": float(
                        similarities[idx]
                    ),
                    "Rank1_Rank2_Gap": float(
                        rank1_rank2_gap
                    ),
                    "Top1_Chunk": top1_chunk,
                    "Top1_Score": float(
                        top1_score
                    ),
                }
            )

    # =========================================================================
    # BLOCK 17 — CROSS-REPRESENTATION COMPARISON
    # =========================================================================

    raw_result = retrieval_results["Raw"]
    centered_result = retrieval_results["Mean-Centered"]
    whitened_result = retrieval_results["Whitened"]
    reduced_result = retrieval_results[
        "Intrinsic-Dimensionality-Reduced"
    ]

    raw_top5 = raw_result["chunks"]
    centered_top5 = centered_result["chunks"]
    whitened_top5 = whitened_result["chunks"]
    reduced_top5 = reduced_result["chunks"]

    raw_scores = raw_result["scores"]
    centered_scores = centered_result["scores"]
    whitened_scores = whitened_result["scores"]
    reduced_scores = reduced_result["scores"]

    # -------------------------------------------------------------------------
    # Top-1 changes
    # -------------------------------------------------------------------------

    top1_changed_centered = (
        raw_result["top1_chunk"]
        != centered_result["top1_chunk"]
    )

    top1_changed_whitened = (
        raw_result["top1_chunk"]
        != whitened_result["top1_chunk"]
    )

    top1_changed_reduced = (
        raw_result["top1_chunk"]
        != reduced_result["top1_chunk"]
    )

    # -------------------------------------------------------------------------
    # Top-5 membership changes
    # -------------------------------------------------------------------------

    raw_set = set(raw_top5)

    top5_changed_centered = (
        raw_set != set(centered_top5)
    )

    top5_changed_whitened = (
        raw_set != set(whitened_top5)
    )

    top5_changed_reduced = (
        raw_set != set(reduced_top5)
    )

    # -------------------------------------------------------------------------
    # Ranking changes
    # -------------------------------------------------------------------------

    ranking_changed_centered = (
        raw_top5 != centered_top5
    )

    ranking_changed_whitened = (
        raw_top5 != whitened_top5
    )

    ranking_changed_reduced = (
        raw_top5 != reduced_top5
    )

    # -------------------------------------------------------------------------
    # Similarity-score changes
    #
    # Because the representations themselves change, we report whether
    # the Top-5 score vectors differ from the raw representation.
    # -------------------------------------------------------------------------

    scores_changed_centered = not np.allclose(
        raw_scores,
        centered_scores,
        rtol=1e-6,
        atol=1e-6,
    )

    scores_changed_whitened = not np.allclose(
        raw_scores,
        whitened_scores,
        rtol=1e-6,
        atol=1e-6,
    )

    scores_changed_reduced = not np.allclose(
        raw_scores,
        reduced_scores,
        rtol=1e-6,
        atol=1e-6,
    )

    # -------------------------------------------------------------------------
    # Summary row
    # -------------------------------------------------------------------------

    summary_row = {
        "Question_ID": question_id,
        "Question_Type": question_type,
        "Question": question,

        # ---------------------------------------------------------------------
        # RAW
        # ---------------------------------------------------------------------

        "Raw_Top1_Chunk": raw_result["top1_chunk"],
        "Raw_Top1_Score": raw_result["top1_score"],
        "Raw_Rank1_Rank2_Gap": raw_result["gap"],
        "Raw_Top5_Chunk_IDs": " | ".join(raw_top5),
        "Raw_Top5_Cosine_Scores": " | ".join(
            f"{x:.6f}" for x in raw_scores
        ),

        # ---------------------------------------------------------------------
        # MEAN CENTERED
        # ---------------------------------------------------------------------

        "MeanCentered_Top1_Chunk":
            centered_result["top1_chunk"],

        "MeanCentered_Top1_Score":
            centered_result["top1_score"],

        "MeanCentered_Rank1_Rank2_Gap":
            centered_result["gap"],

        "MeanCentered_Top5_Chunk_IDs":
            " | ".join(centered_top5),

        "MeanCentered_Top5_Cosine_Scores":
            " | ".join(
                f"{x:.6f}" for x in centered_scores
            ),

        # ---------------------------------------------------------------------
        # WHITENED
        # ---------------------------------------------------------------------

        "Whitened_Top1_Chunk":
            whitened_result["top1_chunk"],

        "Whitened_Top1_Score":
            whitened_result["top1_score"],

        "Whitened_Rank1_Rank2_Gap":
            whitened_result["gap"],

        "Whitened_Top5_Chunk_IDs":
            " | ".join(whitened_top5),

        "Whitened_Top5_Cosine_Scores":
            " | ".join(
                f"{x:.6f}" for x in whitened_scores
            ),

        # ---------------------------------------------------------------------
        # REDUCED
        # ---------------------------------------------------------------------

        "Reduced_Top1_Chunk":
            reduced_result["top1_chunk"],

        "Reduced_Top1_Score":
            reduced_result["top1_score"],

        "Reduced_Rank1_Rank2_Gap":
            reduced_result["gap"],

        "Reduced_Top5_Chunk_IDs":
            " | ".join(reduced_top5),

        "Reduced_Top5_Cosine_Scores":
            " | ".join(
                f"{x:.6f}" for x in reduced_scores
            ),

        # ---------------------------------------------------------------------
        # CHANGE FLAGS
        # ---------------------------------------------------------------------

        "Top1_Changed_Raw_vs_MeanCentered":
            top1_changed_centered,

        "Top1_Changed_Raw_vs_Whitened":
            top1_changed_whitened,

        "Top1_Changed_Raw_vs_Reduced":
            top1_changed_reduced,

        "Top5_Membership_Changed_Raw_vs_MeanCentered":
            top5_changed_centered,

        "Top5_Membership_Changed_Raw_vs_Whitened":
            top5_changed_whitened,

        "Top5_Membership_Changed_Raw_vs_Reduced":
            top5_changed_reduced,

        "Ranking_Changed_Raw_vs_MeanCentered":
            ranking_changed_centered,

        "Ranking_Changed_Raw_vs_Whitened":
            ranking_changed_whitened,

        "Ranking_Changed_Raw_vs_Reduced":
            ranking_changed_reduced,

        "Scores_Changed_Raw_vs_MeanCentered":
            scores_changed_centered,

        "Scores_Changed_Raw_vs_Whitened":
            scores_changed_whitened,

        "Scores_Changed_Raw_vs_Reduced":
            scores_changed_reduced,
    }

    summary_rows.append(summary_row)


# =============================================================================
# BLOCK 18 — WRITE DETAILED CSV
# =============================================================================

print("\n")
print("=" * 100)
print("WRITING DETAILED CSV")
print("=" * 100)

detailed_df = pd.DataFrame(
    detailed_rows
)

detailed_df.to_csv(
    DETAILED_OUTPUT,
    index=False,
)

print(
    f"Detailed rows written : "
    f"{len(detailed_df)}"
)

print(
    f"Expected rows         : "
    f"{4 * 4 * 5}"
)

print(
    f"File                  : "
    f"{DETAILED_OUTPUT}"
)


# =============================================================================
# BLOCK 19 — WRITE SUMMARY CSV
# =============================================================================

print("\n")
print("=" * 100)
print("WRITING SUMMARY CSV")
print("=" * 100)

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)

print(
    f"Summary rows written : "
    f"{len(summary_df)}"
)

print(
    f"Expected rows        : 4"
)

print(
    f"File                 : "
    f"{SUMMARY_OUTPUT}"
)


# =============================================================================
# BLOCK 20 — FINAL SUMMARY
# =============================================================================

print("\n")
print("=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)

print(
    f"""
Questions analysed                  : {len(QUESTIONS)}
Representations                     : {len(representations)}
Ranks per representation            : 5
Detailed evidence rows              : {len(detailed_df)}
Summary rows                         : {len(summary_df)}

Raw dimension                       : {raw_embeddings.shape[1]}
Mean-centered dimension             : {mean_centered_embeddings.shape[1]}
Whitened dimension                  : {whitened_embeddings.shape[1]}
Intrinsic-reduced dimension         : {reduced_embeddings.shape[1]}

Detailed CSV:
{DETAILED_OUTPUT}

Summary CSV:
{SUMMARY_OUTPUT}
"""
)

print("=" * 100)
print("No corpus was modified.")
print("No original embedding file was modified.")
print("No inter-chunk similarity calculation was performed.")
print("No hubness calculation was performed.")
print("No transformation was applied to the stored representations.")
print("=" * 100)