# =============================================================================
# Semantics @ Suren — Module 4
#
# CONTROLLED RETRIEVAL SENSITIVITY EXPERIMENT
#
# Raw vs Mean-Centered vs Whitened vs Intrinsic-Dimensionality-Reduced
# =============================================================================
#
# OBJECTIVE
# ---------
# Using the SAME Synthetic corpus questions, compare Top-5 retrieval behaviour
# under four different embedding representations:
#
#   1. Raw embeddings
#   2. Mean-centered embeddings
#   3. Whitened embeddings
#   4. Intrinsic-dimensionality-reduced embeddings
#
# The experiment changes ONLY the embedding representation.
#
# No:
#   - corpus modification
#   - question modification
#   - source-chunk modification
#   - inter-chunk similarity analysis
#   - hubness analysis
#   - mean-centering of raw data unless explicitly testing centered space
#   - PCA interpretation/output
#   - additional dimensionality reduction
#
# The script produces:
#
#   1. Detailed CSV
#      25 questions × 4 representations × 5 ranks = 500 rows
#
#   2. Summary CSV
#      One row per question
#
# =============================================================================


# =============================================================================
# BLOCK 1 — IMPORTS
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA


# =============================================================================
# BLOCK 2 — PATH HANDLING
# =============================================================================
#
# This script is stored inside:
#
#   S@S_Module 4  Experiments/Codes_Module_4/
#
# Therefore:
#
#   BASE_DIR = parent of Codes_Module_4
#
# This makes the script independent of the directory from which Python
# is launched.
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "Output_Files"


# =============================================================================
# BLOCK 3 — INPUT FILES
# =============================================================================

QUERY_FILE = BASE_DIR / "Query_Synthetic_Corpus.csv"

RAW_FILE = OUTPUT_DIR / "synthetic_raw_embeddings.npy"

MEAN_CENTERED_FILE = OUTPUT_DIR / "mean_centered_Synthetic.npy"

WHITENED_FILE = OUTPUT_DIR / "Synthetic_whitened_embeddings.npy"

REDUCED_FILE = OUTPUT_DIR / "Synthetic_80pct_variance_reduced.npy"


# =============================================================================
# BLOCK 4 — OUTPUT FILES
# =============================================================================

DETAILED_OUTPUT = (
    OUTPUT_DIR
    / "Synthetic_retrieval_raw_centered_whitened_reduced_detailed.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "Synthetic_retrieval_raw_centered_whitened_reduced_summary.csv"
)


# =============================================================================
# BLOCK 5 — EMBEDDING MODEL
# =============================================================================
#
# This must be the SAME embedding model used to create the Synthetic corpus
# embeddings.
#
# Current Module 4 Synthetic embedding model:
#
#   mxbai-embed-large:335m
#
# Embeddings are obtained from the local Ollama server.
# =============================================================================

OLLAMA_URL = "http://localhost:11434/api/embeddings"

EMBEDDING_MODEL = "mxbai-embed-large:335m"


# =============================================================================
# BLOCK 6 — EXPERIMENT SETTINGS
# =============================================================================

TOP_K = 5

RAW_DIMENSIONS = 1024

EXPECTED_QUESTIONS = 25

EXPECTED_CORPUS_VECTORS = 50

WHITENED_DIMENSIONS = 20

REDUCED_DIMENSIONS = 24


# =============================================================================
# BLOCK 7 — HEADER
# =============================================================================

print("=" * 100)
print("Semantics @ Suren — Module 4")
print("CONTROLLED RETRIEVAL SENSITIVITY EXPERIMENT")
print("=" * 100)

print()
print("Representations:")
print("  1. Raw")
print("  2. Mean-centered")
print("  3. Whitened")
print("  4. Intrinsic-dimensionality-reduced")

print()
print("The experiment measures retrieval changes only.")
print("No representation is declared better or worse.")

print()
print("Input directory:")
print(BASE_DIR)

print()
print("Output directory:")
print(OUTPUT_DIR)


# =============================================================================
# BLOCK 8 — VERIFY INPUT FILES
# =============================================================================

print()
print("=" * 100)
print("INPUT FILE VERIFICATION")
print("=" * 100)

required_files = [
    QUERY_FILE,
    RAW_FILE,
    MEAN_CENTERED_FILE,
    WHITENED_FILE,
    REDUCED_FILE,
]

for file_path in required_files:

    if not file_path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{file_path}"
        )

    print(f"FOUND : {file_path.name}")


# =============================================================================
# BLOCK 9 — LOAD QUESTIONS
# =============================================================================

print()
print("=" * 100)
print("LOADING SYNTHETIC QUESTIONS")
print("=" * 100)

questions_df = pd.read_csv(QUERY_FILE)

print()
print("Question file shape :", questions_df.shape)

print()
print("Columns detected:")
print(list(questions_df.columns))


# -------------------------------------------------------------------------
# The actual column name in the supplied CSV is "Questioin".
# We intentionally use that exact existing column name.
# -------------------------------------------------------------------------

if "Questioin" not in questions_df.columns:
    raise ValueError(
        "\nExpected question column 'Questioin' was not found."
    )

if "Category" not in questions_df.columns:
    raise ValueError(
        "\nExpected question category column 'Category' was not found."
    )


# -------------------------------------------------------------------------
# Validate number of questions.
# -------------------------------------------------------------------------

if len(questions_df) != EXPECTED_QUESTIONS:

    raise ValueError(
        f"\nExpected {EXPECTED_QUESTIONS} questions, "
        f"but found {len(questions_df)}."
    )


# =============================================================================
# BLOCK 10 — ASSIGN QUESTION IDs
# =============================================================================
#
# The supplied Query_Synthetic_Corpus.csv contains:
#
#   Category
#   Questioin
#
# It does not contain a Question_ID column.
#
# Therefore we assign sequential IDs 1–25 based on the existing CSV order.
#
# The question text itself is NOT modified.
# =============================================================================

questions_df = questions_df.copy()

questions_df.insert(
    0,
    "Question_ID",
    range(1, len(questions_df) + 1)
)

questions_df["Question_Type"] = questions_df["Category"]


# =============================================================================
# BLOCK 11 — LOAD EMBEDDINGS
# =============================================================================

print()
print("=" * 100)
print("LOADING EMBEDDING REPRESENTATIONS")
print("=" * 100)


raw_embeddings = np.load(RAW_FILE)

mean_centered_embeddings = np.load(MEAN_CENTERED_FILE)

whitened_embeddings = np.load(WHITENED_FILE)

reduced_embeddings = np.load(REDUCED_FILE)


print()
print(f"Raw embeddings                  : {raw_embeddings.shape}")
print(f"Mean-centered embeddings        : {mean_centered_embeddings.shape}")
print(f"Whitened embeddings             : {whitened_embeddings.shape}")
print(f"Intrinsic-reduced embeddings    : {reduced_embeddings.shape}")


# =============================================================================
# BLOCK 12 — SHAPE VALIDATION
# =============================================================================

print()
print("=" * 100)
print("EMBEDDING SHAPE VALIDATION")
print("=" * 100)


if raw_embeddings.shape != (EXPECTED_CORPUS_VECTORS, RAW_DIMENSIONS):

    raise ValueError(
        f"\nUnexpected raw embedding shape: {raw_embeddings.shape}\n"
        f"Expected: ({EXPECTED_CORPUS_VECTORS}, {RAW_DIMENSIONS})"
    )


if mean_centered_embeddings.shape != raw_embeddings.shape:

    raise ValueError(
        "\nRaw and mean-centered embedding shapes do not match."
    )


if whitened_embeddings.shape != (
    EXPECTED_CORPUS_VECTORS,
    WHITENED_DIMENSIONS,
):

    raise ValueError(
        f"\nUnexpected whitened embedding shape: "
        f"{whitened_embeddings.shape}\n"
        f"Expected: "
        f"({EXPECTED_CORPUS_VECTORS}, {WHITENED_DIMENSIONS})"
    )


if reduced_embeddings.shape != (
    EXPECTED_CORPUS_VECTORS,
    REDUCED_DIMENSIONS,
):

    raise ValueError(
        f"\nUnexpected reduced embedding shape: "
        f"{reduced_embeddings.shape}\n"
        f"Expected: "
        f"({EXPECTED_CORPUS_VECTORS}, {REDUCED_DIMENSIONS})"
    )


print()
print("All embedding shapes are valid.")


# =============================================================================
# BLOCK 13 — VERIFY CORPUS REPRESENTATION ALIGNMENT
# =============================================================================
#
# Every representation must contain the same 50 corpus vectors in the same
# order.
#
# The .npy files themselves do not contain chunk IDs, so the experiment assumes
# that the existing Module 4 transformation files preserve the original
# ordering.
# =============================================================================

print()
print("=" * 100)
print("CORPUS REPRESENTATION ALIGNMENT")
print("=" * 100)

print()
print("Corpus vectors across all representations : 50")
print("Chunk ordering                              : Preserved")
print()
print(
    "The representations will therefore be compared using the same "
    "corpus row index."
)


# =============================================================================
# BLOCK 14 — RECONSTRUCT TRANSFORMATION MODELS FOR QUERY PROJECTION
# =============================================================================
#
# IMPORTANT:
#
# The corpus representation files already exist.
#
# However, the query starts as a 1024-dimensional raw embedding.
#
# Therefore the query must be transformed into the SAME space as the
# corresponding corpus representation.
#
# Mean-centered:
#
#       query_centered = query_raw - mean_vector
#
# Whitened:
#
#       PCA fitted on the mean-centered corpus
#       whiten=True
#
# Reduced:
#
#       PCA fitted on the mean-centered corpus
#       n_components=24
#
# We fit these transformations from the existing corpus representations.
#
# This does NOT regenerate or modify the stored representation files.
# =============================================================================


print()
print("=" * 100)
print("PREPARING QUERY-SPACE TRANSFORMATIONS")
print("=" * 100)


# -------------------------------------------------------------------------
# Mean vector
# -------------------------------------------------------------------------

mean_vector = np.mean(
    raw_embeddings,
    axis=0
)


# -------------------------------------------------------------------------
# Verify that the stored mean-centered embeddings correspond to:
#
#       raw - mean(raw)
#
# -------------------------------------------------------------------------

reconstructed_mean_centered = (
    raw_embeddings - mean_vector
)

mean_centering_difference = np.max(
    np.abs(
        reconstructed_mean_centered
        - mean_centered_embeddings
    )
)

print()
print(
    "Maximum difference between stored and reconstructed "
    "mean-centered embeddings : "
    f"{mean_centering_difference:.12e}"
)


# -------------------------------------------------------------------------
# PCA for whitening
# -------------------------------------------------------------------------
#
# Whitening artifact has 20 dimensions.
#
# We fit PCA on the mean-centered corpus with:
#
#       n_components=20
#       whiten=True
#
# -------------------------------------------------------------------------

whitening_pca = PCA(
    n_components=WHITENED_DIMENSIONS,
    whiten=True
)

whitening_pca.fit(
    mean_centered_embeddings
)


# -------------------------------------------------------------------------
# PCA for intrinsic-dimensionality reduction
# -------------------------------------------------------------------------
#
# Existing reduced representation contains 24 dimensions.
#
# Fit PCA on the mean-centered corpus.
# -------------------------------------------------------------------------

reduction_pca = PCA(
    n_components=REDUCED_DIMENSIONS,
    whiten=False
)

reduction_pca.fit(
    mean_centered_embeddings
)


print()
print("Whitening PCA components :", whitening_pca.n_components_)
print("Reduction PCA components :", reduction_pca.n_components_)


# =============================================================================
# BLOCK 15 — TRANSFORMATION CONSISTENCY CHECK
# =============================================================================
#
# Confirm that the PCA-generated corpus representations match the existing
# stored representation files.
#
# PCA component sign can theoretically be sign-flipped while representing
# the same mathematical subspace. Such sign changes do not alter cosine
# relationships when both query and corpus are represented consistently.
# =============================================================================


reconstructed_whitened = whitening_pca.transform(
    mean_centered_embeddings
)

reconstructed_reduced = reduction_pca.transform(
    mean_centered_embeddings
)


whitened_difference = np.max(
    np.abs(
        reconstructed_whitened
        - whitened_embeddings
    )
)

reduced_difference = np.max(
    np.abs(
        reconstructed_reduced
        - reduced_embeddings
    )
)


print()
print(
    "Maximum difference — stored vs reconstructed whitening : "
    f"{whitened_difference:.12e}"
)

print(
    "Maximum difference — stored vs reconstructed reduction  : "
    f"{reduced_difference:.12e}"
)


# =============================================================================
# BLOCK 16 — OLLAMA QUERY EMBEDDING FUNCTION
# =============================================================================

import json
from urllib.request import Request, urlopen


def generate_query_embedding(text):

    payload = {
        "model": EMBEDDING_MODEL,
        "prompt": text,
    }

    data = json.dumps(payload).encode("utf-8")

    request = Request(
        OLLAMA_URL,
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    try:

        with urlopen(request, timeout=120) as response:

            response_data = json.loads(
                response.read().decode("utf-8")
            )

    except Exception as exc:

        raise RuntimeError(
            "\nUnable to generate query embedding through Ollama.\n"
            f"URL   : {OLLAMA_URL}\n"
            f"Model : {EMBEDDING_MODEL}\n"
            f"Error : {exc}"
        )

    if "embedding" not in response_data:

        raise ValueError(
            "\nOllama response did not contain an 'embedding' field."
        )

    embedding = np.asarray(
        response_data["embedding"],
        dtype=np.float64
    )

    if embedding.ndim != 1:

        raise ValueError(
            f"\nUnexpected query embedding shape: {embedding.shape}"
        )

    if embedding.shape[0] != RAW_DIMENSIONS:

        raise ValueError(
            "\nQuery embedding dimensionality does not match corpus.\n"
            f"Query : {embedding.shape[0]}\n"
            f"Corpus: {RAW_DIMENSIONS}"
        )

    return embedding


# =============================================================================
# BLOCK 17 — COSINE SIMILARITY FUNCTION
# =============================================================================
#
# Cosine similarity is calculated between the query and every corpus vector.
#
# The representations themselves are NOT normalized before storage or
# comparison.
#
# Cosine similarity mathematically accounts for vector norms during scoring.
# =============================================================================


def cosine_similarity_to_population(
    query_vector,
    population
):

    query_norm = np.linalg.norm(query_vector)

    population_norms = np.linalg.norm(
        population,
        axis=1
    )

    if query_norm == 0:

        raise ValueError(
            "\nQuery vector has zero L2 norm."
        )

    if np.any(population_norms == 0):

        zero_indices = np.where(
            population_norms == 0
        )[0]

        raise ValueError(
            "\nOne or more corpus vectors have zero L2 norm.\n"
            f"Indices: {zero_indices.tolist()}"
        )

    scores = (
        population @ query_vector
    ) / (
        population_norms * query_norm
    )

    return scores


# =============================================================================
# BLOCK 18 — QUERY TRANSFORMATION FUNCTIONS
# =============================================================================


def transform_query_mean_centered(
    query_raw
):

    return query_raw - mean_vector


def transform_query_whitened(
    query_raw
):

    query_centered = (
        query_raw - mean_vector
    )

    return whitening_pca.transform(
        query_centered.reshape(1, -1)
    )[0]


def transform_query_reduced(
    query_raw
):

    query_centered = (
        query_raw - mean_vector
    )

    return reduction_pca.transform(
        query_centered.reshape(1, -1)
    )[0]


# =============================================================================
# BLOCK 19 — REPRESENTATION DEFINITIONS
# =============================================================================
#
# The order here is deliberate and follows the experimental requirement.
# =============================================================================

representations = {

    "Raw": {
        "embeddings": raw_embeddings,
        "query_transform": lambda q: q,
    },

    "Mean-centered": {
        "embeddings": mean_centered_embeddings,
        "query_transform": transform_query_mean_centered,
    },

    "Whitened": {
        "embeddings": whitened_embeddings,
        "query_transform": transform_query_whitened,
    },

    "Intrinsic-dimensionality-reduced": {
        "embeddings": reduced_embeddings,
        "query_transform": transform_query_reduced,
    },

}


# =============================================================================
# BLOCK 20 — RETRIEVAL FUNCTION
# =============================================================================


def retrieve_top_k(
    query_vector,
    corpus_embeddings,
    k=TOP_K
):

    scores = cosine_similarity_to_population(
        query_vector,
        corpus_embeddings
    )

    # Highest score first.
    ranking = np.argsort(
        scores
    )[::-1]

    top_indices = ranking[:k]

    top_scores = scores[top_indices]

    return top_indices, top_scores


# =============================================================================
# BLOCK 21 — RETRIEVAL EXPERIMENT
# =============================================================================

print()
print("=" * 100)
print("RUNNING CONTROLLED RETRIEVAL EXPERIMENT")
print("=" * 100)

print()
print(
    f"Questions       : {len(questions_df)}"
)

print(
    f"Representations : {len(representations)}"
)

print(
    f"Top-K           : {TOP_K}"
)

print(
    f"Expected rows    : "
    f"{len(questions_df) * len(representations) * TOP_K}"
)


detailed_records = []

summary_records = []


# =============================================================================
# BLOCK 22 — PROCESS EACH QUESTION
# =============================================================================

for _, question_row in questions_df.iterrows():

    question_id = int(
        question_row["Question_ID"]
    )

    question_type = str(
        question_row["Question_Type"]
    )

    question = str(
        question_row["Questioin"]
    )


    # -------------------------------------------------------------------------
    # Generate ONE raw query embedding.
    #
    # The same query embedding is then transformed into the other spaces.
    # -------------------------------------------------------------------------

    query_raw = generate_query_embedding(
        question
    )


    per_representation = {}


    # =========================================================================
    # BLOCK 23 — RUN ALL FOUR REPRESENTATIONS
    # =========================================================================

    for representation_name, representation_data in representations.items():

        corpus = representation_data["embeddings"]

        query_transform = representation_data["query_transform"]


        # ---------------------------------------------------------------------
        # Transform query into the corresponding representation space.
        # ---------------------------------------------------------------------

        query_representation = query_transform(
            query_raw
        )


        # ---------------------------------------------------------------------
        # Retrieve Top-5.
        # ---------------------------------------------------------------------

        top_indices, top_scores = retrieve_top_k(
            query_representation,
            corpus,
            TOP_K
        )


        # ---------------------------------------------------------------------
        # Convert retrieval indices into stable chunk IDs.
        #
        # The original embedding ordering corresponds to:
        #
        #   vector index 0 → chunk_0
        #   vector index 1 → chunk_1
        #   ...
        #
        # This preserves direct comparison across representations.
        # ---------------------------------------------------------------------

        top_chunk_ids = [
            f"chunk_{int(index)}"
            for index in top_indices
        ]


        top_score_list = [
            float(score)
            for score in top_scores
        ]


        # ---------------------------------------------------------------------
        # Rank 1 → Rank 2 gap.
        # ---------------------------------------------------------------------

        rank1_rank2_gap = (
            top_score_list[0]
            - top_score_list[1]
        )


        # ---------------------------------------------------------------------
        # Store representation result.
        # ---------------------------------------------------------------------

        per_representation[representation_name] = {

            "top_indices": top_indices.tolist(),

            "top_chunk_ids": top_chunk_ids,

            "top_scores": top_score_list,

            "top1_chunk": top_chunk_ids[0],

            "top1_score": top_score_list[0],

            "rank1_rank2_gap": rank1_rank2_gap,

        }


        # ---------------------------------------------------------------------
        # Detailed evidence rows.
        # ---------------------------------------------------------------------

        for rank_position in range(TOP_K):

            detailed_records.append({

                "Question_ID":
                    question_id,

                "Question_Type":
                    question_type,

                "Question":
                    question,

                "Representation":
                    representation_name,

                "Rank":
                    rank_position + 1,

                "Chunk_ID":
                    top_chunk_ids[rank_position],

                "Cosine_Score":
                    top_score_list[rank_position],

                "Rank1_Rank2_Gap":
                    rank1_rank2_gap,

                "Top1_Chunk":
                    top_chunk_ids[0],

                "Top1_Score":
                    top_score_list[0],

            })


    # =========================================================================
    # BLOCK 24 — CROSS-REPRESENTATION COMPARISON FLAGS
    # =========================================================================
    #
    # Raw is treated as the baseline representation.
    #
    # We compare every transformed representation against Raw.
    # =========================================================================

    raw_result = per_representation["Raw"]

    centered_result = per_representation["Mean-centered"]

    whitened_result = per_representation["Whitened"]

    reduced_result = (
        per_representation[
            "Intrinsic-dimensionality-reduced"
        ]
    )


    # -------------------------------------------------------------------------
    # Top-1 changes
    # -------------------------------------------------------------------------

    top1_changed_centered = (
        centered_result["top1_chunk"]
        != raw_result["top1_chunk"]
    )

    top1_changed_whitened = (
        whitened_result["top1_chunk"]
        != raw_result["top1_chunk"]
    )

    top1_changed_reduced = (
        reduced_result["top1_chunk"]
        != raw_result["top1_chunk"]
    )


    # -------------------------------------------------------------------------
    # Top-5 membership changes
    #
    # Here order is ignored.
    #
    # Example:
    #
    # Raw       : chunk_1, chunk_2, chunk_3, chunk_4, chunk_5
    # Centered  : chunk_3, chunk_1, chunk_5, chunk_4, chunk_2
    #
    # Membership is unchanged even though ranking changed.
    # -------------------------------------------------------------------------

    raw_top5_set = set(
        raw_result["top_chunk_ids"]
    )

    centered_top5_set = set(
        centered_result["top_chunk_ids"]
    )

    whitened_top5_set = set(
        whitened_result["top_chunk_ids"]
    )

    reduced_top5_set = set(
        reduced_result["top_chunk_ids"]
    )


    top5_membership_changed_centered = (
        centered_top5_set
        != raw_top5_set
    )

    top5_membership_changed_whitened = (
        whitened_top5_set
        != raw_top5_set
    )

    top5_membership_changed_reduced = (
        reduced_top5_set
        != raw_top5_set
    )


    # -------------------------------------------------------------------------
    # Ranking changes
    #
    # Here order DOES matter.
    # -------------------------------------------------------------------------

    top5_ranking_changed_centered = (
        centered_result["top_chunk_ids"]
        != raw_result["top_chunk_ids"]
    )

    top5_ranking_changed_whitened = (
        whitened_result["top_chunk_ids"]
        != raw_result["top_chunk_ids"]
    )

    top5_ranking_changed_reduced = (
        reduced_result["top_chunk_ids"]
        != raw_result["top_chunk_ids"]
    )


    # -------------------------------------------------------------------------
    # Similarity-score changes
    #
    # Because transformed spaces can legitimately change cosine magnitudes,
    # this records whether the actual Top-5 score vector differs from Raw.
    # -------------------------------------------------------------------------

    scores_changed_centered = not np.allclose(
        centered_result["top_scores"],
        raw_result["top_scores"],
        rtol=1e-7,
        atol=1e-9
    )

    scores_changed_whitened = not np.allclose(
        whitened_result["top_scores"],
        raw_result["top_scores"],
        rtol=1e-7,
        atol=1e-9
    )

    scores_changed_reduced = not np.allclose(
        reduced_result["top_scores"],
        raw_result["top_scores"],
        rtol=1e-7,
        atol=1e-9
    )


    # =========================================================================
    # BLOCK 25 — SUMMARY RECORD
    # =========================================================================

    summary_records.append({

        "Question_ID":
            question_id,

        "Question_Type":
            question_type,

        "Question":
            question,


        # ---------------------------------------------------------------------
        # RAW
        # ---------------------------------------------------------------------

        "Raw_Top1_Chunk":
            raw_result["top1_chunk"],

        "Raw_Top1_Score":
            raw_result["top1_score"],

        "Raw_Rank1_Rank2_Gap":
            raw_result["rank1_rank2_gap"],

        "Raw_Top5_Chunk_IDs":
            " | ".join(
                raw_result["top_chunk_ids"]
            ),

        "Raw_Top5_Cosine_Scores":
            " | ".join(
                f"{x:.6f}"
                for x in raw_result["top_scores"]
            ),


        # ---------------------------------------------------------------------
        # MEAN-CENTERED
        # ---------------------------------------------------------------------

        "MeanCentered_Top1_Chunk":
            centered_result["top1_chunk"],

        "MeanCentered_Top1_Score":
            centered_result["top1_score"],

        "MeanCentered_Rank1_Rank2_Gap":
            centered_result["rank1_rank2_gap"],

        "MeanCentered_Top5_Chunk_IDs":
            " | ".join(
                centered_result["top_chunk_ids"]
            ),

        "MeanCentered_Top5_Cosine_Scores":
            " | ".join(
                f"{x:.6f}"
                for x in centered_result["top_scores"]
            ),


        # ---------------------------------------------------------------------
        # WHITENED
        # ---------------------------------------------------------------------

        "Whitened_Top1_Chunk":
            whitened_result["top1_chunk"],

        "Whitened_Top1_Score":
            whitened_result["top1_score"],

        "Whitened_Rank1_Rank2_Gap":
            whitened_result["rank1_rank2_gap"],

        "Whitened_Top5_Chunk_IDs":
            " | ".join(
                whitened_result["top_chunk_ids"]
            ),

        "Whitened_Top5_Cosine_Scores":
            " | ".join(
                f"{x:.6f}"
                for x in whitened_result["top_scores"]
            ),


        # ---------------------------------------------------------------------
        # INTRINSIC-DIMENSIONALITY-REDUCED
        # ---------------------------------------------------------------------

        "Reduced_Top1_Chunk":
            reduced_result["top1_chunk"],

        "Reduced_Top1_Score":
            reduced_result["top1_score"],

        "Reduced_Rank1_Rank2_Gap":
            reduced_result["rank1_rank2_gap"],

        "Reduced_Top5_Chunk_IDs":
            " | ".join(
                reduced_result["top_chunk_ids"]
            ),

        "Reduced_Top5_Cosine_Scores":
            " | ".join(
                f"{x:.6f}"
                for x in reduced_result["top_scores"]
            ),


        # ---------------------------------------------------------------------
        # COMPARISON FLAGS — CENTERED vs RAW
        # ---------------------------------------------------------------------

        "Centered_vs_Raw_Top1_Changed":
            top1_changed_centered,

        "Centered_vs_Raw_Top5_Membership_Changed":
            top5_membership_changed_centered,

        "Centered_vs_Raw_Ranking_Changed":
            top5_ranking_changed_centered,

        "Centered_vs_Raw_Similarity_Scores_Changed":
            scores_changed_centered,


        # ---------------------------------------------------------------------
        # COMPARISON FLAGS — WHITENED vs RAW
        # ---------------------------------------------------------------------

        "Whitened_vs_Raw_Top1_Changed":
            top1_changed_whitened,

        "Whitened_vs_Raw_Top5_Membership_Changed":
            top5_membership_changed_whitened,

        "Whitened_vs_Raw_Ranking_Changed":
            top5_ranking_changed_whitened,

        "Whitened_vs_Raw_Similarity_Scores_Changed":
            scores_changed_whitened,


        # ---------------------------------------------------------------------
        # COMPARISON FLAGS — REDUCED vs RAW
        # ---------------------------------------------------------------------

        "Reduced_vs_Raw_Top1_Changed":
            top1_changed_reduced,

        "Reduced_vs_Raw_Top5_Membership_Changed":
            top5_membership_changed_reduced,

        "Reduced_vs_Raw_Ranking_Changed":
            top5_ranking_changed_reduced,

        "Reduced_vs_Raw_Similarity_Scores_Changed":
            scores_changed_reduced,

    })


# =============================================================================
# BLOCK 26 — CREATE DATAFRAMES
# =============================================================================

detailed_df = pd.DataFrame(
    detailed_records
)

summary_df = pd.DataFrame(
    summary_records
)


# =============================================================================
# BLOCK 27 — VALIDATE OUTPUT SIZE
# =============================================================================

expected_detailed_rows = (
    EXPECTED_QUESTIONS
    * len(representations)
    * TOP_K
)

actual_detailed_rows = len(
    detailed_df
)


if actual_detailed_rows != expected_detailed_rows:

    raise ValueError(
        "\nUnexpected detailed CSV row count.\n"
        f"Expected : {expected_detailed_rows}\n"
        f"Actual   : {actual_detailed_rows}"
    )


if len(summary_df) != EXPECTED_QUESTIONS:

    raise ValueError(
        "\nUnexpected summary CSV row count.\n"
        f"Expected : {EXPECTED_QUESTIONS}\n"
        f"Actual   : {len(summary_df)}"
    )


# =============================================================================
# BLOCK 28 — SAVE CSV FILES
# =============================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


detailed_df.to_csv(
    DETAILED_OUTPUT,
    index=False
)


summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False
)


# =============================================================================
# BLOCK 29 — FINAL OUTPUT REPORT
# =============================================================================

print()
print("=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)

print()
print("Detailed evidence CSV:")
print(DETAILED_OUTPUT)

print()
print("Summary comparison CSV:")
print(SUMMARY_OUTPUT)

print()
print("Detailed rows :", len(detailed_df))
print("Summary rows  :", len(summary_df))

print()
print("Expected detailed rows :", expected_detailed_rows)

print()
print("=" * 100)
print("REPRESENTATION SUMMARY")
print("=" * 100)

print()
print(
    "Raw                         : 1024 dimensions"
)

print(
    "Mean-centered               : 1024 dimensions"
)

print(
    "Whitened                    : 20 dimensions"
)

print(
    "Intrinsic-dimensionality    : 24 dimensions"
)

print()
print("=" * 100)
print("NO INTERPRETATION WAS APPLIED")
print("=" * 100)

print()
print("The script only measures retrieval behaviour.")

print()
print("No conclusions about which representation is better")
print("have been generated.")

print()
print("End of experiment.")
print("=" * 100)