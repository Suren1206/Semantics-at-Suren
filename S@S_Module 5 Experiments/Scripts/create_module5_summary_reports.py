# ============================================================
# S@S MODULE 5
# CONSOLIDATED RETRIEVAL SUMMARY REPORTS
#
# Purpose:
#   Generate one consolidated TXT report for each of the
#   five trained representation spaces.
#
# Inputs:
#   - Questions_B01_150.csv
#   - Five trained MiniLM models
#   - Five B01 embedding files
#
# Output:
#   5 TXT files under Outputs/
#
# Each TXT contains:
#   - Experiment/model information
#   - Corpus information
#   - Aggregate retrieval statistics
#   - Rank-wise statistics
#   - HIGH/MEDIUM/LOW classification
#   - Top-1 statistics
#   - Rank-1 / Rank-2 gap statistics
#   - Explicit five intervention benchmark cases
#
# No interpretation such as "better" or "worse" is made.
# ============================================================


from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUTS = ROOT / "Inputs"
OUTPUTS = ROOT / "Outputs"
MODELS = OUTPUTS / "Models"

QUESTIONS_FILE = (
    INPUTS / "Questions_B01_150.csv"
)

MATRIX_FILE = (
    INPUTS / "Intervention_Matrix.csv"
)

INDEX_MAPPING_FILE = (
    INPUTS
    / "20260424_B01"
    / "index_to_chunk_id.json"
)


# ============================================================
# 2. RETRIEVAL CONFIGURATION
# ============================================================

TOP_K = 5

HIGH_THRESHOLD = 0.75
MEDIUM_THRESHOLD = 0.60


# ============================================================
# 3. FIVE REPRESENTATION SPACES
#
# IMPORTANT:
# The five .npy files are now under Inputs/.
# The trained models remain under Outputs/Models/.
# ============================================================

REPRESENTATIONS = {

    "E2_Contrastive": {

        "model": (
            MODELS
            / "E2_Contrastive_MiniLM"
        ),

        "embeddings": (
            INPUTS
            / "E2_Contrastive_MiniLM_B01_embeddings.npy"
        ),

        "label": "E2 — Contrastive",
    },


    "E3_Triplet": {

        "model": (
            MODELS
            / "E3_Triplet_MiniLM"
        ),

        "embeddings": (
            INPUTS
            / "E3_Triplet_MiniLM_B01_embeddings.npy"
        ),

        "label": "E3 — Triplet",
    },


    "E4_InfoNCE": {

        "model": (
            MODELS
            / "E4_InfoNCE_MiniLM"
        ),

        "embeddings": (
            INPUTS
            / "E4_InfoNCE_MiniLM_B01_embeddings.npy"
        ),

        "label": "E4 — InfoNCE",
    },


    "E5_HardNegative": {

        "model": (
            MODELS
            / "E5_HardNegative_MiniLM"
        ),

        "embeddings": (
            INPUTS
            / "E5_HardNegative_MiniLM_B01_embeddings.npy"
        ),

        "label": "E5 — Hard Negative",
    },


    "E6_AngularMargin": {

        "model": (
            MODELS
            / "E6_AngularMargin_MiniLM"
        ),

        "embeddings": (
            INPUTS
            / "E6_AngularMargin_MiniLM_B01_embeddings.npy"
        ),

        "label": "E6 — Angular Margin",
    },
}


# ============================================================
# 4. LOAD QUESTIONS
# ============================================================

print("=" * 80)
print("S@S MODULE 5 — CONSOLIDATED RETRIEVAL REPORTS")
print("=" * 80)

print()
print("Loading benchmark questions...")

questions = pd.read_csv(
    QUESTIONS_FILE
)

if "Qno" not in questions.columns:

    raise ValueError(
        "Questions_B01_150.csv must contain "
        "the column 'Qno'."
    )


if "Question" not in questions.columns:

    raise ValueError(
        "Questions_B01_150.csv must contain "
        "the column 'Question'."
    )


if len(questions) != 150:

    raise ValueError(
        f"Expected 150 benchmark questions. "
        f"Found {len(questions)}."
    )


questions["Qno"] = (
    questions["Qno"]
    .astype(int)
)

questions["Question"] = (
    questions["Question"]
    .astype(str)
)


print(
    f"Questions loaded : {len(questions)}"
)


# ============================================================
# 5. LOAD INDEX → CHUNK MAPPING
# ============================================================

print()
print("Loading B01 index-to-chunk mapping...")

if not INDEX_MAPPING_FILE.exists():

    raise FileNotFoundError(
        f"Missing index mapping:\n"
        f"{INDEX_MAPPING_FILE}"
    )


with open(
    INDEX_MAPPING_FILE,
    "r",
    encoding="utf-8"
) as f:

    index_mapping = json.load(f)


def numeric_key(value):

    try:
        return int(value)

    except Exception:

        return str(value)


if isinstance(
    index_mapping,
    dict
):

    sorted_items = sorted(
        index_mapping.items(),
        key=lambda item:
            numeric_key(item[0])
    )

    corpus_chunk_ids = [
        str(value)
        for _, value in sorted_items
    ]

else:

    corpus_chunk_ids = [
        str(value)
        for value in index_mapping
    ]


if len(corpus_chunk_ids) != 161:

    raise ValueError(
        f"Expected 161 B01 chunk IDs. "
        f"Found {len(corpus_chunk_ids)}."
    )


print(
    f"Corpus chunks    : {len(corpus_chunk_ids)}"
)


# ============================================================
# 6. LOAD INTERVENTION MATRIX
# ============================================================

print()
print("Loading intervention matrix...")

if not MATRIX_FILE.exists():

    raise FileNotFoundError(
        f"Missing intervention matrix:\n"
        f"{MATRIX_FILE}"
    )


matrix = pd.read_csv(
    MATRIX_FILE
)


CASE_COLUMN = "Unnamed: 0"

CONTRASTIVE_COLUMN = (
    "Contrastive: what constitutes "
    "the positive and negatives?"
)

TRIPLET_COLUMN = (
    "Triplet: which chunk is the negative?"
)

INFONCE_COLUMN = (
    "InfoNCE: what constitutes "
    "the candidate set?"
)

HARDNEG_COLUMN = (
    "Hard Negative: is chunk_4 explicitly "
    "designated as the hard negative?"
)

ANGULAR_COLUMN = (
    "Angular Margin: what relationship "
    "receives the angular margin?"
)


required_columns = [

    CASE_COLUMN,
    CONTRASTIVE_COLUMN,
    TRIPLET_COLUMN,
    INFONCE_COLUMN,
    HARDNEG_COLUMN,
    ANGULAR_COLUMN,
]


missing_columns = [

    column
    for column in required_columns
    if column not in matrix.columns
]


if missing_columns:

    raise ValueError(
        "Intervention_Matrix.csv is missing "
        f"columns: {missing_columns}"
    )


# ============================================================
# 7. PARSE INTERVENTION MATRIX
# ============================================================

def extract_qno(value):

    if pd.isna(value):

        return None

    match = re.search(
        r"\bQ(\d+)\b",
        str(value)
    )

    if match:

        return int(
            match.group(1)
        )

    return None


def extract_chunks(value):

    if pd.isna(value):

        return []

    matches = re.findall(
        r"chunk[_\s-]*(\d+)",
        str(value),
        flags=re.IGNORECASE
    )

    return [
        f"chunk_{int(number)}"
        for number in matches
    ]


benchmark_matrix = {}


for _, row in matrix.iterrows():

    qno = extract_qno(
        row[CASE_COLUMN]
    )

    if qno is None:

        continue

    benchmark_matrix[qno] = {

        "contrastive":
            extract_chunks(
                row[CONTRASTIVE_COLUMN]
            ),

        "triplet":
            extract_chunks(
                row[TRIPLET_COLUMN]
            ),

        "infonce":
            extract_chunks(
                row[INFONCE_COLUMN]
            ),

        "hard_negative":
            extract_chunks(
                row[HARDNEG_COLUMN]
            ),

        "angular":
            extract_chunks(
                row[ANGULAR_COLUMN]
            ),
    }


# ============================================================
# 8. STATISTICS HELPER
# ============================================================

def rank_statistics(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    return {

        "mean":
            float(np.mean(values)),

        "median":
            float(np.median(values)),

        "min":
            float(np.min(values)),

        "max":
            float(np.max(values)),
    }


# ============================================================
# 9. CREATE ONE REPORT
# ============================================================

def create_report(
    experiment_name,
    config
):

    print()
    print("=" * 80)
    print(
        f"PROCESSING {experiment_name}"
    )
    print("=" * 80)


    model_path = config["model"]

    embedding_path = config["embeddings"]


    # --------------------------------------------------------
    # Validate files.
    # --------------------------------------------------------

    if not model_path.exists():

        raise FileNotFoundError(
            f"Trained model not found:\n"
            f"{model_path}"
        )


    if not embedding_path.exists():

        raise FileNotFoundError(
            f"Embedding file not found:\n"
            f"{embedding_path}"
        )


    # --------------------------------------------------------
    # Load trained model.
    # --------------------------------------------------------

    print(
        f"Loading model:\n"
        f"{model_path}"
    )

    model = SentenceTransformer(
        str(model_path)
    )


    # --------------------------------------------------------
    # Load B01 embeddings.
    # --------------------------------------------------------

    print(
        f"Loading embeddings:\n"
        f"{embedding_path}"
    )

    corpus_embeddings = np.load(
        embedding_path
    )


    if corpus_embeddings.shape != (
        161,
        384
    ):

        raise ValueError(

            f"{experiment_name}: "
            f"expected embedding shape "
            f"(161, 384), got "
            f"{corpus_embeddings.shape}"
        )


    # --------------------------------------------------------
    # Normalize corpus vectors.
    # --------------------------------------------------------

    norms = np.linalg.norm(
        corpus_embeddings,
        axis=1,
        keepdims=True
    )


    if np.any(
        norms == 0
    ):

        raise ValueError(
            f"{experiment_name}: "
            f"zero-norm corpus embedding found."
        )


    corpus_embeddings = (
        corpus_embeddings
        / norms
    )


    # --------------------------------------------------------
    # Encode all 150 questions using the corresponding
    # trained representation model.
    # --------------------------------------------------------

    print(
        "Encoding 150 benchmark questions..."
    )

    query_embeddings = model.encode(

        questions[
            "Question"
        ].tolist(),

        batch_size=32,

        normalize_embeddings=True,

        convert_to_numpy=True,

        show_progress_bar=True
    )


    query_embeddings = np.asarray(
        query_embeddings,
        dtype=np.float32
    )


    if query_embeddings.shape != (
        150,
        384
    ):

        raise ValueError(

            f"{experiment_name}: "
            f"expected query shape "
            f"(150, 384), got "
            f"{query_embeddings.shape}"
        )


    # --------------------------------------------------------
    # Query × corpus cosine similarity.
    # --------------------------------------------------------

    similarity_matrix = (
        query_embeddings
        @ corpus_embeddings.T
    )


    # ========================================================
    # COLLECT RETRIEVAL RESULTS
    # ========================================================

    rank_scores = {

        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
    }


    all_scores = []

    top1_scores = []

    top1_chunks = []

    rank12_gaps = []

    question_results = []


    for question_index in range(
        len(questions)
    ):

        qno = int(
            questions.iloc[
                question_index
            ]["Qno"]
        )


        question = str(
            questions.iloc[
                question_index
            ]["Question"]
        )


        scores = similarity_matrix[
            question_index
        ]


        top_indices = np.argsort(
            -scores
        )[:TOP_K]


        retrieved = []


        for rank, corpus_index in enumerate(

            top_indices,

            start=1
        ):

            chunk_id = (
                corpus_chunk_ids[
                    corpus_index
                ]
            )


            score = float(
                scores[
                    corpus_index
                ]
            )


            rank_scores[
                rank
            ].append(score)


            all_scores.append(
                score
            )


            retrieved.append({

                "rank":
                    rank,

                "chunk_id":
                    chunk_id,

                "score":
                    score,
            })


        top1 = retrieved[0]["score"]

        top2 = retrieved[1]["score"]

        gap = top1 - top2


        top1_scores.append(
            top1
        )


        top1_chunks.append(
            retrieved[0]["chunk_id"]
        )


        rank12_gaps.append(
            gap
        )


        question_results.append({

            "qno":
                qno,

            "question":
                question,

            "retrieved":
                retrieved,

            "top1":
                top1,

            "top2":
                top2,

            "gap":
                gap,
        })


    # ========================================================
    # AGGREGATE STATISTICS
    # ========================================================

    all_scores = np.asarray(
        all_scores,
        dtype=np.float64
    )


    top1_scores = np.asarray(
        top1_scores,
        dtype=np.float64
    )


    rank12_gaps = np.asarray(
        rank12_gaps,
        dtype=np.float64
    )


    # ========================================================
    # QUESTION-LEVEL CLASSIFICATION
    # ========================================================

    high_count = int(
        np.sum(
            top1_scores
            >= HIGH_THRESHOLD
        )
    )


    medium_count = int(
        np.sum(

            (top1_scores
             >= MEDIUM_THRESHOLD)

            &

            (top1_scores
             < HIGH_THRESHOLD)
        )
    )


    low_count = int(
        np.sum(
            top1_scores
            < MEDIUM_THRESHOLD
        )
    )


    high_pct = (
        high_count
        / 150
        * 100
    )


    medium_pct = (
        medium_count
        / 150
        * 100
    )


    low_pct = (
        low_count
        / 150
        * 100
    )


    # ========================================================
    # BUILD TXT
    # ========================================================

    lines = []


    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    lines.append(
        "=" * 80
    )

    lines.append(
        "B01 — COMPLETE RETRIEVAL OUTPUT"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")


    lines.append(
        "Query File       : "
        "Questions_B01_150.csv"
    )


    lines.append(
        "Embedding Model  : "
        f"{config['label']}"
    )


    lines.append(
        "Base Architecture: "
        "sentence-transformers/all-MiniLM-L6-v2"
    )


    lines.append(
        "Corpus Vectors   : 161"
    )


    lines.append(
        "Corpus Dimension : 384"
    )


    lines.append(
        f"Top-K            : {TOP_K}"
    )


    lines.append(
        f"HIGH Threshold   : >= "
        f"{HIGH_THRESHOLD:.2f}"
    )


    lines.append(
        f"MEDIUM Threshold : >= "
        f"{MEDIUM_THRESHOLD:.2f} "
        f"and < {HIGH_THRESHOLD:.2f}"
    )


    lines.append(
        f"LOW Threshold    : < "
        f"{MEDIUM_THRESHOLD:.2f}"
    )


    # ========================================================
    # AGGREGATE RETRIEVAL STATISTICS
    # ========================================================

    lines.append("")

    lines.append(
        "=" * 80
    )

    lines.append(
        "AGGREGATE RETRIEVAL STATISTICS"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")


    lines.append(
        f"Total Questions          : 150"
    )


    lines.append(
        f"Corpus Vectors           : 161"
    )


    lines.append(
        f"Corpus Dimensions        : 384"
    )


    lines.append(
        f"Top-K                    : {TOP_K}"
    )


    lines.append(
        f"Total Retrieved Chunks   : "
        f"{150 * TOP_K}"
    )


    # ========================================================
    # ALL RETRIEVED RESULTS
    # ========================================================

    lines.append("")

    lines.append(
        "ALL RETRIEVED RESULTS"
    )

    lines.append(
        "---------------------"
    )


    lines.append(
        f"Minimum Cosine          : "
        f"{np.min(all_scores):.6f}"
    )


    lines.append(
        f"Maximum Cosine          : "
        f"{np.max(all_scores):.6f}"
    )


    lines.append(
        f"Mean Cosine             : "
        f"{np.mean(all_scores):.6f}"
    )


    lines.append(
        f"Median Cosine           : "
        f"{np.median(all_scores):.6f}"
    )


    # ========================================================
    # RANK-WISE STATISTICS
    # ========================================================

    lines.append("")

    lines.append(
        "RANK-WISE RETRIEVAL STATISTICS"
    )

    lines.append(
        "------------------------------"
    )


    for rank in range(1, 6):

        stats = rank_statistics(
            rank_scores[rank]
        )


        lines.append(

            f"Rank {rank}: "
            f"Mean={stats['mean']:.6f}, "
            f"Median={stats['median']:.6f}, "
            f"Min={stats['min']:.6f}, "
            f"Max={stats['max']:.6f}"
        )


    # ========================================================
    # QUESTION-LEVEL CLASSIFICATION
    # ========================================================

    lines.append("")

    lines.append(
        "QUESTION-LEVEL RETRIEVAL CLASSIFICATION"
    )

    lines.append(
        "----------------------------------------"
    )


    lines.append(

        f"HIGH   (>= {HIGH_THRESHOLD:.2f}) : "
        f"{high_count:4d} questions "
        f"({high_pct:.2f}%)"
    )


    lines.append(

        f"MEDIUM ({MEDIUM_THRESHOLD:.2f}–"
        f"{HIGH_THRESHOLD:.2f}) : "
        f"{medium_count:4d} questions "
        f"({medium_pct:.2f}%)"
    )


    lines.append(

        f"LOW    (< {MEDIUM_THRESHOLD:.2f}) : "
        f"{low_count:4d} questions "
        f"({low_pct:.2f}%)"
    )


    # ========================================================
    # TOP-1 STATISTICS
    # ========================================================

    lines.append("")

    lines.append(
        "TOP-1 SCORE STATISTICS"
    )

    lines.append(
        "----------------------"
    )


    lines.append(

        f"Minimum Top-1 Score     : "
        f"{np.min(top1_scores):.6f}"
    )


    lines.append(

        f"Maximum Top-1 Score     : "
        f"{np.max(top1_scores):.6f}"
    )


    lines.append(

        f"Mean Top-1 Score        : "
        f"{np.mean(top1_scores):.6f}"
    )


    lines.append(

        f"Median Top-1 Score      : "
        f"{np.median(top1_scores):.6f}"
    )


    # ========================================================
    # RANK-1 / RANK-2 GAP
    # ========================================================

    lines.append("")

    lines.append(
        "RANK-1 / RANK-2 GAP STATISTICS"
    )

    lines.append(
        "-------------------------------"
    )


    lines.append(

        f"Minimum Gap             : "
        f"{np.min(rank12_gaps):.6f}"
    )


    lines.append(

        f"Maximum Gap             : "
        f"{np.max(rank12_gaps):.6f}"
    )


    lines.append(

        f"Mean Gap                : "
        f"{np.mean(rank12_gaps):.6f}"
    )


    lines.append(

        f"Median Gap              : "
        f"{np.median(rank12_gaps):.6f}"
    )


    # ========================================================
    # EXPLICIT FIVE BENCHMARK CASES
    # ========================================================

    lines.append("")

    lines.append(
        "=" * 80
    )

    lines.append(
        "EXPLICIT INTERVENTION BENCHMARK CASES"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")


    for qno in [
        56,
        103,
        108,
        112,
        148,
    ]:

        record = next(

            item
            for item in question_results

            if item["qno"] == qno
        )


        lines.append(
            f"Q{qno}: "
            f"{record['question']}"
        )

        lines.append("")


        lines.append(
            "RETRIEVED TOP-5"
        )

        lines.append(
            "----------------"
        )


        for item in record["retrieved"]:

            lines.append(

                f"Rank {item['rank']}: "
                f"{item['chunk_id']} "
                f"(Cosine="
                f"{item['score']:.6f})"
            )


        lines.append("")


        lines.append(
            f"Rank-1 / Rank-2 Gap : "
            f"{record['gap']:.6f}"
        )


        benchmark = benchmark_matrix.get(
            qno
        )


        if benchmark:

            lines.append("")

            lines.append(
                "INTERVENTION MATRIX RELATIONSHIPS"
            )

            lines.append(
                "----------------------------------"
            )


            lines.append(

                "Contrastive : "
                f"{benchmark['contrastive']}"
            )


            lines.append(

                "Triplet     : "
                f"{benchmark['triplet']}"
            )


            lines.append(

                "InfoNCE     : "
                f"{benchmark['infonce']}"
            )


            lines.append(

                "Hard Negative : "
                f"{benchmark['hard_negative']}"
            )


            lines.append(

                "Angular Margin : "
                f"{benchmark['angular']}"
            )


        lines.append("")


    # ========================================================
    # GEOMETRY-RELATED NUMERICAL SUMMARY
    # ========================================================

    lines.append(
        "=" * 80
    )

    lines.append(
        "GEOMETRY-RELATED NUMERICAL SUMMARY"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")


    lines.append(
        "The following measurements describe the "
        "observed retrieval geometry."
    )


    lines.append(
        "They are reported without assigning a "
        "quality interpretation."
    )

    lines.append("")


    lines.append(

        f"Mean Top-1 Cosine       : "
        f"{np.mean(top1_scores):.6f}"
    )


    lines.append(

        f"Median Top-1 Cosine     : "
        f"{np.median(top1_scores):.6f}"
    )


    lines.append(

        f"Mean Rank-1/Rank-2 Gap  : "
        f"{np.mean(rank12_gaps):.6f}"
    )


    lines.append(

        f"Median Rank-1/Rank-2 Gap: "
        f"{np.median(rank12_gaps):.6f}"
    )


    lines.append(

        f"Top-1 Cosine Std        : "
        f"{np.std(top1_scores):.6f}"
    )


    lines.append(

        f"Top-5 Cosine Std        : "
        f"{np.std(all_scores):.6f}"
    )


    # ========================================================
    # END
    # ========================================================

    lines.append("")

    lines.append(
        "=" * 80
    )

    lines.append(
        "END OF B01 RETRIEVAL OUTPUT"
    )

    lines.append(
        "=" * 80
    )


    return "\n".join(lines)


# ============================================================
# 10. GENERATE FIVE REPORTS
# ============================================================

for experiment_name, config in REPRESENTATIONS.items():

    report = create_report(
        experiment_name,
        config
    )


    output_file = (

        OUTPUTS
        / f"{experiment_name}_B01_Retrieval_Summary.txt"
    )


    output_file.write_text(
        report,
        encoding="utf-8"
    )


    print()

    print(
        f"Saved report:"
    )

    print(
        f"  {output_file}"
    )


# ============================================================
# 11. FINAL MESSAGE
# ============================================================

print()

print(
    "=" * 80
)

print(
    "MODULE 5 — FIVE SUMMARY REPORTS COMPLETE"
)

print(
    "=" * 80
)

print()

print(
    "Generated files:"
)

for experiment_name in REPRESENTATIONS:

    print(
        f"  {experiment_name}"
        f"_B01_Retrieval_Summary.txt"
    )

print()

print(
    "Each report contains:"
)

print(
    "  - 150 benchmark questions"
)

print(
    "  - 750 retrieved chunks"
)

print(
    "  - Aggregate cosine statistics"
)

print(
    "  - Rank-wise statistics"
)

print(
    "  - HIGH / MEDIUM / LOW classification"
)

print(
    "  - Top-1 statistics"
)

print(
    "  - Rank-1 / Rank-2 gap statistics"
)

print(
    "  - Five explicit intervention cases"
)

print(
    "  - Geometry-related numerical summary"
)

print(
    "=" * 80
)