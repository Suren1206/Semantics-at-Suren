# ============================================================
# S@S MODULE 5
# CONSOLIDATED BENCHMARK OUTPUT GENERATION
#
# Generates exactly:
#
#   5 XML files
#   5 TXT files
#
# Each file contains all 150 benchmark questions.
#
# For each representation:
#   - trained model
#   - trained B01 corpus embeddings
#   - 150 benchmark questions
#   - Top-5 retrieval
#   - cosine similarities
#   - rank-1 / rank-2 gap
#   - geometry-related retrieval statistics
#   - overall numerical summary
#
# NO interpretation such as "better/worse" is generated.
# ============================================================

from pathlib import Path
import json
import re
import math
from xml.sax.saxutils import escape

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

QUESTIONS_FILE = INPUTS / "Questions_B01_150.csv"

CORPUS_METADATA_FILE = (
    INPUTS
    / "20260424_B01"
    / "chunk_id_to_metadata.json"
)

TOP_K = 5


# ============================================================
# 2. REPRESENTATIONS
# ============================================================

REPRESENTATIONS = {
    "E2_Contrastive": {
        "model_dir": MODELS / "E2_Contrastive_MiniLM",
        "embedding_file": (
            OUTPUTS
            / "E2_Contrastive_MiniLM_B01_embeddings.npy"
        ),
    },

    "E3_Triplet": {
        "model_dir": MODELS / "E3_Triplet_MiniLM",
        "embedding_file": (
            OUTPUTS
            / "E3_Triplet_MiniLM_B01_embeddings.npy"
        ),
    },

    "E4_InfoNCE": {
        "model_dir": MODELS / "E4_InfoNCE_MiniLM",
        "embedding_file": (
            OUTPUTS
            / "E4_InfoNCE_MiniLM_B01_embeddings.npy"
        ),
    },

    "E5_HardNegative": {
        "model_dir": MODELS / "E5_HardNegative_MiniLM",
        "embedding_file": (
            OUTPUTS
            / "E5_HardNegative_MiniLM_B01_embeddings.npy"
        ),
    },

    "E6_AngularMargin": {
        "model_dir": MODELS / "E6_AngularMargin_MiniLM",
        "embedding_file": (
            OUTPUTS
            / "E6_AngularMargin_MiniLM_B01_embeddings.npy"
        ),
    },
}


# ============================================================
# 3. LOAD QUESTIONS
# ============================================================

questions = pd.read_csv(QUESTIONS_FILE)

if "Qno" not in questions.columns:
    raise ValueError(
        "Questions_B01_150.csv must contain 'Qno'."
    )

if "Question" not in questions.columns:
    raise ValueError(
        "Questions_B01_150.csv must contain 'Question'."
    )

if len(questions) != 150:
    raise ValueError(
        f"Expected 150 benchmark questions, "
        f"found {len(questions)}."
    )

questions["Qno"] = questions["Qno"].astype(int)
questions["Question"] = questions["Question"].astype(str)


# ============================================================
# 4. LOAD B01 CORPUS METADATA
# ============================================================

with open(
    CORPUS_METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


def extract_text(value):

    if isinstance(value, str):
        return value

    if not isinstance(value, dict):
        raise ValueError(
            f"Unexpected metadata type: {type(value)}"
        )

    for key in [
        "text",
        "chunk_text",
        "content",
        "document_text",
    ]:

        if key in value:
            return str(value[key])

    raise ValueError(
        "No text field found in metadata entry."
    )


def normalize_chunk_id(value):

    text = str(value).strip()

    match = re.search(
        r"chunk[_\s-]*(\d+)",
        text,
        flags=re.IGNORECASE
    )

    if match:
        return f"chunk_{int(match.group(1))}"

    return text


# ============================================================
# 5. BUILD CORPUS LOOKUP
# ============================================================

chunk_lookup = {}

if isinstance(metadata, dict):

    for chunk_id, value in metadata.items():

        chunk_id = normalize_chunk_id(chunk_id)

        chunk_lookup[
            chunk_id
        ] = extract_text(value)

else:

    for value in metadata:

        if not isinstance(value, dict):
            continue

        chunk_id = (
            value.get("chunk_id")
            or value.get("id")
        )

        if chunk_id is None:
            continue

        chunk_id = normalize_chunk_id(chunk_id)

        chunk_lookup[
            chunk_id
        ] = extract_text(value)


if len(chunk_lookup) != 161:
    raise ValueError(
        f"Expected 161 B01 chunks, "
        f"found {len(chunk_lookup)}."
    )


corpus_chunk_ids = list(chunk_lookup.keys())

corpus_texts = [
    chunk_lookup[cid]
    for cid in corpus_chunk_ids
]


# ============================================================
# 6. INTERVENTION MATRIX
#
# Used only to identify the explicitly selected benchmark
# relationships. No new relationships are inferred.
# ============================================================

MATRIX_FILE = INPUTS / "Intervention_Matrix.csv"

matrix = pd.read_csv(MATRIX_FILE)

CASE_TEXT = "Unnamed: 0"

CONTRASTIVE_COL = (
    "Contrastive: what constitutes the positive and negatives?"
)

TRIPLET_COL = (
    "Triplet: which chunk is the negative?"
)

INFONCE_COL = (
    "InfoNCE: what constitutes the candidate set?"
)

HARD_NEGATIVE_COL = (
    "Hard Negative: is chunk_4 explicitly designated as the hard negative?"
)

ANGULAR_COL = (
    "Angular Margin: what relationship receives the angular margin?"
)


def extract_case(value):

    if pd.isna(value):
        return None

    match = re.search(
        r"\bQ(\d+)\b",
        str(value)
    )

    if match:
        return int(match.group(1))

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
        f"chunk_{int(x)}"
        for x in matches
    ]


matrix_cases = {}

for _, row in matrix.iterrows():

    qno = extract_case(row[CASE_TEXT])

    if qno is None:
        continue

    matrix_cases[qno] = {
        "contrastive": extract_chunks(
            row[CONTRASTIVE_COL]
        ),

        "triplet": extract_chunks(
            row[TRIPLET_COL]
        ),

        "infonce": extract_chunks(
            row[INFONCE_COL]
        ),

        "hard_negative": extract_chunks(
            row[HARD_NEGATIVE_COL]
        ),

        "angular": extract_chunks(
            row[ANGULAR_COL]
        ),
    }


# ============================================================
# 7. BENCHMARK RELATIONSHIP LOOKUP
# ============================================================

# Only the five explicitly intervention-defined cases
# have benchmark relationship information here.

BENCHMARK_CASES = {
    56,
    103,
    108,
    112,
    148,
}


def benchmark_relationship(qno):

    if qno not in matrix_cases:
        return {
            "known": False,
            "positive": None,
            "negatives": [],
        }

    data = matrix_cases[qno]

    # Contrastive column gives the most explicit positive /
    # negative relationship for the benchmark cases.

    contrastive = data["contrastive"]

    positive = (
        contrastive[0]
        if len(contrastive) >= 1
        else None
    )

    negatives = (
        contrastive[1:]
        if len(contrastive) >= 2
        else []
    )

    return {
        "known": qno in BENCHMARK_CASES,
        "positive": positive,
        "negatives": negatives,
    }


# ============================================================
# 8. XML ESCAPING HELPER
# ============================================================

def xml_attr(value):

    return escape(
        str(value),
        {
            '"': "&quot;",
            "'": "&apos;",
        }
    )


# ============================================================
# 9. RETRIEVAL STATISTICS
# ============================================================

def calculate_statistics(
    all_results
):

    rank1_scores = np.array(
        [
            r["results"][0]["score"]
            for r in all_results
        ],
        dtype=np.float64
    )

    rank2_scores = np.array(
        [
            r["results"][1]["score"]
            for r in all_results
        ],
        dtype=np.float64
    )

    rank5_scores = np.array(
        [
            r["results"][4]["score"]
            for r in all_results
        ],
        dtype=np.float64
    )

    rank1_rank2_gap = (
        rank1_scores
        - rank2_scores
    )

    all_top5_scores = np.array(
        [
            item["score"]
            for r in all_results
            for item in r["results"]
        ],
        dtype=np.float64
    )

    stats = {
        "questions": len(all_results),

        "top_k": TOP_K,

        "rank1_mean": float(
            np.mean(rank1_scores)
        ),

        "rank1_std": float(
            np.std(rank1_scores)
        ),

        "rank1_min": float(
            np.min(rank1_scores)
        ),

        "rank1_max": float(
            np.max(rank1_scores)
        ),

        "rank2_mean": float(
            np.mean(rank2_scores)
        ),

        "rank5_mean": float(
            np.mean(rank5_scores)
        ),

        "rank1_rank2_gap_mean": float(
            np.mean(rank1_rank2_gap)
        ),

        "rank1_rank2_gap_std": float(
            np.std(rank1_rank2_gap)
        ),

        "rank1_rank2_gap_min": float(
            np.min(rank1_rank2_gap)
        ),

        "rank1_rank2_gap_max": float(
            np.max(rank1_rank2_gap)
        ),

        "top5_score_mean": float(
            np.mean(all_top5_scores)
        ),

        "top5_score_std": float(
            np.std(all_top5_scores)
        ),

        "top5_score_min": float(
            np.min(all_top5_scores)
        ),

        "top5_score_max": float(
            np.max(all_top5_scores)
        ),
    }

    # --------------------------------------------------------
    # Explicit benchmark cases only.
    # --------------------------------------------------------

    benchmark_results = [
        r
        for r in all_results
        if r["benchmark"]["known"]
    ]

    stats["explicit_benchmark_cases"] = len(
        benchmark_results
    )

    if benchmark_results:

        positive_in_top5 = 0
        positive_rank1 = 0

        for r in benchmark_results:

            positive = (
                r["benchmark"]["positive"]
            )

            retrieved_ids = [
                item["chunk_id"]
                for item in r["results"]
            ]

            if positive in retrieved_ids:
                positive_in_top5 += 1

            if (
                len(r["results"]) > 0
                and r["results"][0]["chunk_id"]
                == positive
            ):
                positive_rank1 += 1

        stats[
            "explicit_positive_in_top5_count"
        ] = positive_in_top5

        stats[
            "explicit_positive_rank1_count"
        ] = positive_rank1

    return stats


# ============================================================
# 10. XML GENERATOR
# ============================================================

def create_xml(
    experiment_name,
    statistics,
    all_results
):

    lines = []

    lines.append(
        '<?xml version="1.0" encoding="UTF-8"?>'
    )

    lines.append(
        f'<benchmark_retrieval '
        f'experiment="{xml_attr(experiment_name)}" '
        f'questions="150" '
        f'top_k="5">'
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    lines.append("  <overall_statistics>")

    for key, value in statistics.items():

        lines.append(
            f'    <statistic '
            f'name="{xml_attr(key)}" '
            f'value="{xml_attr(value)}"/>'
        )

    lines.append(
        "  </overall_statistics>"
    )

    # --------------------------------------------------------
    # Questions
    # --------------------------------------------------------

    lines.append(
        "  <questions>"
    )

    for record in all_results:

        qno = record["qno"]

        lines.append(
            f'    <question '
            f'id="Q{qno}">'
        )

        lines.append(
            f"      <text>"
            f"{escape(record['question'])}"
            f"</text>"
        )

        # Benchmark information.

        benchmark = record["benchmark"]

        lines.append(
            f'      <benchmark '
            f'known="{str(benchmark["known"]).lower()}">'
        )

        if benchmark["positive"]:

            lines.append(
                f'        <positive>'
                f'{escape(benchmark["positive"])}'
                f'</positive>'
            )

        if benchmark["negatives"]:

            lines.append(
                "        <negatives>"
            )

            for negative in benchmark["negatives"]:

                lines.append(
                    f"          <negative>"
                    f"{escape(negative)}"
                    f"</negative>"
                )

            lines.append(
                "        </negatives>"
            )

        lines.append(
            "      </benchmark>"
        )

        # Retrieval results.

        lines.append(
            "      <retrieval>"
        )

        for item in record["results"]:

            lines.append(
                f'        <result '
                f'rank="{item["rank"]}" '
                f'chunk_id="{xml_attr(item["chunk_id"])}" '
                f'cosine_similarity="'
                f'{item["score"]:.8f}">'
            )

            lines.append(
                f"          <text>"
                f"{escape(item['text'])}"
                f"</text>"
            )

            lines.append(
                "        </result>"
            )

        lines.append(
            "      </retrieval>"
        )

        # Geometry-related local information.

        lines.append(
            "      <local_geometry>"
        )

        lines.append(
            f'        <rank1_score>'
            f'{record["rank1_score"]:.8f}'
            f'</rank1_score>'
        )

        lines.append(
            f'        <rank2_score>'
            f'{record["rank2_score"]:.8f}'
            f'</rank2_score>'
        )

        lines.append(
            f'        <rank1_rank2_gap>'
            f'{record["rank1_rank2_gap"]:.8f}'
            f'</rank1_rank2_gap>'
        )

        lines.append(
            "      </local_geometry>"
        )

        lines.append(
            "    </question>"
        )

    lines.append(
        "  </questions>"
    )

    lines.append(
        "</benchmark_retrieval>"
    )

    return "\n".join(lines)


# ============================================================
# 11. TXT GENERATOR
# ============================================================

def create_txt(
    experiment_name,
    statistics,
    all_results
):

    lines = []

    lines.append(
        "=" * 80
    )

    lines.append(
        f"S@S MODULE 5 — {experiment_name}"
    )

    lines.append(
        "FULL B01 BENCHMARK RETRIEVAL"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")
    lines.append(
        "This file contains retrieval results for "
        "all 150 B01 benchmark questions."
    )

    lines.append(
        "Top-K = 5."
    )

    lines.append(
        "Cosine similarity is calculated between "
        "normalized query and corpus embeddings."
    )

    lines.append("")

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    lines.append(
        "=" * 80
    )

    lines.append(
        "OVERALL NUMERICAL SUMMARY"
    )

    lines.append(
        "=" * 80
    )

    for key, value in statistics.items():

        readable = key.replace(
            "_",
            " "
        ).upper()

        if isinstance(value, float):

            lines.append(
                f"{readable:<45} : "
                f"{value:.8f}"
            )

        else:

            lines.append(
                f"{readable:<45} : "
                f"{value}"
            )

    lines.append("")

    lines.append(
        "The statistics above describe the observed "
        "retrieval geometry."
    )

    lines.append(
        "They do not constitute an interpretation "
        "of representation quality."
    )

    # ========================================================
    # ALL 150 QUESTIONS
    # ========================================================

    lines.append("")
    lines.append(
        "=" * 80
    )

    lines.append(
        "150-QUESTION RETRIEVAL RESULTS"
    )

    lines.append(
        "=" * 80
    )

    for record in all_results:

        qno = record["qno"]

        lines.append("")
        lines.append(
            "#" * 80
        )

        lines.append(
            f"QUESTION Q{qno}"
        )

        lines.append(
            f"{record['question']}"
        )

        lines.append(
            "-" * 80
        )

        # Benchmark relationship.

        benchmark = record["benchmark"]

        if benchmark["known"]:

            lines.append(
                "EXPLICIT BENCHMARK RELATIONSHIP"
            )

            lines.append(
                f"Positive : "
                f"{benchmark['positive']}"
            )

            if benchmark["negatives"]:

                lines.append(
                    f"Negatives : "
                    f"{', '.join(benchmark['negatives'])}"
                )

            else:

                lines.append(
                    "Negatives : none explicitly specified"
                )

            lines.append("")

        # Local geometry.

        lines.append(
            "LOCAL RETRIEVAL GEOMETRY"
        )

        lines.append(
            f"Rank-1 similarity : "
            f"{record['rank1_score']:.8f}"
        )

        lines.append(
            f"Rank-2 similarity : "
            f"{record['rank2_score']:.8f}"
        )

        lines.append(
            f"Rank-1 / Rank-2 gap : "
            f"{record['rank1_rank2_gap']:.8f}"
        )

        lines.append("")

        # Top-5.

        lines.append(
            "TOP-5 RETRIEVAL"
        )

        lines.append(
            "-" * 80
        )

        for item in record["results"]:

            lines.append(
                f"Rank {item['rank']} | "
                f"{item['chunk_id']} | "
                f"Cosine = "
                f"{item['score']:.8f}"
            )

            lines.append(
                item["text"]
            )

            lines.append(
                "-" * 80
            )

    # ========================================================
    # END SUMMARY
    # ========================================================

    lines.append("")
    lines.append(
        "=" * 80
    )

    lines.append(
        "END OF BENCHMARK"
    )

    lines.append(
        "=" * 80
    )

    return "\n".join(lines)


# ============================================================
# 12. PROCESS ONE REPRESENTATION
# ============================================================

def process_representation(
    experiment_name,
    config
):

    print()
    print("=" * 80)
    print(
        f"PROCESSING {experiment_name}"
    )
    print("=" * 80)

    model_dir = config["model_dir"]
    embedding_file = config["embedding_file"]

    # --------------------------------------------------------
    # Validate model.
    # --------------------------------------------------------

    if not model_dir.exists():

        raise FileNotFoundError(
            f"Model directory not found:\n"
            f"{model_dir}"
        )

    # --------------------------------------------------------
    # Validate embeddings.
    # --------------------------------------------------------

    if not embedding_file.exists():

        raise FileNotFoundError(
            f"Embedding file not found:\n"
            f"{embedding_file}"
        )

    # --------------------------------------------------------
    # Load trained model.
    # --------------------------------------------------------

    print(
        f"Loading trained model: "
        f"{model_dir}"
    )

    model = SentenceTransformer(
        str(model_dir)
    )

    # --------------------------------------------------------
    # Load trained B01 embeddings.
    # --------------------------------------------------------

    corpus_embeddings = np.load(
        embedding_file
    )

    if corpus_embeddings.shape != (
        161,
        384
    ):

        raise ValueError(
            f"{experiment_name}: expected "
            f"(161, 384), got "
            f"{corpus_embeddings.shape}"
        )

    # --------------------------------------------------------
    # Normalize corpus embeddings.
    # --------------------------------------------------------

    norms = np.linalg.norm(
        corpus_embeddings,
        axis=1,
        keepdims=True
    )

    if np.any(norms == 0):

        raise ValueError(
            f"{experiment_name}: "
            f"zero-norm corpus embedding found."
        )

    corpus_embeddings = (
        corpus_embeddings / norms
    )

    # --------------------------------------------------------
    # Encode ALL 150 questions using the corresponding
    # trained model.
    # --------------------------------------------------------

    print(
        "Encoding 150 benchmark questions..."
    )

    query_embeddings = model.encode(
        questions["Question"].tolist(),
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
            f"{experiment_name}: expected query "
            f"shape (150, 384), got "
            f"{query_embeddings.shape}"
        )

    # --------------------------------------------------------
    # Similarity matrix.
    #
    # 150 queries × 161 corpus chunks
    # --------------------------------------------------------

    similarity_matrix = (
        query_embeddings
        @ corpus_embeddings.T
    )

    print(
        f"Similarity matrix: "
        f"{similarity_matrix.shape}"
    )

    # --------------------------------------------------------
    # Build complete retrieval records.
    # --------------------------------------------------------

    all_results = []

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

            retrieved.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "score": float(
                        scores[corpus_index]
                    ),
                    "text": chunk_lookup[
                        chunk_id
                    ],
                }
            )

        rank1_score = (
            retrieved[0]["score"]
        )

        rank2_score = (
            retrieved[1]["score"]
        )

        rank1_rank2_gap = (
            rank1_score
            - rank2_score
        )

        all_results.append(
            {
                "qno": qno,
                "question": question,
                "results": retrieved,
                "rank1_score": rank1_score,
                "rank2_score": rank2_score,
                "rank1_rank2_gap": rank1_rank2_gap,
                "benchmark":
                    benchmark_relationship(qno),
            }
        )

    # --------------------------------------------------------
    # Overall statistics.
    # --------------------------------------------------------

    statistics = calculate_statistics(
        all_results
    )

    # --------------------------------------------------------
    # Output files.
    # --------------------------------------------------------

    xml_file = (
        OUTPUTS
        / f"{experiment_name}_B01_All150.xml"
    )

    txt_file = (
        OUTPUTS
        / f"{experiment_name}_B01_All150.txt"
    )

    # --------------------------------------------------------
    # Write XML.
    # --------------------------------------------------------

    xml_content = create_xml(
        experiment_name,
        statistics,
        all_results
    )

    xml_file.write_text(
        xml_content,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Write TXT.
    # --------------------------------------------------------

    txt_content = create_txt(
        experiment_name,
        statistics,
        all_results
    )

    txt_file.write_text(
        txt_content,
        encoding="utf-8"
    )

    print()
    print(
        f"XML : {xml_file}"
    )

    print(
        f"TXT : {txt_file}"
    )

    print(
        f"Questions processed : "
        f"{len(all_results)}"
    )

    print(
        f"Rank-1 mean         : "
        f"{statistics['rank1_mean']:.8f}"
    )

    print(
        f"Rank-1/Rank-2 gap   : "
        f"{statistics['rank1_rank2_gap_mean']:.8f}"
    )


# ============================================================
# 13. RUN ALL FIVE
# ============================================================

for experiment_name, config in REPRESENTATIONS.items():

    process_representation(
        experiment_name,
        config
    )


# ============================================================
# 14. FINAL
# ============================================================

print()
print("=" * 80)
print("MODULE 5 — CONSOLIDATED OUTPUT GENERATION COMPLETE")
print("=" * 80)

print(
    "Representations : 5"
)

print(
    "Benchmark       : 150 questions"
)

print(
    "Top-K           : 5"
)

print(
    "XML files       : 5"
)

print(
    "TXT files       : 5"
)

print(
    "Total files     : 10"
)

print()
print("Generated:")

for experiment_name in REPRESENTATIONS:

    print(
        f"  {experiment_name}_B01_All150.xml"
    )

    print(
        f"  {experiment_name}_B01_All150.txt"
    )

print("=" * 80)