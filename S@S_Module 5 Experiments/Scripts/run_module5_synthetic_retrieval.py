from pathlib import Path
import csv
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "Inputs"
MODEL_DIR = BASE_DIR / "Outputs" / "Models"
OUTPUT_DIR = BASE_DIR / "Outputs"

CORPUS_FILE = INPUT_DIR / "Synthetic_Corpus.csv"
QUERY_FILE = INPUT_DIR / "Query_Synthetic_Corpus.csv"


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

TOP_K = 5

MODELS = {
    "E2": (
        "Contrastive",
        MODEL_DIR / "E2_Contrastive_MiniLM",
        OUTPUT_DIR / "E2_Contrastive_Synthetic_MiniLM_retrieval.txt",
    ),
    "E3": (
        "Triplet",
        MODEL_DIR / "E3_Triplet_MiniLM",
        OUTPUT_DIR / "E3_Triplet_Synthetic_MiniLM_retrieval.txt",
    ),
    "E4": (
        "InfoNCE",
        MODEL_DIR / "E4_InfoNCE_MiniLM",
        OUTPUT_DIR / "E4_InfoNCE_Synthetic_MiniLM_retrieval.txt",
    ),
    "E5": (
        "HardNegative",
        MODEL_DIR / "E5_HardNegative_MiniLM",
        OUTPUT_DIR / "E5_HardNegative_Synthetic_MiniLM_retrieval.txt",
    ),
    "E6": (
        "AngularMargin",
        MODEL_DIR / "E6_AngularMargin_MiniLM",
        OUTPUT_DIR / "E6_AngularMargin_Synthetic_MiniLM_retrieval.txt",
    ),
}


# ============================================================
# HELPERS
# ============================================================

def cosine_similarity_matrix(query_embeddings, corpus_embeddings):
    """
    Compute cosine similarity between every query and corpus vector.
    """
    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    corpus_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    query_norms[query_norms == 0] = 1.0
    corpus_norms[corpus_norms == 0] = 1.0

    q = query_embeddings / query_norms
    c = corpus_embeddings / corpus_norms

    return np.matmul(q, c.T)


def classify_score(score):
    if score >= 0.75:
        return "HIGH"
    elif score >= 0.60:
        return "MEDIUM"
    else:
        return "LOW"


def find_column(fieldnames, candidates):
    """
    Return the first matching column using case-insensitive matching.
    """
    normalized = {
        str(name).strip().lower(): name
        for name in fieldnames
    }

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    return None


# ============================================================
# LOAD SYNTHETIC CORPUS
# ============================================================

def load_synthetic_corpus():

    rows = []

    with open(CORPUS_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            # Ignore completely empty CSV records.
            if not row or all(not str(value).strip() for value in row):
                continue

            if len(row) < 4:
                raise ValueError(
                    f"Synthetic corpus row has fewer than 4 columns: {row}"
                )

            level_1 = row[0].strip()
            level_2 = row[1].strip()
            chunk_title = row[2].strip()
            text = row[3].strip()

            rows.append({
                "level_1": level_1,
                "level_2": level_2,
                "chunk_title": chunk_title,
                "text": text,
            })

    if len(rows) != 50:
        raise ValueError(
            f"Synthetic corpus validation failed. "
            f"Expected 50 rows, found {len(rows)}."
        )

    return rows


# ============================================================
# LOAD SYNTHETIC QUESTIONS
# ============================================================

def load_synthetic_questions():

    questions = []

    with open(QUERY_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("Synthetic query file has no header.")

        question_column = find_column(
            reader.fieldnames,
            [
                "question",
                "query",
                "questioin",
            ],
        )

        category_column = find_column(
            reader.fieldnames,
            [
                "category",
                "Category",
            ],
        )

        if question_column is None:
            raise ValueError(
                "Could not find a question column. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:

            question = str(row[question_column]).strip()

            if not question:
                continue

            category = ""

            if category_column is not None:
                category = str(row[category_column]).strip()

            questions.append({
                "category": category,
                "question": question,
            })

    if len(questions) != 25:
        raise ValueError(
            f"Synthetic question validation failed. "
            f"Expected 25 questions, found {len(questions)}."
        )

    return questions


# ============================================================
# BUILD RETRIEVAL RESULTS
# ============================================================

def retrieve(
    model,
    corpus_rows,
    questions,
):

    corpus_texts = [
        row["text"]
        for row in corpus_rows
    ]

    question_texts = [
        row["question"]
        for row in questions
    ]

    print("Encoding synthetic corpus...")

    corpus_embeddings = model.encode(
        corpus_texts,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )

    print("Encoding synthetic questions...")

    query_embeddings = model.encode(
        question_texts,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )

    if corpus_embeddings.shape[0] != 50:
        raise ValueError(
            f"Expected 50 corpus embeddings, "
            f"found {corpus_embeddings.shape[0]}"
        )

    if query_embeddings.shape[0] != 25:
        raise ValueError(
            f"Expected 25 query embeddings, "
            f"found {query_embeddings.shape[0]}"
        )

    if corpus_embeddings.shape[1] != 384:
        raise ValueError(
            f"Expected MiniLM dimension 384, "
            f"found {corpus_embeddings.shape[1]}"
        )

    if query_embeddings.shape[1] != 384:
        raise ValueError(
            f"Expected MiniLM dimension 384, "
            f"found {query_embeddings.shape[1]}"
        )

    similarities = cosine_similarity_matrix(
        query_embeddings,
        corpus_embeddings,
    )

    all_results = []

    for q_index, question in enumerate(questions):

        scores = similarities[q_index]

        top_indices = np.argsort(scores)[::-1][:TOP_K]

        question_results = []

        for rank, corpus_index in enumerate(top_indices, start=1):

            row = corpus_rows[corpus_index]
            score = float(scores[corpus_index])

            question_results.append({
                "rank": rank,
                "corpus_index": int(corpus_index),
                "chunk_id": f"chunk_{corpus_index}",
                "score": score,
                "text": row["text"],
            })

        all_results.append({
            "category": question["category"],
            "question": question["question"],
            "results": question_results,
        })

    return all_results


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(all_results):

    all_scores = []
    rank_scores = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
    }

    top1_scores = []

    for question_result in all_results:

        for result in question_result["results"]:

            score = result["score"]

            all_scores.append(score)
            rank_scores[result["rank"]].append(score)

            if result["rank"] == 1:
                top1_scores.append(score)

    def stats(values):

        values = np.asarray(values, dtype=float)

        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    high = sum(
        1
        for score in top1_scores
        if score >= 0.75
    )

    medium = sum(
        1
        for score in top1_scores
        if 0.60 <= score < 0.75
    )

    low = sum(
        1
        for score in top1_scores
        if score < 0.60
    )

    return {
        "all": stats(all_scores),
        "rank": {
            rank: stats(values)
            for rank, values in rank_scores.items()
        },
        "top1": stats(top1_scores),
        "high": high,
        "medium": medium,
        "low": low,
    }


# ============================================================
# WRITE OUTPUT
# ============================================================

def write_output(
    output_file,
    intervention_name,
    all_results,
    statistics,
):

    lines = []

    lines.append("=" * 80)
    lines.append(
        f"SYNTHETIC — {intervention_name.upper()} MiniLM "
        f"COMPLETE RETRIEVAL OUTPUT"
    )
    lines.append("=" * 80)
    lines.append("")
    lines.append(
        "Query File       : Query_Synthetic_Corpus.csv"
    )
    lines.append(
        f"Embedding Model  : {intervention_name} MiniLM"
    )
    lines.append(
        "Corpus Vectors   : 50"
    )
    lines.append(
        "Corpus Dimension : 384"
    )
    lines.append(
        "Total Questions  : 25"
    )
    lines.append(
        "Top-K            : 5"
    )
    lines.append(
        "HIGH Threshold   : >= 0.75"
    )
    lines.append(
        "MEDIUM Threshold : >= 0.60 and < 0.75"
    )
    lines.append(
        "LOW Threshold    : < 0.60"
    )
    lines.append("")

    lines.append("=" * 80)
    lines.append("AGGREGATE RETRIEVAL STATISTICS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Total Questions        : 25")
    lines.append("Corpus Vectors         : 50")
    lines.append("Corpus Dimensions      : 384")
    lines.append("Top-K                  : 5")
    lines.append("Total Retrieved Chunks : 125")
    lines.append("")

    lines.append("ALL RETRIEVED COSINE SCORES")
    lines.append("----------------------------")
    lines.append(
        f"Minimum : {statistics['all']['min']:.6f}"
    )
    lines.append(
        f"Maximum : {statistics['all']['max']:.6f}"
    )
    lines.append(
        f"Mean    : {statistics['all']['mean']:.6f}"
    )
    lines.append(
        f"Median  : {statistics['all']['median']:.6f}"
    )
    lines.append("")

    lines.append("RANK-WISE RETRIEVAL STATISTICS")
    lines.append("------------------------------")

    for rank in range(1, 6):

        s = statistics["rank"][rank]

        lines.append(
            f"Rank {rank}: "
            f"Mean={s['mean']:.6f}, "
            f"Median={s['median']:.6f}, "
            f"Min={s['min']:.6f}, "
            f"Max={s['max']:.6f}"
        )

    lines.append("")

    lines.append("TOP-1 SCORE STATISTICS")
    lines.append("----------------------")
    lines.append(
        f"Minimum : {statistics['top1']['min']:.6f}"
    )
    lines.append(
        f"Maximum : {statistics['top1']['max']:.6f}"
    )
    lines.append(
        f"Mean    : {statistics['top1']['mean']:.6f}"
    )
    lines.append(
        f"Median  : {statistics['top1']['median']:.6f}"
    )
    lines.append("")

    lines.append(
        "QUESTION-LEVEL RETRIEVAL CLASSIFICATION"
    )
    lines.append("----------------------------------------")
    lines.append(
        f"HIGH   : {statistics['high']:4d} "
        f"({statistics['high'] / 25 * 100:.2f}%)"
    )
    lines.append(
        f"MEDIUM : {statistics['medium']:4d} "
        f"({statistics['medium'] / 25 * 100:.2f}%)"
    )
    lines.append(
        f"LOW    : {statistics['low']:4d} "
        f"({statistics['low'] / 25 * 100:.2f}%)"
    )
    lines.append("")

    lines.append("=" * 80)

    # ========================================================
    # QUESTION RESULTS
    # ========================================================

    for q_number, question_result in enumerate(
        all_results,
        start=1,
    ):

        lines.append("")
        lines.append("=" * 80)
        lines.append(
            f"QUESTION {q_number}"
        )
        lines.append("=" * 80)
        lines.append("")

        lines.append(
            f"Category : {question_result['category']}"
        )

        lines.append(
            f"Question : {question_result['question']}"
        )

        lines.append("")

        for result in question_result["results"]:

            routing = classify_score(result["score"])

            lines.append(
                f"Rank {result['rank']}"
            )
            lines.append(
                f"Corpus Index       : {result['corpus_index']}"
            )
            lines.append(
                f"Chunk ID            : {result['chunk_id']}"
            )
            lines.append(
                f"Cosine Similarity   : {result['score']:.6f}"
            )
            lines.append(
                f"Text                : {result['text']}"
            )
            lines.append(
                f"Routing Classification : {routing}"
            )
            lines.append("")

    # ========================================================
    # END STATISTICS — kept to match baseline structure
    # ========================================================

    lines.append("")
    lines.append("=" * 80)
    lines.append("AGGREGATE RETRIEVAL STATISTICS")
    lines.append("=" * 80)
    lines.append("")

    lines.append(
        f"Total Questions        : 25"
    )
    lines.append(
        f"Corpus Vectors         : 50"
    )
    lines.append(
        f"Corpus Dimensions      : 384"
    )
    lines.append(
        f"Top-K                  : 5"
    )
    lines.append(
        f"Total Retrieved Chunks : 125"
    )
    lines.append("")

    lines.append("ALL RETRIEVED COSINE SCORES")
    lines.append("----------------------------")
    lines.append(
        f"Minimum : {statistics['all']['min']:.6f}"
    )
    lines.append(
        f"Maximum : {statistics['all']['max']:.6f}"
    )
    lines.append(
        f"Mean    : {statistics['all']['mean']:.6f}"
    )
    lines.append(
        f"Median  : {statistics['all']['median']:.6f}"
    )
    lines.append("")

    lines.append("RANK-WISE RETRIEVAL STATISTICS")
    lines.append("------------------------------")

    for rank in range(1, 6):

        s = statistics["rank"][rank]

        lines.append(
            f"Rank {rank}: "
            f"Mean={s['mean']:.6f}, "
            f"Median={s['median']:.6f}, "
            f"Min={s['min']:.6f}, "
            f"Max={s['max']:.6f}"
        )

    lines.append("")

    lines.append("TOP-1 SCORE STATISTICS")
    lines.append("----------------------")
    lines.append(
        f"Minimum : {statistics['top1']['min']:.6f}"
    )
    lines.append(
        f"Maximum : {statistics['top1']['max']:.6f}"
    )
    lines.append(
        f"Mean    : {statistics['top1']['mean']:.6f}"
    )
    lines.append(
        f"Median  : {statistics['top1']['median']:.6f}"
    )
    lines.append("")

    lines.append(
        "QUESTION-LEVEL RETRIEVAL CLASSIFICATION"
    )
    lines.append("----------------------------------------")
    lines.append(
        f"HIGH   : {statistics['high']:4d} "
        f"({statistics['high'] / 25 * 100:.2f}%)"
    )
    lines.append(
        f"MEDIUM : {statistics['medium']:4d} "
        f"({statistics['medium'] / 25 * 100:.2f}%)"
    )
    lines.append(
        f"LOW    : {statistics['low']:4d} "
        f"({statistics['low'] / 25 * 100:.2f}%)"
    )
    lines.append("")

    lines.append("=" * 80)
    lines.append("END OF SYNTHETIC RETRIEVAL OUTPUT")
    lines.append("=" * 80)

    output_file.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("MODULE 5 — SYNTHETIC VALIDATION RETRIEVAL")
    print("=" * 80)
    print()

    print(f"Base directory : {BASE_DIR}")
    print(f"Corpus file    : {CORPUS_FILE}")
    print(f"Query file     : {QUERY_FILE}")
    print()

    corpus_rows = load_synthetic_corpus()
    questions = load_synthetic_questions()

    print(
        f"Validated synthetic corpus : {len(corpus_rows)} chunks"
    )
    print(
        f"Validated synthetic queries : {len(questions)} questions"
    )
    print()

    for model_id, (
        intervention_name,
        model_path,
        output_file,
    ) in MODELS.items():

        print("=" * 80)
        print(
            f"{model_id} — {intervention_name} MiniLM"
        )
        print("=" * 80)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model directory not found:\n{model_path}"
            )

        print(f"Loading model: {model_path}")

        model = SentenceTransformer(
            str(model_path)
        )

        all_results = retrieve(
            model,
            corpus_rows,
            questions,
        )

        statistics = calculate_statistics(
            all_results
        )

        write_output(
            output_file,
            intervention_name,
            all_results,
            statistics,
        )

        print()
        print(
            f"Output written to:\n{output_file}"
        )
        print()

        # Explicitly release model before loading next model.
        del model

    print("=" * 80)
    print("ALL FIVE SYNTHETIC RETRIEVAL EXPERIMENTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()