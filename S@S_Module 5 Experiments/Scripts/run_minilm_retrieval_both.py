from pathlib import Path
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# BASE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "Inputs"
OUTPUT_DIR = BASE_DIR / "Outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5

HIGH_THRESHOLD = 0.75
MID_THRESHOLD = 0.60


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("=" * 70)
print("MODULE 5 — ALL-MiniLM-L6-v2 RETRIEVAL EXPERIMENT")
print("=" * 70)

print(f"\nLoading model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully.")


# ============================================================
# QUERY EMBEDDING
# ============================================================

def embed_query(question):

    embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=False
    )

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    norm = np.linalg.norm(embedding)

    if norm == 0:
        raise ValueError(
            "Query embedding has zero norm."
        )

    return embedding / norm


# ============================================================
# CORPUS NORMALIZATION
# ============================================================

def normalize_corpus(embeddings):

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
            "Corpus contains a zero-norm vector."
        )

    return embeddings / norms


# ============================================================
# CLASSIFICATION
# ============================================================

def classify(score):

    if score >= HIGH_THRESHOLD:
        return "HIGH"

    elif score >= MID_THRESHOLD:
        return "MEDIUM"

    else:
        return "LOW"


# ============================================================
# FIND QUESTION COLUMN
# ============================================================

def find_question_column(df):

    for column in df.columns:

        name = column.lower()

        if (
            "question" in name
            or "query" in name
            or "questioin" in name
        ):
            return column

    raise ValueError(
        "Could not identify the question column."
    )


# ============================================================
# B01 RETRIEVAL
# ============================================================

def run_b01():

    print("\n")
    print("=" * 70)
    print("B01 — 150 QUESTIONS")
    print("=" * 70)

    # --------------------------------------------------------
    # INPUT FILES
    # --------------------------------------------------------

    b01_dir = INPUT_DIR / "20260424_B01"

    corpus_file = (
        INPUT_DIR /
        "B01_all-MiniLM-L6-v2_embeddings.npy"
    )

    query_file = (
        INPUT_DIR /
        "Questions_B01_150.csv"
    )

    metadata_file = (
        b01_dir /
        "chunk_id_to_metadata.json"
    )

    index_file = (
        b01_dir /
        "index_to_chunk_id.json"
    )

    # --------------------------------------------------------
    # OUTPUT FILES
    # --------------------------------------------------------

    output_csv = (
        OUTPUT_DIR /
        "B01_all-MiniLM-L6-v2_retrieval.csv"
    )

    output_txt = (
        OUTPUT_DIR /
        "B01_all-MiniLM-L6-v2_retrieval.txt"
    )

    # --------------------------------------------------------
    # LOAD CORPUS
    # --------------------------------------------------------

    corpus = np.load(
        corpus_file
    )

    print(
        f"MiniLM corpus shape: {corpus.shape}"
    )

    if corpus.shape[1] != 384:

        raise ValueError(
            "Expected MiniLM corpus dimension = 384."
        )

    corpus = normalize_corpus(
        corpus
    )

    # --------------------------------------------------------
    # LOAD B01 METADATA
    # --------------------------------------------------------

    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    with open(
        index_file,
        "r",
        encoding="utf-8"
    ) as f:

        index_to_chunk = json.load(f)

    # --------------------------------------------------------
    # LOAD QUESTIONS
    # --------------------------------------------------------

    queries = pd.read_csv(
        query_file
    )

    question_column = find_question_column(
        queries
    )

    print(
        f"Question file: {query_file.name}"
    )

    print(
        f"Questions loaded: {len(queries)}"
    )

    if len(queries) != 150:

        print(
            f"WARNING: expected 150 questions, "
            f"but found {len(queries)}."
        )

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    results = []
    classifications = []

    for row_number, row in queries.iterrows():

        question_id = row_number + 1

        question = str(
            row[question_column]
        )

        print(
            f"Processing B01 "
            f"{question_id}/{len(queries)}",
            end="\r"
        )

        # SAME MiniLM model as corpus
        query_vector = embed_query(
            question
        )

        # Cosine similarity
        scores = corpus @ query_vector

        # TOP 5
        top_indices = np.argsort(
            -scores
        )[:TOP_K]

        top1_score = float(
            scores[top_indices[0]]
        )

        classifications.append({
            "Question_ID":
                question_id,

            "Top_1_Score":
                top1_score,

            "Classification":
                classify(top1_score)
        })

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            chunk_id = str(
                index_to_chunk[str(index)]
            )

            chunk_info = metadata.get(
                chunk_id,
                {}
            )

            results.append({

                "Question_ID":
                    question_id,

                "Question":
                    question,

                "Rank":
                    rank,

                "Corpus_Index":
                    int(index),

                "Chunk_ID":
                    chunk_id,

                "Cosine_Similarity":
                    float(scores[index]),

                "Header":
                    chunk_info.get(
                        "header",
                        ""
                    ),

                "Text":
                    chunk_info.get(
                        "text",
                        ""
                    )
            })

    print()

    results_df = pd.DataFrame(
        results
    )

    classification_df = pd.DataFrame(
        classifications
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    results_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # SAVE TXT
    # --------------------------------------------------------

    write_b01_txt(
        output_txt,
        results_df,
        classification_df,
        queries,
        question_column,
        corpus
    )

    print(
        f"\nB01 CSV: {output_csv}"
    )

    print(
        f"B01 TXT: {output_txt}"
    )


# ============================================================
# B01 TXT
# ============================================================

def write_b01_txt(
    output_file,
    results_df,
    classification_df,
    queries,
    question_column,
    corpus
):

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("=" * 80 + "\n")
        f.write(
            "B01 — ALL-MiniLM-L6-v2 "
            "COMPLETE RETRIEVAL OUTPUT\n"
        )
        f.write("=" * 80 + "\n\n")

        f.write(
            "Query File       : "
            "Questions_B01_150.csv\n"
        )

        f.write(
            f"Embedding Model  : "
            f"{MODEL_NAME}\n"
        )

        f.write(
            f"Corpus Vectors   : "
            f"{len(corpus)}\n"
        )

        f.write(
            f"Corpus Dimension : "
            f"{corpus.shape[1]}\n"
        )

        f.write(
            f"Total Questions  : "
            f"{len(queries)}\n"
        )

        f.write(
            f"Top-K            : "
            f"{TOP_K}\n"
        )

        f.write(
            f"HIGH Threshold   : "
            f">= {HIGH_THRESHOLD:.2f}\n"
        )

        f.write(
            f"MEDIUM Threshold : "
            f">= {MID_THRESHOLD:.2f} "
            f"and < {HIGH_THRESHOLD:.2f}\n"
        )

        f.write(
            f"LOW Threshold    : "
            f"< {MID_THRESHOLD:.2f}\n\n"
        )

        # ----------------------------------------------------
        # EVERY QUESTION
        # ----------------------------------------------------

        for question_id in range(
            1,
            len(queries) + 1
        ):

            row = queries.iloc[
                question_id - 1
            ]

            question = str(
                row[question_column]
            )

            q_results = results_df[
                results_df["Question_ID"]
                == question_id
            ].sort_values(
                "Rank"
            )

            classification = (
                classification_df[
                    classification_df[
                        "Question_ID"
                    ] == question_id
                ].iloc[0]
            )

            f.write("=" * 80 + "\n")
            f.write(
                f"QUESTION {question_id}\n"
            )
            f.write("=" * 80 + "\n")

            f.write(
                f"Question : {question}\n\n"
            )

            for _, result in q_results.iterrows():

                f.write("-" * 80 + "\n")

                f.write(
                    f"Rank              : "
                    f"{int(result['Rank'])}\n"
                )

                f.write(
                    f"Corpus Index      : "
                    f"{int(result['Corpus_Index'])}\n"
                )

                f.write(
                    f"Chunk ID          : "
                    f"{result['Chunk_ID']}\n"
                )

                f.write(
                    f"Cosine Similarity : "
                    f"{result['Cosine_Similarity']:.6f}\n"
                )

                f.write(
                    f"Header            : "
                    f"{result['Header']}\n"
                )

                f.write(
                    "Text              :\n"
                )

                f.write(
                    f"{result['Text']}\n"
                )

            f.write("\n")

            f.write("-" * 80 + "\n")
            f.write(
                "ROUTING CLASSIFICATION\n"
            )
            f.write("-" * 80 + "\n")

            f.write(
                f"Top-1 Score : "
                f"{classification['Top_1_Score']:.6f}\n"
            )

            f.write(
                f"Result      : "
                f"{classification['Classification']}\n"
            )

            f.write("\n")

        # ----------------------------------------------------
        # AGGREGATE STATISTICS
        # ----------------------------------------------------

        similarities = results_df[
            "Cosine_Similarity"
        ]

        top1 = classification_df[
            "Top_1_Score"
        ]

        f.write("=" * 80 + "\n")
        f.write(
            "AGGREGATE RETRIEVAL STATISTICS\n"
        )
        f.write("=" * 80 + "\n\n")

        f.write(
            f"Total Questions        : "
            f"{len(queries)}\n"
        )

        f.write(
            f"Corpus Vectors         : "
            f"{len(corpus)}\n"
        )

        f.write(
            f"Corpus Dimensions      : "
            f"{corpus.shape[1]}\n"
        )

        f.write(
            f"Top-K                  : "
            f"{TOP_K}\n"
        )

        f.write(
            f"Total Retrieved Chunks : "
            f"{len(results_df)}\n\n"
        )

        # ----------------------------------------------------
        # Overall cosine statistics
        # ----------------------------------------------------

        f.write(
            "ALL RETRIEVED COSINE SCORES\n"
        )

        f.write(
            "----------------------------\n"
        )

        f.write(
            f"Minimum : "
            f"{similarities.min():.6f}\n"
        )

        f.write(
            f"Maximum : "
            f"{similarities.max():.6f}\n"
        )

        f.write(
            f"Mean    : "
            f"{similarities.mean():.6f}\n"
        )

        f.write(
            f"Median  : "
            f"{similarities.median():.6f}\n\n"
        )

        # ----------------------------------------------------
        # Rank-wise statistics
        # ----------------------------------------------------

        f.write(
            "RANK-WISE RETRIEVAL STATISTICS\n"
        )

        f.write(
            "------------------------------\n"
        )

        for rank in range(
            1,
            TOP_K + 1
        ):

            scores = results_df.loc[
                results_df["Rank"] == rank,
                "Cosine_Similarity"
            ]

            f.write(
                f"Rank {rank}: "
                f"Mean={scores.mean():.6f}, "
                f"Median={scores.median():.6f}, "
                f"Min={scores.min():.6f}, "
                f"Max={scores.max():.6f}\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # Top-1 statistics
        # ----------------------------------------------------

        f.write(
            "TOP-1 SCORE STATISTICS\n"
        )

        f.write(
            "----------------------\n"
        )

        f.write(
            f"Minimum : "
            f"{top1.min():.6f}\n"
        )

        f.write(
            f"Maximum : "
            f"{top1.max():.6f}\n"
        )

        f.write(
            f"Mean    : "
            f"{top1.mean():.6f}\n"
        )

        f.write(
            f"Median  : "
            f"{top1.median():.6f}\n\n"
        )

        # ----------------------------------------------------
        # HIGH / MEDIUM / LOW
        # ----------------------------------------------------

        counts = classification_df[
            "Classification"
        ].value_counts()

        total = len(
            classification_df
        )

        high = int(
            counts.get("HIGH", 0)
        )

        medium = int(
            counts.get("MEDIUM", 0)
        )

        low = int(
            counts.get("LOW", 0)
        )

        f.write(
            "QUESTION-LEVEL RETRIEVAL "
            "CLASSIFICATION\n"
        )

        f.write(
            "----------------------------------------\n"
        )

        f.write(
            f"HIGH   : {high:>4} "
            f"({high / total * 100:.2f}%)\n"
        )

        f.write(
            f"MEDIUM : {medium:>4} "
            f"({medium / total * 100:.2f}%)\n"
        )

        f.write(
            f"LOW    : {low:>4} "
            f"({low / total * 100:.2f}%)\n"
        )

        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write(
            "END OF B01 RETRIEVAL OUTPUT\n"
        )
        f.write("=" * 80 + "\n")


# ============================================================
# SYNTHETIC RETRIEVAL
# ============================================================

def run_synthetic():

    print("\n")
    print("=" * 70)
    print(
        "SYNTHETIC — ALL-MiniLM-L6-v2 RETRIEVAL"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # INPUT FILES
    # --------------------------------------------------------

    corpus_file = (
        INPUT_DIR /
        "Synthetic_all-MiniLM-L6-v2_embeddings.npy"
    )

    query_file = (
        INPUT_DIR /
        "Query_Synthetic_Corpus.csv"
    )

    source_file = (
        INPUT_DIR /
        "Synthetic_Corpus.csv"
    )

    # --------------------------------------------------------
    # OUTPUT FILES
    # --------------------------------------------------------

    output_csv = (
        OUTPUT_DIR /
        "Synthetic_all-MiniLM-L6-v2_retrieval.csv"
    )

    output_txt = (
        OUTPUT_DIR /
        "Synthetic_all-MiniLM-L6-v2_retrieval.txt"
    )

    # --------------------------------------------------------
    # Load MiniLM corpus
    # --------------------------------------------------------

    corpus = np.load(
        corpus_file
    )

    print(
        f"MiniLM corpus shape: "
        f"{corpus.shape}"
    )

    if corpus.shape[1] != 384:

        raise ValueError(
            "Expected MiniLM corpus dimension = 384."
        )

    corpus = normalize_corpus(
        corpus
    )

    # --------------------------------------------------------
    # Load original Synthetic corpus
    # --------------------------------------------------------

    source_df = pd.read_csv(
        source_file
    )

    text_column = source_df.columns[3]

    texts = (
        source_df[text_column]
        .astype(str)
        .tolist()
    )

    if len(texts) != len(corpus):

        raise ValueError(
            "Synthetic text count does not "
            "match embedding count."
        )

    # --------------------------------------------------------
    # Load questions
    # --------------------------------------------------------

    queries = pd.read_csv(
        query_file
    )

    question_column = find_question_column(
        queries
    )

    print(
        f"Questions loaded: {len(queries)}"
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    results = []
    classifications = []

    for row_number, row in queries.iterrows():

        question_id = row_number + 1

        question = str(
            row[question_column]
        )

        category = ""

        if "Category" in queries.columns:

            category = str(
                row["Category"]
            )

        print(
            f"Processing Synthetic "
            f"{question_id}/{len(queries)}",
            end="\r"
        )

        # SAME MiniLM model as corpus
        query_vector = embed_query(
            question
        )

        scores = corpus @ query_vector

        # TOP 5
        top_indices = np.argsort(
            -scores
        )[:TOP_K]

        top1_score = float(
            scores[top_indices[0]]
        )

        classifications.append({
            "Question_ID":
                question_id,

            "Top_1_Score":
                top1_score,

            "Classification":
                classify(top1_score)
        })

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            results.append({

                "Question_ID":
                    question_id,

                "Category":
                    category,

                "Question":
                    question,

                "Rank":
                    rank,

                "Corpus_Index":
                    int(index),

                "Chunk_ID":
                    f"chunk_{index}",

                "Cosine_Similarity":
                    float(scores[index]),

                "Text":
                    texts[index]
            })

    print()

    results_df = pd.DataFrame(
        results
    )

    classification_df = pd.DataFrame(
        classifications
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    results_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # SAVE TXT
    # --------------------------------------------------------

    write_synthetic_txt(
        output_txt,
        results_df,
        classification_df,
        queries,
        question_column,
        corpus
    )

    print(
        f"\nSynthetic CSV: {output_csv}"
    )

    print(
        f"Synthetic TXT: {output_txt}"
    )


# ============================================================
# SYNTHETIC TXT
# ============================================================

def write_synthetic_txt(
    output_file,
    results_df,
    classification_df,
    queries,
    question_column,
    corpus
):

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("=" * 80 + "\n")
        f.write(
            "SYNTHETIC — ALL-MiniLM-L6-v2 "
            "COMPLETE RETRIEVAL OUTPUT\n"
        )
        f.write("=" * 80 + "\n\n")

        f.write(
            "Query File       : "
            "Query_Synthetic_Corpus.csv\n"
        )

        f.write(
            f"Embedding Model  : "
            f"{MODEL_NAME}\n"
        )

        f.write(
            f"Corpus Vectors   : "
            f"{len(corpus)}\n"
        )

        f.write(
            f"Corpus Dimension : "
            f"{corpus.shape[1]}\n"
        )

        f.write(
            f"Total Questions  : "
            f"{len(queries)}\n"
        )

        f.write(
            f"Top-K            : "
            f"{TOP_K}\n"
        )

        f.write(
            f"HIGH Threshold   : "
            f">= {HIGH_THRESHOLD:.2f}\n"
        )

        f.write(
            f"MEDIUM Threshold : "
            f">= {MID_THRESHOLD:.2f} "
            f"and < {HIGH_THRESHOLD:.2f}\n"
        )

        f.write(
            f"LOW Threshold    : "
            f"< {MID_THRESHOLD:.2f}\n\n"
        )

        # ----------------------------------------------------
        # EVERY QUESTION
        # ----------------------------------------------------

        for question_id in range(
            1,
            len(queries) + 1
        ):

            row = queries.iloc[
                question_id - 1
            ]

            question = str(
                row[question_column]
            )

            category = ""

            if "Category" in queries.columns:

                category = str(
                    row["Category"]
                )

            q_results = results_df[
                results_df["Question_ID"]
                == question_id
            ].sort_values(
                "Rank"
            )

            classification = (
                classification_df[
                    classification_df[
                        "Question_ID"
                    ] == question_id
                ].iloc[0]
            )

            f.write("=" * 80 + "\n")
            f.write(
                f"QUESTION {question_id}\n"
            )
            f.write("=" * 80 + "\n")

            f.write(
                f"Category : {category}\n"
            )

            f.write(
                f"Question : {question}\n\n"
            )

            for _, result in q_results.iterrows():

                f.write("-" * 80 + "\n")

                f.write(
                    f"Rank              : "
                    f"{int(result['Rank'])}\n"
                )

                f.write(
                    f"Corpus Index      : "
                    f"{int(result['Corpus_Index'])}\n"
                )

                f.write(
                    f"Chunk ID          : "
                    f"{result['Chunk_ID']}\n"
                )

                f.write(
                    f"Cosine Similarity : "
                    f"{result['Cosine_Similarity']:.6f}\n"
                )

                f.write(
                    "Text              :\n"
                )

                f.write(
                    f"{result['Text']}\n"
                )

            f.write("\n")

            f.write("-" * 80 + "\n")
            f.write(
                "ROUTING CLASSIFICATION\n"
            )
            f.write("-" * 80 + "\n")

            f.write(
                f"Top-1 Score : "
                f"{classification['Top_1_Score']:.6f}\n"
            )

            f.write(
                f"Result      : "
                f"{classification['Classification']}\n"
            )

            f.write("\n")

        # ----------------------------------------------------
        # AGGREGATE STATISTICS
        # ----------------------------------------------------

        similarities = results_df[
            "Cosine_Similarity"
        ]

        top1 = classification_df[
            "Top_1_Score"
        ]

        f.write("=" * 80 + "\n")
        f.write(
            "AGGREGATE RETRIEVAL STATISTICS\n"
        )
        f.write("=" * 80 + "\n\n")

        f.write(
            f"Total Questions        : "
            f"{len(queries)}\n"
        )

        f.write(
            f"Corpus Vectors         : "
            f"{len(corpus)}\n"
        )

        f.write(
            f"Corpus Dimensions      : "
            f"{corpus.shape[1]}\n"
        )

        f.write(
            f"Top-K                  : "
            f"{TOP_K}\n"
        )

        f.write(
            f"Total Retrieved Chunks : "
            f"{len(results_df)}\n\n"
        )

        f.write(
            "ALL RETRIEVED COSINE SCORES\n"
        )

        f.write(
            "----------------------------\n"
        )

        f.write(
            f"Minimum : "
            f"{similarities.min():.6f}\n"
        )

        f.write(
            f"Maximum : "
            f"{similarities.max():.6f}\n"
        )

        f.write(
            f"Mean    : "
            f"{similarities.mean():.6f}\n"
        )

        f.write(
            f"Median  : "
            f"{similarities.median():.6f}\n\n"
        )

        # ----------------------------------------------------
        # Rank-wise
        # ----------------------------------------------------

        f.write(
            "RANK-WISE RETRIEVAL STATISTICS\n"
        )

        f.write(
            "------------------------------\n"
        )

        for rank in range(
            1,
            TOP_K + 1
        ):

            scores = results_df.loc[
                results_df["Rank"] == rank,
                "Cosine_Similarity"
            ]

            f.write(
                f"Rank {rank}: "
                f"Mean={scores.mean():.6f}, "
                f"Median={scores.median():.6f}, "
                f"Min={scores.min():.6f}, "
                f"Max={scores.max():.6f}\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # Top-1
        # ----------------------------------------------------

        f.write(
            "TOP-1 SCORE STATISTICS\n"
        )

        f.write(
            "----------------------\n"
        )

        f.write(
            f"Minimum : "
            f"{top1.min():.6f}\n"
        )

        f.write(
            f"Maximum : "
            f"{top1.max():.6f}\n"
        )

        f.write(
            f"Mean    : "
            f"{top1.mean():.6f}\n"
        )

        f.write(
            f"Median  : "
            f"{top1.median():.6f}\n\n"
        )

        # ----------------------------------------------------
        # HIGH / MEDIUM / LOW
        # ----------------------------------------------------

        counts = classification_df[
            "Classification"
        ].value_counts()

        total = len(
            classification_df
        )

        high = int(
            counts.get("HIGH", 0)
        )

        medium = int(
            counts.get("MEDIUM", 0)
        )

        low = int(
            counts.get("LOW", 0)
        )

        f.write(
            "QUESTION-LEVEL RETRIEVAL "
            "CLASSIFICATION\n"
        )

        f.write(
            "----------------------------------------\n"
        )

        f.write(
            f"HIGH   : {high:>4} "
            f"({high / total * 100:.2f}%)\n"
        )

        f.write(
            f"MEDIUM : {medium:>4} "
            f"({medium / total * 100:.2f}%)\n"
        )

        f.write(
            f"LOW    : {low:>4} "
            f"({low / total * 100:.2f}%)\n"
        )

        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write(
            "END OF SYNTHETIC RETRIEVAL OUTPUT\n"
        )
        f.write("=" * 80 + "\n")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_b01()

    run_synthetic()

    print("\n")
    print("=" * 70)
    print(
        "BOTH MiniLM RETRIEVAL EXPERIMENTS COMPLETE"
    )
    print("=" * 70)

    print("\nExpected outputs:")

    print(
        "Outputs/B01_all-MiniLM-L6-v2_retrieval.csv"
    )

    print(
        "Outputs/B01_all-MiniLM-L6-v2_retrieval.txt"
    )

    print(
        "Outputs/Synthetic_all-MiniLM-L6-v2_retrieval.csv"
    )

    print(
        "Outputs/Synthetic_all-MiniLM-L6-v2_retrieval.txt"
    )