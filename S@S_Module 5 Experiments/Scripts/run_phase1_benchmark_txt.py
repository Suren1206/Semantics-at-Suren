from pathlib import Path
import json
import requests
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "Inputs"
B01_DIR = INPUT_DIR / "20260424_B01"

QUERY_FILE = INPUT_DIR / "Questions_B01_150.csv"

EMBEDDINGS_FILE = B01_DIR / "embeddings.npy"
INDEX_TO_CHUNK_FILE = B01_DIR / "index_to_chunk_id.json"
METADATA_FILE = B01_DIR / "chunk_id_to_metadata.json"

OUTPUT_DIR = BASE_DIR / "Outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TXT = OUTPUT_DIR / "phase1_benchmark_249_retrieval.txt"


# ============================================================
# SETTINGS
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "mxbai-embed-large:335m"

TOP_K = 5

HIGH_THRESHOLD = 0.75
MID_THRESHOLD = 0.60


# ============================================================
# LOAD CORPUS
# ============================================================

print("\nLoading B01 corpus...")

corpus_embeddings = np.load(EMBEDDINGS_FILE)

print(f"Corpus shape: {corpus_embeddings.shape}")

# Normalize corpus vectors
corpus_norms = np.linalg.norm(
    corpus_embeddings,
    axis=1,
    keepdims=True
)

corpus_embeddings = corpus_embeddings / corpus_norms


# ============================================================
# LOAD MAPPINGS
# ============================================================

with open(INDEX_TO_CHUNK_FILE, "r", encoding="utf-8") as f:
    index_to_chunk_id = json.load(f)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    chunk_metadata = json.load(f)


def get_chunk_id(index):
    return str(index_to_chunk_id[str(index)])


# ============================================================
# EMBEDD QUERY
# ============================================================

def embed_query(question):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": EMBEDDING_MODEL,
            "prompt": question
        },
        timeout=120
    )

    response.raise_for_status()

    embedding = np.array(
        response.json()["embedding"],
        dtype=np.float32
    )

    norm = np.linalg.norm(embedding)

    if norm == 0:
        raise ValueError("Query embedding has zero norm.")

    return embedding / norm


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
# LOAD QUESTIONS
# ============================================================

queries = pd.read_csv(QUERY_FILE)

question_column = None

for column in queries.columns:

    column_lower = column.lower()

    if (
        "question" in column_lower
        or "query" in column_lower
        or "questioin" in column_lower
    ):
        question_column = column
        break


if question_column is None:
    raise ValueError(
        "Could not identify the question column."
    )


# ============================================================
# RETRIEVAL
# ============================================================

all_results = []
question_classifications = []

print("\nRunning B01 retrieval...\n")

for row_number, row in queries.iterrows():

    question_id = row_number + 1
    question = str(row[question_column])

    query_embedding = embed_query(question)

    # Normalized dot product = cosine similarity
    scores = corpus_embeddings @ query_embedding

    top_indices = np.argsort(-scores)[:TOP_K]

    top_score = float(scores[top_indices[0]])

    classification = classify(top_score)

    question_classifications.append({
        "Question_ID": question_id,
        "Top_Score": top_score,
        "Classification": classification
    })

    for rank, corpus_index in enumerate(top_indices, start=1):

        chunk_id = get_chunk_id(corpus_index)
        metadata = chunk_metadata.get(chunk_id, {})

        all_results.append({
            "Question_ID": question_id,
            "Question": question,
            "Rank": rank,
            "Corpus_Index": int(corpus_index),
            "Chunk_ID": chunk_id,
            "Cosine_Similarity": float(scores[corpus_index]),
            "Header": metadata.get("header", ""),
            "Text": metadata.get("text", "")
        })


results_df = pd.DataFrame(all_results)
classification_df = pd.DataFrame(question_classifications)


# ============================================================
# WRITE COMPLETE TXT
# ============================================================

with open(
    OUTPUT_TXT,
    "w",
    encoding="utf-8"
) as f:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    f.write("=" * 80 + "\n")
    f.write("B01 — COMPLETE RETRIEVAL OUTPUT\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Query File       : {QUERY_FILE.name}\n")
    f.write(f"Embedding Model  : {EMBEDDING_MODEL}\n")
    f.write(f"Corpus Vectors   : {len(corpus_embeddings)}\n")
    f.write(f"Corpus Dimension : {corpus_embeddings.shape[1]}\n")
    f.write(f"Top-K            : {TOP_K}\n")
    f.write(f"HIGH Threshold   : >= {HIGH_THRESHOLD:.2f}\n")
    f.write(
        f"MEDIUM Threshold : >= {MID_THRESHOLD:.2f} and < {HIGH_THRESHOLD:.2f}\n"
    )
    f.write(f"LOW Threshold    : < {MID_THRESHOLD:.2f}\n\n")

    # --------------------------------------------------------
    # EVERY QUESTION
    # --------------------------------------------------------

    for question_id in range(1, len(queries) + 1):

        question_row = queries.iloc[question_id - 1]
        question = str(question_row[question_column])

        q_results = results_df[
            results_df["Question_ID"] == question_id
        ].sort_values("Rank")

        q_class = classification_df[
            classification_df["Question_ID"] == question_id
        ].iloc[0]

        f.write("=" * 80 + "\n")
        f.write(f"QUESTION {question_id}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Question : {question}\n\n")

        for _, result in q_results.iterrows():

            f.write("-" * 80 + "\n")
            f.write(f"Rank              : {int(result['Rank'])}\n")
            f.write(f"Corpus Index      : {int(result['Corpus_Index'])}\n")
            f.write(f"Chunk ID          : {result['Chunk_ID']}\n")
            f.write(
                f"Cosine Similarity : {result['Cosine_Similarity']:.6f}\n"
            )
            f.write(f"Header            : {result['Header']}\n")
            f.write("Text              :\n")
            f.write(f"{result['Text']}\n\n")

        f.write("-" * 80 + "\n")
        f.write("ROUTING CLASSIFICATION\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"Top-1 Score : {q_class['Top_Score']:.6f}\n"
        )
        f.write(
            f"Result      : {q_class['Classification']}\n"
        )
        f.write("-" * 80 + "\n\n")


    # ========================================================
    # AGGREGATE RETRIEVAL STATISTICS
    # ========================================================

    similarities = results_df["Cosine_Similarity"]

    f.write("\n")
    f.write("=" * 80 + "\n")
    f.write("AGGREGATE RETRIEVAL STATISTICS\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Total Questions          : {len(queries)}\n")
    f.write(f"Corpus Vectors           : {len(corpus_embeddings)}\n")
    f.write(f"Corpus Dimensions        : {corpus_embeddings.shape[1]}\n")
    f.write(f"Top-K                    : {TOP_K}\n")
    f.write(f"Total Retrieved Chunks   : {len(results_df)}\n\n")

    f.write("ALL RETRIEVED RESULTS\n")
    f.write("---------------------\n")
    f.write(
        f"Minimum Cosine          : {similarities.min():.6f}\n"
    )
    f.write(
        f"Maximum Cosine          : {similarities.max():.6f}\n"
    )
    f.write(
        f"Mean Cosine             : {similarities.mean():.6f}\n"
    )
    f.write(
        f"Median Cosine           : {similarities.median():.6f}\n\n"
    )


    # --------------------------------------------------------
    # RANK-WISE STATISTICS
    # --------------------------------------------------------

    f.write("RANK-WISE RETRIEVAL STATISTICS\n")
    f.write("------------------------------\n")

    for rank in range(1, TOP_K + 1):

        rank_scores = results_df.loc[
            results_df["Rank"] == rank,
            "Cosine_Similarity"
        ]

        f.write(
            f"Rank {rank}: "
            f"Mean={rank_scores.mean():.6f}, "
            f"Median={rank_scores.median():.6f}, "
            f"Min={rank_scores.min():.6f}, "
            f"Max={rank_scores.max():.6f}\n"
        )

    f.write("\n")


    # --------------------------------------------------------
    # HIGH / MEDIUM / LOW
    # --------------------------------------------------------

    counts = classification_df[
        "Classification"
    ].value_counts()

    total_questions = len(classification_df)

    high_count = int(counts.get("HIGH", 0))
    medium_count = int(counts.get("MEDIUM", 0))
    low_count = int(counts.get("LOW", 0))

    f.write("QUESTION-LEVEL RETRIEVAL CLASSIFICATION\n")
    f.write("----------------------------------------\n")

    f.write(
        f"HIGH   (>= {HIGH_THRESHOLD:.2f})"
        f" : {high_count:>4} questions"
        f" ({high_count / total_questions * 100:.2f}%)\n"
    )

    f.write(
        f"MEDIUM ({MID_THRESHOLD:.2f}–{HIGH_THRESHOLD:.2f})"
        f" : {medium_count:>4} questions"
        f" ({medium_count / total_questions * 100:.2f}%)\n"
    )

    f.write(
        f"LOW    (< {MID_THRESHOLD:.2f})"
        f" : {low_count:>4} questions"
        f" ({low_count / total_questions * 100:.2f}%)\n"
    )

    f.write("\n")


    # --------------------------------------------------------
    # TOP-1 SCORE DISTRIBUTION
    # --------------------------------------------------------

    top1_scores = classification_df["Top_Score"]

    f.write("TOP-1 SCORE STATISTICS\n")
    f.write("----------------------\n")

    f.write(
        f"Minimum Top-1 Score     : {top1_scores.min():.6f}\n"
    )
    f.write(
        f"Maximum Top-1 Score     : {top1_scores.max():.6f}\n"
    )
    f.write(
        f"Mean Top-1 Score        : {top1_scores.mean():.6f}\n"
    )
    f.write(
        f"Median Top-1 Score      : {top1_scores.median():.6f}\n"
    )
    f.write("\n")


    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    f.write("=" * 80 + "\n")
    f.write("END OF B01 RETRIEVAL OUTPUT\n")
    f.write("=" * 80 + "\n")


print("==============================================")
print("B01 TXT RETRIEVAL COMPLETE")
print("==============================================")
print(f"Questions : {len(queries)}")
print(f"Top-K     : {TOP_K}")
print(f"Output    : {OUTPUT_TXT}")
print("==============================================")