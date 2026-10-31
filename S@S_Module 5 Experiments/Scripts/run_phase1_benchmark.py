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

OUTPUT_CSV = OUTPUT_DIR / "phase1_benchmark_249_retrieval.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "phase1_benchmark_249_summary.txt"


# ============================================================
# SETTINGS
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "mxbai-embed-large:335m"

TOP_K = 5


# ============================================================
# LOAD B01 CORPUS
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
# LOAD CHUNK MAPPINGS
# ============================================================

with open(INDEX_TO_CHUNK_FILE, "r", encoding="utf-8") as f:
    index_to_chunk_id = json.load(f)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    chunk_metadata = json.load(f)


def get_chunk_id(index):
    return str(index_to_chunk_id[str(index)])


# ============================================================
# EMBEDDING FUNCTION
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
# LOAD QUESTIONS
# ============================================================

queries = pd.read_csv(QUERY_FILE)

print("\nQuery file columns:")
print(list(queries.columns))


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

print(f"Using question column: {question_column}")


# ============================================================
# RETRIEVAL
# ============================================================

results = []

print("\nRunning B01 retrieval...\n")

for row_number, row in queries.iterrows():

    question_id = row_number + 1
    question = str(row[question_column])

    query_embedding = embed_query(question)

    # Since both query and corpus vectors are normalized,
    # dot product = cosine similarity.
    scores = corpus_embeddings @ query_embedding

    # TOP 5
    top_indices = np.argsort(-scores)[:TOP_K]

    for rank, corpus_index in enumerate(top_indices, start=1):

        chunk_id = get_chunk_id(corpus_index)

        metadata = chunk_metadata.get(chunk_id, {})

        results.append({
            "Question_ID": question_id,
            "Question": question,
            "Rank": rank,
            "Corpus_Index": int(corpus_index),
            "Chunk_ID": chunk_id,
            "Cosine_Similarity": float(scores[corpus_index]),
            "Header": metadata.get("header", ""),
            "Text": metadata.get("text", "")
        })


# ============================================================
# SAVE CSV
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8"
)


# ============================================================
# CREATE SUMMARY
# ============================================================

similarities = results_df["Cosine_Similarity"]

summary_lines = [
    "B01 RETRIEVAL SUMMARY",
    "=====================",
    "",
    f"Query file              : {QUERY_FILE.name}",
    f"Embedding model         : {EMBEDDING_MODEL}",
    f"Corpus vectors          : {len(corpus_embeddings)}",
    f"Corpus dimensions       : {corpus_embeddings.shape[1]}",
    f"Total questions         : {len(queries)}",
    f"Top-K                   : {TOP_K}",
    f"Total retrieved results: {len(results_df)}",
    "",
    "Cosine similarity across all retrieved results",
    "-----------------------------------------------",
    f"Minimum                 : {similarities.min():.6f}",
    f"Maximum                 : {similarities.max():.6f}",
    f"Mean                    : {similarities.mean():.6f}",
    f"Median                  : {similarities.median():.6f}",
    "",
    "Results by rank",
    "---------------"
]

for rank in range(1, TOP_K + 1):

    rank_scores = results_df.loc[
        results_df["Rank"] == rank,
        "Cosine_Similarity"
    ]

    summary_lines.append(
        f"Rank {rank}: "
        f"Mean={rank_scores.mean():.6f}, "
        f"Min={rank_scores.min():.6f}, "
        f"Max={rank_scores.max():.6f}"
    )

summary_lines.extend([
    "",
    f"CSV output             : {OUTPUT_CSV}",
    f"Summary output         : {OUTPUT_SUMMARY}"
])


with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    f.write("\n".join(summary_lines))


# ============================================================
# FINISH
# ============================================================

print("==============================================")
print("B01 RETRIEVAL COMPLETE")
print("==============================================")
print(f"Questions          : {len(queries)}")
print(f"Corpus vectors     : {len(corpus_embeddings)}")
print(f"Top-K              : {TOP_K}")
print(f"Retrieved results  : {len(results_df)}")
print()
print(f"CSV     : {OUTPUT_CSV}")
print(f"Summary : {OUTPUT_SUMMARY}")
print("==============================================")