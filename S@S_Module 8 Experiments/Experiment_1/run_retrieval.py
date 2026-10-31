from pathlib import Path
import json
import random
import time

import numpy as np
import requests


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "datasets" / "scifact"
EMBED_DIR = BASE_DIR / "embeddings"
RESULTS_DIR = BASE_DIR / "results"

EMBED_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

CORPUS_FILE = DATA_DIR / "corpus.jsonl"
QUERIES_FILE = DATA_DIR / "queries.jsonl"

CORPUS_EMBEDDINGS_FILE = EMBED_DIR / "corpus_embeddings.npy"
CORPUS_IDS_FILE = EMBED_DIR / "corpus_ids.json"

SELECTED_QUERIES_FILE = RESULTS_DIR / "selected_queries.json"
RESULTS_FILE = RESULTS_DIR / "top5_results.json"

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL_NAME = "mxbai-embed-large:335m"

BATCH_SIZE = 32
TOP_K = 5
RANDOM_SEED = 42


# ============================================================
# Utility functions
# ============================================================

def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def get_embeddings(texts):
    """
    Generate embeddings using the local Ollama server.

    Texts are sent in batches to avoid creating one very large
    request.
    """

    all_embeddings = []

    total = len(texts)

    for start in range(0, total, BATCH_SIZE):

        batch = texts[start:start + BATCH_SIZE]

        payload = {
            "model": MODEL_NAME,
            "input": batch
        }

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )

        response.raise_for_status()

        data = response.json()

        embeddings = data["embeddings"]

        all_embeddings.extend(embeddings)

        end = min(start + BATCH_SIZE, total)

        print(
            f"Embedded {end}/{total}",
            end="\r",
            flush=True
        )

    print()

    return np.asarray(all_embeddings, dtype=np.float32)


def normalize_embeddings(embeddings):
    """
    L2-normalize embeddings so that dot product equals
    cosine similarity.
    """

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    return embeddings / np.maximum(norms, 1e-12)


# ============================================================
# Load corpus and queries
# ============================================================

print("\nLoading SciFact...")

corpus_records = load_jsonl(CORPUS_FILE)
query_records = load_jsonl(QUERIES_FILE)

print(f"Corpus documents : {len(corpus_records)}")
print(f"Total queries    : {len(query_records)}")


# ============================================================
# Prepare corpus
# ============================================================

corpus_ids = [
    record["_id"]
    for record in corpus_records
]

corpus_texts = [
    record["title"] + "\n" + record["text"]
    for record in corpus_records
]


# ============================================================
# Select the same 10 test queries
# ============================================================

# SciFact test query IDs are determined from the BEIR test qrels.
# We load the test query IDs from qrels ONLY to identify the test
# split. We do NOT read or use relevance judgments.

QRELS_FILE = DATA_DIR / "qrels" / "test.tsv"

test_query_ids = []

with open(QRELS_FILE, "r", encoding="utf-8") as f:

    next(f)  # skip header

    for line in f:

        if line.strip():

            query_id = line.split("\t")[0]

            if query_id not in test_query_ids:
                test_query_ids.append(query_id)


test_query_ids = set(test_query_ids)


test_queries = {
    record["_id"]: record["text"]
    for record in query_records
    if record["_id"] in test_query_ids
}


random.seed(RANDOM_SEED)

selected_ids = random.sample(
    list(test_queries.keys()),
    10
)

selected_queries = {
    query_id: test_queries[query_id]
    for query_id in selected_ids
}


with open(
    SELECTED_QUERIES_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        selected_queries,
        f,
        indent=2,
        ensure_ascii=False
    )


print("\nSelected queries:")

for rank, query_id in enumerate(selected_ids, start=1):

    print(
        f"{rank:2}. {query_id}: "
        f"{selected_queries[query_id]}"
    )


# ============================================================
# Corpus embeddings
# ============================================================

if CORPUS_EMBEDDINGS_FILE.exists():

    print("\nExisting corpus embeddings found.")
    print("Loading them instead of regenerating...")

    corpus_embeddings = np.load(
        CORPUS_EMBEDDINGS_FILE
    )

else:

    print("\nGenerating corpus embeddings...")
    print(f"Model: {MODEL_NAME}")
    print(f"Documents: {len(corpus_texts)}")

    start_time = time.time()

    corpus_embeddings = get_embeddings(
        corpus_texts
    )

    elapsed = time.time() - start_time

    print(
        f"Corpus embedding completed in "
        f"{elapsed / 60:.2f} minutes"
    )

    np.save(
        CORPUS_EMBEDDINGS_FILE,
        corpus_embeddings
    )

    with open(
        CORPUS_IDS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            corpus_ids,
            f
        )


print(
    f"\nEmbedding shape: "
    f"{corpus_embeddings.shape}"
)


# ============================================================
# Normalize corpus embeddings
# ============================================================

corpus_embeddings = normalize_embeddings(
    corpus_embeddings
)


# ============================================================
# Query embeddings
# ============================================================

query_texts = [
    selected_queries[query_id]
    for query_id in selected_ids
]

print("\nGenerating query embeddings...")

query_embeddings = get_embeddings(
    query_texts
)

query_embeddings = normalize_embeddings(
    query_embeddings
)


# ============================================================
# Retrieval
# ============================================================

print("\nRunning Top-5 retrieval...")

results = {}

for i, query_id in enumerate(selected_ids):

    query_vector = query_embeddings[i]

    # Because both vectors are L2-normalized,
    # dot product = cosine similarity.

    similarities = corpus_embeddings @ query_vector

    top_indices = np.argsort(
        similarities
    )[-TOP_K:][::-1]

    retrieved = []

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        record = corpus_records[index]

        retrieved.append(
            {
                "rank": rank,
                "corpus_id": record["_id"],
                "similarity": float(
                    similarities[index]
                ),
                "title": record["title"],
                "text": record["text"]
            }
        )

    results[query_id] = {
        "query": selected_queries[query_id],
        "results": retrieved
    }


# ============================================================
# Save results
# ============================================================

with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# Display results
# ============================================================

print("\n" + "=" * 80)
print("TOP-5 RETRIEVAL RESULTS")
print("=" * 80)

for query_number, query_id in enumerate(
    selected_ids,
    start=1
):

    print("\n")
    print("-" * 80)

    print(
        f"QUERY {query_number} "
        f"(ID: {query_id})"
    )

    print(
        selected_queries[query_id]
    )

    print("-" * 80)

    for result in results[query_id]["results"]:

        print(
            f"\nRank {result['rank']}"
        )

        print(
            f"Similarity: "
            f"{result['similarity']:.6f}"
        )

        print(
            f"Corpus ID: "
            f"{result['corpus_id']}"
        )

        print(
            f"Title: "
            f"{result['title']}"
        )

        print(
            f"Text: "
            f"{result['text']}"
        )


print("\n" + "=" * 80)

print(
    f"Results saved to:\n"
    f"{RESULTS_FILE}"
)

print("=" * 80)