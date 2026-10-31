from pathlib import Path
import json
import csv
import requests
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_DIR = PROJECT_ROOT / "Outputs"

CORPUS_FILE = DATA_DIR / "corpus.json"
QUERY_FILE = DATA_DIR / "queries.json"

OUTPUT_CSV = OUTPUT_DIR / "phase1_mxbai_retrieval.csv"
OUTPUT_TXT = OUTPUT_DIR / "phase1_mxbai_retrieval.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "mxbai-embed-large:335m"

TOP_K = 5

HIGH_THRESHOLD = 0.75
MEDIUM_THRESHOLD = 0.60


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# LOAD CORPUS
# ============================================================

def load_corpus():

    data = load_json(CORPUS_FILE)

    if "chunks" not in data:
        raise ValueError(
            "corpus.json does not contain 'chunks'."
        )

    chunks = data["chunks"]

    if not chunks:
        raise ValueError(
            "Corpus contains no chunks."
        )

    for i, chunk in enumerate(chunks):

        for field in ("chunk_id", "topic", "text"):

            if field not in chunk:
                raise ValueError(
                    f"Corpus chunk {i} is missing '{field}'."
                )

        if not str(chunk["text"]).strip():
            raise ValueError(
                f"Corpus chunk {i} has empty text."
            )

    return chunks


# ============================================================
# LOAD QUERIES
# ============================================================

def load_queries():

    data = load_json(QUERY_FILE)

    if not isinstance(data, list):
        raise ValueError(
            "queries.json must contain a list."
        )

    if not data:
        raise ValueError(
            "queries.json contains no queries."
        )

    for i, item in enumerate(data):

        for field in ("query_id", "query"):

            if field not in item:
                raise ValueError(
                    f"Query {i} is missing '{field}'."
                )

        if not str(item["query"]).strip():
            raise ValueError(
                f"Query {i} has empty text."
            )

    return data


# ============================================================
# EMBEDDING
# ============================================================

def embed_text(text):

    payload = {
        "model": EMBEDDING_MODEL,
        "prompt": text,
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120,
        )

    except requests.RequestException as exc:

        raise RuntimeError(
            "Could not connect to Ollama. "
            "Make sure Ollama is running."
        ) from exc

    if response.status_code != 200:

        raise RuntimeError(
            f"Ollama request failed: "
            f"{response.status_code}\n"
            f"{response.text}"
        )

    data = response.json()

    if "embedding" not in data:
        raise RuntimeError(
            "Ollama response contains no embedding."
        )

    return np.asarray(
        data["embedding"],
        dtype=np.float32,
    )


def embed_corpus(chunks):

    print()
    print("Embedding corpus...")

    embeddings = []

    for i, chunk in enumerate(chunks, start=1):

        print(
            f"[{i:>3}/{len(chunks)}] "
            f"{chunk['chunk_id']}"
        )

        embeddings.append(
            embed_text(chunk["text"])
        )

    return np.vstack(embeddings)


def embed_queries(queries):

    print()
    print("Embedding queries...")

    embeddings = []

    for i, item in enumerate(queries, start=1):

        print(
            f"[{i:>3}/{len(queries)}] "
            f"{item['query_id']}"
        )

        embeddings.append(
            embed_text(item["query"])
        )

    return np.vstack(embeddings)


# ============================================================
# COSINE RETRIEVAL
# ============================================================

def normalize_rows(matrix):

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0):
        raise ValueError(
            "Zero vector encountered."
        )

    return matrix / norms


def retrieve(
    query_embeddings,
    corpus_embeddings,
    chunks,
):

    queries_normalized = normalize_rows(
        query_embeddings
    )

    corpus_normalized = normalize_rows(
        corpus_embeddings
    )

    similarity_matrix = (
        queries_normalized
        @ corpus_normalized.T
    )

    all_results = []

    for query_index, scores in enumerate(
        similarity_matrix
    ):

        top_indices = np.argsort(
            scores
        )[::-1][:TOP_K]

        query_results = []

        for rank, corpus_index in enumerate(
            top_indices,
            start=1,
        ):

            chunk = chunks[corpus_index]

            query_results.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "topic": chunk["topic"],
                "score": float(
                    scores[corpus_index]
                ),
                "text": chunk["text"],
            })

        all_results.append(query_results)

    return all_results


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(queries, results):

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "query_id",
            "query",
            "rank",
            "chunk_id",
            "topic",
            "cosine_similarity",
            "text",
        ])

        for query, query_results in zip(
            queries,
            results,
        ):

            for result in query_results:

                writer.writerow([
                    query["query_id"],
                    query["query"],
                    result["rank"],
                    result["chunk_id"],
                    result["topic"],
                    f"{result['score']:.6f}",
                    result["text"],
                ])


# ============================================================
# SUMMARY
# ============================================================

def calculate_summary(results):

    top_scores = [
        query_results[0]["score"]
        for query_results in results
    ]

    high = sum(
        score >= HIGH_THRESHOLD
        for score in top_scores
    )

    medium = sum(
        MEDIUM_THRESHOLD <= score < HIGH_THRESHOLD
        for score in top_scores
    )

    low = sum(
        score < MEDIUM_THRESHOLD
        for score in top_scores
    )

    return high, medium, low


# ============================================================
# SAVE TXT
# ============================================================

def save_txt(
    queries,
    results,
    corpus_embeddings,
):

    high, medium, low = calculate_summary(
        results
    )

    total_questions = len(queries)

    lines = []

    lines.append("=" * 80)
    lines.append(
        "PHASE 1 — ORDINARY SEMANTIC RETRIEVAL"
    )
    lines.append("=" * 80)
    lines.append("")

    lines.append(
        f"Embedding Model      : {EMBEDDING_MODEL}"
    )
    lines.append(
        "Chunk Representation : ONE VECTOR"
    )
    lines.append(
        "Query Representation : ONE VECTOR"
    )
    lines.append(
        "Similarity            : COSINE"
    )
    lines.append(
        f"Corpus Chunks         : {len(corpus_embeddings)}"
    )
    lines.append(
        f"Embedding Dimension   : {corpus_embeddings.shape[1]}"
    )
    lines.append(
        f"Total Questions       : {total_questions}"
    )
    lines.append(
        f"Top-K                 : {TOP_K}"
    )

    lines.append("")
    lines.append("=" * 80)
    lines.append("RETRIEVAL SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    lines.append(
        f"Total Questions : {total_questions}"
    )

    lines.append(
        f"Top             : {high} "
        f"({high / total_questions * 100:.2f}%)"
    )

    lines.append(
        f"Medium          : {medium} "
        f"({medium / total_questions * 100:.2f}%)"
    )

    lines.append(
        f"Low             : {low} "
        f"({low / total_questions * 100:.2f}%)"
    )

    lines.append("")
    lines.append(
        "Top    : >= 0.75"
    )
    lines.append(
        "Medium : >= 0.60 and < 0.75"
    )
    lines.append(
        "Low    : < 0.60"
    )

    lines.append("")
    lines.append("=" * 80)
    lines.append("QUERY RESULTS")
    lines.append("=" * 80)

    for query, query_results in zip(
        queries,
        results,
    ):

        lines.append("")
        lines.append(
            f"QUERY {query['query_id']}"
        )
        lines.append(
            f"Question : {query['query']}"
        )
        lines.append("")

        for result in query_results:

            lines.append(
                f"Rank {result['rank']}"
            )
            lines.append(
                f"Chunk ID          : {result['chunk_id']}"
            )
            lines.append(
                f"Topic             : {result['topic']}"
            )
            lines.append(
                f"Cosine Similarity : {result['score']:.6f}"
            )
            lines.append(
                f"Text              : {result['text']}"
            )
            lines.append("")

    lines.append("=" * 80)
    lines.append(
        "END OF PHASE 1 RETRIEVAL"
    )
    lines.append("=" * 80)

    OUTPUT_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "PHASE 1 — ORDINARY SEMANTIC RETRIEVAL"
    )
    print("=" * 80)

    print()
    print(
        f"Project root : {PROJECT_ROOT}"
    )
    print(
        f"Corpus       : {CORPUS_FILE}"
    )
    print(
        f"Queries      : {QUERY_FILE}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    corpus = load_corpus()
    queries = load_queries()

    print()
    print(
        f"Validated corpus  : {len(corpus)} chunks"
    )
    print(
        f"Validated queries : {len(queries)}"
    )

    # --------------------------------------------------------
    # Embed
    # --------------------------------------------------------

    corpus_embeddings = embed_corpus(
        corpus
    )

    query_embeddings = embed_queries(
        queries
    )

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if (
        corpus_embeddings.shape[1]
        != query_embeddings.shape[1]
    ):
        raise ValueError(
            "Corpus and query embedding dimensions "
            "do not match."
        )

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    print()
    print("Running cosine retrieval...")

    results = retrieve(
        query_embeddings,
        corpus_embeddings,
        corpus,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_csv(
        queries,
        results,
    )

    save_txt(
        queries,
        results,
        corpus_embeddings,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    high, medium, low = calculate_summary(
        results
    )

    print()
    print("=" * 80)
    print("PHASE 1 COMPLETED")
    print("=" * 80)

    print()
    print(
        f"Top    : {high} "
        f"({high / len(queries) * 100:.2f}%)"
    )
    print(
        f"Medium : {medium} "
        f"({medium / len(queries) * 100:.2f}%)"
    )
    print(
        f"Low    : {low} "
        f"({low / len(queries) * 100:.2f}%)"
    )

    print()
    print(
        f"CSV : {OUTPUT_CSV}"
    )
    print(
        f"TXT : {OUTPUT_TXT}"
    )


if __name__ == "__main__":
    main()