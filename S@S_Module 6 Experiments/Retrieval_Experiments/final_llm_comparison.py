from pathlib import Path
import json
import pandas as pd
import requests

from sentence_transformers import CrossEncoder


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

QUERIES_FILE = INPUT_DIR / "queries.json"
CORPUS_FILE = INPUT_DIR / "corpus.json"

DENSE_FILE = OUTPUT_DIR / "phase1_mxbai_retrieval.csv"
MAXSIM_FILE = OUTPUT_DIR / "phase4_maxsim_scores.csv"
COLBERT_FILE = OUTPUT_DIR / "phase6_colbert_top3.csv"

FINAL_OUTPUT = OUTPUT_DIR / "final_llm_response_comparison.txt"


# ============================================================
# MODELS
# ============================================================

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

TEMPERATURE = 0
MAX_TOKENS = 250


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


queries_data = load_json(QUERIES_FILE)
corpus_data = load_json(CORPUS_FILE)


# ============================================================
# EXTRACT QUERIES
# ============================================================

def extract_queries(data):

    if isinstance(data, dict):

        if "queries" in data:
            return data["queries"]

        raise ValueError(
            f"Could not find 'queries' in {QUERIES_FILE}"
        )

    if isinstance(data, list):
        return data

    raise ValueError(
        f"Unsupported query JSON structure in {QUERIES_FILE}"
    )


queries = extract_queries(queries_data)


# ============================================================
# EXTRACT CORPUS
# ============================================================

def extract_corpus(data):

    if isinstance(data, dict):

        if "chunks" in data:
            return data["chunks"]

        raise ValueError(
            f"Could not find 'chunks' in {CORPUS_FILE}"
        )

    if isinstance(data, list):
        return data

    raise ValueError(
        f"Unsupported corpus JSON structure in {CORPUS_FILE}"
    )


corpus = extract_corpus(corpus_data)


# ============================================================
# CORPUS LOOKUP
# ============================================================

def normalize_chunk_id(value):

    return str(value).strip()


corpus_lookup = {}

for chunk in corpus:

    chunk_id = normalize_chunk_id(
        chunk.get("chunk_id")
    )

    text = chunk.get("text", "")

    corpus_lookup[chunk_id] = text


# ============================================================
# QUERY LOOKUP
# ============================================================

query_lookup = {}

for item in queries:

    query_id = str(
        item.get(
            "query_id",
            item.get("id")
        )
    ).strip()

    query_text = item.get(
        "query",
        item.get("text", "")
    )

    query_lookup[query_id] = query_text


# ============================================================
# LOAD RETRIEVAL RESULTS
# ============================================================

dense_df = pd.read_csv(DENSE_FILE)
maxsim_df = pd.read_csv(MAXSIM_FILE)
colbert_df = pd.read_csv(COLBERT_FILE)


print("\nDense retrieval columns:")
print(list(dense_df.columns))

print("\nMaxSim columns:")
print(list(maxsim_df.columns))

print("\nColBERT columns:")
print(list(colbert_df.columns))


# ============================================================
# COLUMN IDENTIFICATION
# ============================================================

def find_column(df, candidates, description):

    lower_map = {
        str(col).lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise ValueError(
        f"Could not identify {description} column.\n"
        f"Available columns: {list(df.columns)}"
    )


# Dense

dense_query_col = find_column(
    dense_df,
    ["query_id", "qid", "query"],
    "dense query"
)

dense_chunk_col = find_column(
    dense_df,
    ["chunk_id", "chunk"],
    "dense chunk"
)

dense_rank_col = find_column(
    dense_df,
    ["rank", "retrieval_rank"],
    "dense rank"
)


# MaxSim

maxsim_query_col = find_column(
    maxsim_df,
    ["query_id", "qid", "query"],
    "MaxSim query"
)

maxsim_chunk_col = find_column(
    maxsim_df,
    ["chunk_id", "chunk"],
    "MaxSim chunk"
)

maxsim_rank_col = find_column(
    maxsim_df,
    ["rank", "retrieval_rank"],
    "MaxSim rank"
)


# ColBERT

colbert_query_col = find_column(
    colbert_df,
    ["query_id", "qid", "query"],
    "ColBERT query"
)

colbert_chunk_col = find_column(
    colbert_df,
    ["chunk_id", "chunk"],
    "ColBERT chunk"
)

colbert_rank_col = find_column(
    colbert_df,
    ["rank", "retrieval_rank"],
    "ColBERT rank"
)


# ============================================================
# NORMALIZE DATA TYPES
# ============================================================

for df, query_col, chunk_col, rank_col in [
    (
        dense_df,
        dense_query_col,
        dense_chunk_col,
        dense_rank_col
    ),
    (
        maxsim_df,
        maxsim_query_col,
        maxsim_chunk_col,
        maxsim_rank_col
    ),
    (
        colbert_df,
        colbert_query_col,
        colbert_chunk_col,
        colbert_rank_col
    )
]:

    df[query_col] = (
        df[query_col]
        .astype(str)
        .str.strip()
    )

    df[chunk_col] = (
        df[chunk_col]
        .astype(str)
        .str.strip()
    )

    df[rank_col] = pd.to_numeric(
        df[rank_col],
        errors="coerce"
    )


# ============================================================
# LOAD CROSS-ENCODER
# ============================================================

print("\nLoading Cross-Encoder...")

cross_encoder = CrossEncoder(
    CROSS_ENCODER_MODEL
)

print("Cross-Encoder loaded.")


# ============================================================
# DENSE TOP-5
# ============================================================

def get_dense_top5(query_id):

    rows = dense_df[
        dense_df[dense_query_col] == query_id
    ].copy()

    rows = rows.sort_values(
        dense_rank_col
    ).head(5)

    return rows[dense_chunk_col].tolist()


# ============================================================
# CROSS-ENCODER RERANKING
# ============================================================

def cross_encoder_top3(
    query_id,
    query_text
):

    candidate_ids = get_dense_top5(
        query_id
    )

    if not candidate_ids:

        raise ValueError(
            f"No dense candidates found for {query_id}"
        )

    pairs = []
    valid_chunk_ids = []

    for chunk_id in candidate_ids:

        if chunk_id not in corpus_lookup:

            print(
                f"WARNING: chunk {chunk_id} "
                f"not found in corpus."
            )

            continue

        chunk_text = corpus_lookup[
            chunk_id
        ]

        pairs.append(
            [
                query_text,
                chunk_text
            ]
        )

        valid_chunk_ids.append(
            chunk_id
        )

    if not pairs:

        raise ValueError(
            f"No valid candidates for {query_id}"
        )

    scores = cross_encoder.predict(
        pairs,
        show_progress_bar=False
    )

    scored = list(
        zip(
            valid_chunk_ids,
            scores
        )
    )

    scored.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [
        chunk_id
        for chunk_id, score in scored[:3]
    ]


# ============================================================
# EXISTING TOP-3
# ============================================================

def get_existing_top3(
    df,
    query_col,
    chunk_col,
    rank_col,
    query_id
):

    rows = df[
        df[query_col] == query_id
    ].copy()

    rows = rows.sort_values(
        rank_col
    ).head(3)

    return rows[chunk_col].tolist()


# ============================================================
# FORMAT EVIDENCE FOR LLM
# ============================================================

def format_evidence(chunk_ids):

    evidence_parts = []

    for rank, chunk_id in enumerate(
        chunk_ids,
        start=1
    ):

        text = corpus_lookup.get(
            chunk_id,
            ""
        )

        evidence_parts.append(
            f"[Evidence {rank} | {chunk_id}]\n"
            f"{text}"
        )

    return "\n\n".join(
        evidence_parts
    )


# ============================================================
# OLLAMA
# ============================================================

def ask_llm(
    query,
    chunk_ids
):

    evidence = format_evidence(
        chunk_ids
    )

    prompt = f"""
Answer the user's question using only the evidence provided below.

Question:
{query}

Evidence:
{evidence}

Instructions:
- Use only the supplied evidence.
- Do not invent facts.
- Do not use outside knowledge.
- If the evidence does not fully answer the question, say so.
- Give a concise and direct answer.
"""

    payload = {

        "model": OLLAMA_MODEL,

        "prompt": prompt,

        "stream": False,

        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data.get(
        "response",
        ""
    ).strip()


# ============================================================
# MAIN EXPERIMENT
# ============================================================

results = []


print("\n")
print("=" * 80)
print("FINAL LLM COMPARISON")
print("=" * 80)


for query_id, query_text in query_lookup.items():

    print("\n" + "=" * 80)

    print(
        f"Processing {query_id}"
    )

    print(
        query_text
    )

    # --------------------------------------------------------
    # Cross-Encoder
    # --------------------------------------------------------

    cross_ids = cross_encoder_top3(
        query_id,
        query_text
    )

    # --------------------------------------------------------
    # MaxSim
    # --------------------------------------------------------

    maxsim_ids = get_existing_top3(
        maxsim_df,
        maxsim_query_col,
        maxsim_chunk_col,
        maxsim_rank_col,
        query_id
    )

    # --------------------------------------------------------
    # ColBERT
    # --------------------------------------------------------

    colbert_ids = get_existing_top3(
        colbert_df,
        colbert_query_col,
        colbert_chunk_col,
        colbert_rank_col,
        query_id
    )

    print(
        f"Cross-Encoder Top3 : {cross_ids}"
    )

    print(
        f"MaxSim Top3        : {maxsim_ids}"
    )

    print(
        f"ColBERT Top3       : {colbert_ids}"
    )

    # --------------------------------------------------------
    # LLM responses
    # --------------------------------------------------------

    print(
        "Generating Cross-Encoder response..."
    )

    response_cross = ask_llm(
        query_text,
        cross_ids
    )

    print(
        "Generating MaxSim response..."
    )

    response_maxsim = ask_llm(
        query_text,
        maxsim_ids
    )

    print(
        "Generating ColBERT response..."
    )

    response_colbert = ask_llm(
        query_text,
        colbert_ids
    )

    results.append(
        {
            "query_id": query_id,
            "query": query_text,

            "cross_ids": cross_ids,
            "maxsim_ids": maxsim_ids,
            "colbert_ids": colbert_ids,

            "response_cross": response_cross,
            "response_maxsim": response_maxsim,
            "response_colbert": response_colbert
        }
    )


# ============================================================
# WRITE TEXT REPORT
# ============================================================

with open(
    FINAL_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FINAL LLM RESPONSE COMPARISON\n"
    )

    f.write(
        "Cross-Encoder vs MaxSim vs ColBERT\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    for result in results:

        f.write(
            "=" * 80 + "\n"
        )

        f.write(
            f"QUERY {result['query_id']}\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            "Question:\n"
        )

        f.write(
            result["query"]
        )

        f.write(
            "\n\n"
        )

        # ----------------------------------------------------
        # Cross Encoder
        # ----------------------------------------------------

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            "CROSS-ENCODER\n"
        )

        f.write(
            "-" * 80 + "\n\n"
        )

        f.write(
            "Top-3 Evidence:\n"
        )

        for rank, chunk_id in enumerate(
            result["cross_ids"],
            start=1
        ):

            f.write(
                f"{rank}. {chunk_id}\n"
            )

        f.write(
            "\nResponse:\n"
        )

        f.write(
            result["response_cross"]
        )

        f.write(
            "\n\n"
        )

        # ----------------------------------------------------
        # MaxSim
        # ----------------------------------------------------

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            "MAXSIM\n"
        )

        f.write(
            "-" * 80 + "\n\n"
        )

        f.write(
            "Top-3 Evidence:\n"
        )

        for rank, chunk_id in enumerate(
            result["maxsim_ids"],
            start=1
        ):

            f.write(
                f"{rank}. {chunk_id}\n"
            )

        f.write(
            "\nResponse:\n"
        )

        f.write(
            result["response_maxsim"]
        )

        f.write(
            "\n\n"
        )

        # ----------------------------------------------------
        # ColBERT
        # ----------------------------------------------------

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            "COLBERT\n"
        )

        f.write(
            "-" * 80 + "\n\n"
        )

        f.write(
            "Top-3 Evidence:\n"
        )

        for rank, chunk_id in enumerate(
            result["colbert_ids"],
            start=1
        ):

            f.write(
                f"{rank}. {chunk_id}\n"
            )

        f.write(
            "\nResponse:\n"
        )

        f.write(
            result["response_colbert"]
        )

        f.write(
            "\n\n"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)

print(
    f"\nOutput file:\n{FINAL_OUTPUT}"
)

print(
    f"\nQueries processed: {len(results)}"
)