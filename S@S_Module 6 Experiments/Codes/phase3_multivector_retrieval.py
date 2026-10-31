from pathlib import Path
import json
import csv
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "Outputs"

PHASE2_NPZ = OUTPUT_DIR / "phase2_multivector_representations.npz"
PHASE2_METADATA = OUTPUT_DIR / "phase2_multivector_metadata.json"

OUTPUT_CSV = OUTPUT_DIR / "phase3_vector_level_matches.csv"
OUTPUT_TXT = OUTPUT_DIR / "phase3_multivector_retrieval.txt"


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

TOP_VECTOR_MATCHES = 5


# ============================================================
# LOAD PHASE 2 ARTIFACTS
# ============================================================

print("=" * 70)
print("PHASE 3 — SIMPLE MULTI-VECTOR RETRIEVAL")
print("=" * 70)

print("\nLoading Phase 2 artifacts...")

if not PHASE2_NPZ.exists():
    raise FileNotFoundError(
        f"Phase 2 vector file not found:\n{PHASE2_NPZ}"
    )

if not PHASE2_METADATA.exists():
    raise FileNotFoundError(
        f"Phase 2 metadata file not found:\n{PHASE2_METADATA}"
    )

data = np.load(PHASE2_NPZ)

vectors = data["vectors"]
vector_ids = data["vector_ids"]

with open(PHASE2_METADATA, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Loaded vectors      : {vectors.shape}")
print(f"Loaded vector IDs   : {len(vector_ids)}")


# ============================================================
# NORMALIZE VECTORS
# ============================================================
#
# Phase 3 uses cosine similarity.
#
# After normalization:
#
#       cosine(q, v) = q · v
#
# No aggregation is performed here.
# ============================================================

print("\nNormalizing vectors for cosine similarity...")

norms = np.linalg.norm(vectors, axis=1, keepdims=True)

if np.any(norms == 0):
    raise ValueError("Zero-length vector detected.")

normalized_vectors = vectors / norms


# ============================================================
# BUILD VECTOR-ID → MATRIX-INDEX LOOKUP
# ============================================================

print("Building vector index lookup...")

vector_id_to_index = {
    str(vector_id): index
    for index, vector_id in enumerate(vector_ids)
}


# ============================================================
# BUILD CORPUS VECTOR RECORDS
# ============================================================
#
# Each record describes ONE corpus sub-vector:
#
#   vector_id
#   parent chunk
#   token
#
# The parent chunk is retained only as provenance.
# ============================================================

corpus_vector_records = []

for chunk_id, chunk_info in metadata["corpus"].items():

    tokens = chunk_info["tokens"]
    chunk_vector_ids = chunk_info["vector_ids"]

    if len(tokens) != len(chunk_vector_ids):
        raise ValueError(
            f"Token/vector count mismatch for {chunk_id}: "
            f"{len(tokens)} tokens vs "
            f"{len(chunk_vector_ids)} vectors."
        )

    for token, vector_id in zip(
        tokens,
        chunk_vector_ids
    ):

        if vector_id not in vector_id_to_index:
            raise KeyError(
                f"Vector ID {vector_id} from {chunk_id} "
                f"not found in NPZ."
            )

        corpus_vector_records.append(
            {
                "vector_id": vector_id,
                "chunk_id": chunk_id,
                "token": token,
            }
        )


# ============================================================
# BUILD QUERY VECTOR RECORDS
# ============================================================

query_vector_records = []

for query_id, query_info in metadata["queries"].items():

    query_text = query_info["query"]

    tokens = query_info["tokens"]
    query_vector_ids = query_info["vector_ids"]

    if len(tokens) != len(query_vector_ids):
        raise ValueError(
            f"Token/vector count mismatch for {query_id}: "
            f"{len(tokens)} tokens vs "
            f"{len(query_vector_ids)} vectors."
        )

    for token, vector_id in zip(
        tokens,
        query_vector_ids
    ):

        if vector_id not in vector_id_to_index:
            raise KeyError(
                f"Vector ID {vector_id} from {query_id} "
                f"not found in NPZ."
            )

        query_vector_records.append(
            {
                "query_id": query_id,
                "query": query_text,
                "vector_id": vector_id,
                "token": token,
            }
        )


print(
    f"Corpus vectors      : {len(corpus_vector_records)}"
)

print(
    f"Query vectors       : {len(query_vector_records)}"
)


# ============================================================
# BUILD CORPUS MATRIX
# ============================================================

corpus_indices = [
    vector_id_to_index[item["vector_id"]]
    for item in corpus_vector_records
]

corpus_matrix = normalized_vectors[corpus_indices]


# ============================================================
# VECTOR-LEVEL RETRIEVAL
# ============================================================
#
# For EVERY query sub-vector:
#
#       q_i
#        |
#        +---- cosine ----> corpus vector 1
#        |
#        +---- cosine ----> corpus vector 2
#        |
#        +---- cosine ----> corpus vector 3
#        |
#        +---- cosine ----> ...
#
# We retain the strongest individual corpus vectors.
#
# IMPORTANT:
#
# NO:
#   - MaxSim
#   - chunk aggregation
#   - query aggregation
#   - chunk-level score
#   - reranking
#
# Parent chunk is only provenance.
# ============================================================

print("\nPerforming exhaustive vector-level retrieval...")

csv_rows = []
text_sections = []

current_query_id = None

for query_record in query_vector_records:

    query_id = query_record["query_id"]
    query_text = query_record["query"]
    query_vector_id = query_record["vector_id"]
    query_token = query_record["token"]

    # --------------------------------------------------------
    # Add query heading when we encounter a new query.
    # --------------------------------------------------------

    if query_id != current_query_id:

        current_query_id = query_id

        text_sections.append("")
        text_sections.append("=" * 70)
        text_sections.append(
            f"{query_id}: {query_text}"
        )
        text_sections.append("=" * 70)

    # --------------------------------------------------------
    # Retrieve the ONE query vector.
    # --------------------------------------------------------

    query_index = vector_id_to_index[
        query_vector_id
    ]

    query_vector = normalized_vectors[
        query_index
    ]

    # --------------------------------------------------------
    # Compare this query vector against EVERY
    # corpus sub-vector.
    # --------------------------------------------------------

    similarities = corpus_matrix @ query_vector

    # --------------------------------------------------------
    # Identify the strongest individual corpus vectors.
    # --------------------------------------------------------

    top_indices = np.argsort(
        similarities
    )[::-1][:TOP_VECTOR_MATCHES]

    text_sections.append(
        f"\nQuery vector: {query_vector_id}"
    )

    text_sections.append(
        f"Query token : {query_token}"
    )

    for rank, corpus_position in enumerate(
        top_indices,
        start=1
    ):

        corpus_record = corpus_vector_records[
            corpus_position
        ]

        score = float(
            similarities[corpus_position]
        )

        csv_rows.append(
            {
                "query_id": query_id,
                "query": query_text,
                "query_vector_id": query_vector_id,
                "query_token": query_token,
                "rank": rank,
                "corpus_vector_id": corpus_record[
                    "vector_id"
                ],
                "corpus_token": corpus_record[
                    "token"
                ],
                "parent_chunk_id": corpus_record[
                    "chunk_id"
                ],
                "cosine_similarity": score,
            }
        )

        text_sections.append(
            f"  {rank}. "
            f"{corpus_record['vector_id']} "
            f"| token={corpus_record['token']} "
            f"| parent={corpus_record['chunk_id']} "
            f"| cosine={score:.6f}"
        )


# ============================================================
# WRITE CSV
# ============================================================

print("\nWriting CSV...")

csv_fields = [
    "query_id",
    "query",
    "query_vector_id",
    "query_token",
    "rank",
    "corpus_vector_id",
    "corpus_token",
    "parent_chunk_id",
    "cosine_similarity",
]

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=csv_fields
    )

    writer.writeheader()
    writer.writerows(csv_rows)


# ============================================================
# WRITE HUMAN-READABLE REPORT
# ============================================================

print("Writing text report...")

report_header = [
    "PHASE 3 — SIMPLE MULTI-VECTOR RETRIEVAL",
    "",
    "Purpose:",
    (
        "Identify the closest corpus sub-vectors "
        "for each query sub-vector."
    ),
    "",
    "Representation:",
    "Query  -> multiple contextual vectors",
    "Chunk  -> multiple contextual vectors",
    "",
    "Retrieval:",
    (
        "Each query vector is compared against "
        "every corpus vector using cosine similarity."
    ),
    "",
    "Important:",
    "No MaxSim is used.",
    "No chunk-level aggregation is used.",
    "No query-level aggregation is used.",
    (
        "Parent chunk IDs are retained only "
        "as provenance."
    ),
    "",
    (
        "Top individual vector matches retained "
        f"per query vector: {TOP_VECTOR_MATCHES}"
    ),
    "",
]

with open(
    OUTPUT_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write("\n".join(report_header))
    f.write("\n")
    f.write("\n".join(text_sections))


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3 COMPLETE")
print("=" * 70)

print(
    f"\nTotal query vectors  : "
    f"{len(query_vector_records)}"
)

print(
    f"Total corpus vectors : "
    f"{len(corpus_vector_records)}"
)

print(
    f"Matches recorded     : "
    f"{len(csv_rows)}"
)

print("\nOutputs:")
print(f"  {OUTPUT_CSV}")
print(f"  {OUTPUT_TXT}")

print(
    "\nPhase 3 has identified nearest corpus vectors "
    "for each individual query vector."
)

print(
    "No chunk-level scores or MaxSim scores "
    "have been calculated."
)