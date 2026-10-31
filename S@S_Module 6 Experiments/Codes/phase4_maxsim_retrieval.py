from pathlib import Path
import json
import csv

import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_NPZ = PROJECT_ROOT / "Outputs" / "phase2_multivector_representations.npz"
INPUT_METADATA = PROJECT_ROOT / "Outputs" / "phase2_multivector_metadata.json"

OUTPUT_CSV = PROJECT_ROOT / "Outputs" / "phase4_maxsim_scores.csv"
OUTPUT_TXT = PROJECT_ROOT / "Outputs" / "phase4_maxsim_report.txt"


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

TOP_K = 3


# ============================================================
# LOAD PHASE 2 ARTIFACTS
# ============================================================

print("Loading Phase 2 representations...")

data = np.load(INPUT_NPZ)

vectors = data["vectors"].astype(np.float32)

with open(INPUT_METADATA, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Loaded vectors: {vectors.shape}")


# ============================================================
# NORMALIZE VECTORS
# ============================================================
#
# Phase 4 uses cosine similarity.
#
# cosine(q, c) = normalized_q dot normalized_c
#
# We normalize once here and then use dot products.
# ============================================================

norms = np.linalg.norm(vectors, axis=1, keepdims=True)

norms[norms == 0] = 1.0

vectors_normalized = vectors / norms


# ============================================================
# BUILD VECTOR LOOKUP
# ============================================================

vector_id_to_index = {}

for index, vector_id in enumerate(data["vector_ids"]):
    vector_id_to_index[str(vector_id)] = index


# ============================================================
# BUILD CORPUS REPRESENTATION
# ============================================================

corpus = {}

for chunk_id, item in metadata["corpus"].items():

    vector_ids = item["vector_ids"]
    tokens = item["tokens"]

    corpus_vectors = []

    for vector_id, token in zip(vector_ids, tokens):

        vector_index = vector_id_to_index[str(vector_id)]

        corpus_vectors.append(
            {
                "vector_id": str(vector_id),
                "token": token,
                "vector_index": vector_index,
            }
        )

    corpus[chunk_id] = {
        "text": item["text"],
        "vectors": corpus_vectors,
    }


# ============================================================
# BUILD QUERY REPRESENTATION
# ============================================================

queries = {}

for query_id, item in metadata["queries"].items():

    vector_ids = item["vector_ids"]
    tokens = item["tokens"]

    query_vectors = []

    for vector_id, token in zip(vector_ids, tokens):

        vector_index = vector_id_to_index[str(vector_id)]

        query_vectors.append(
            {
                "vector_id": str(vector_id),
                "token": token,
                "vector_index": vector_index,
            }
        )

    queries[query_id] = {
        "query": item["query"],
        "vectors": query_vectors,
    }


# ============================================================
# PHASE 4 — MAXSIM
# ============================================================
#
# For every query Q and corpus chunk C:
#
#       MaxSim(Q, C)
#
#       = SUM_i max_j cosine(q_i, c_j)
#
# where:
#
#       q_i = one query vector
#       c_j = one vector belonging to the corpus chunk
#
# Each query vector independently finds its strongest
# matching vector inside the chunk.
#
# Those strongest matches are then summed.
#
# This is the explicit fine-grained interaction mechanism.
# ============================================================


all_results = []

report_lines = []

report_lines.append("=" * 80)
report_lines.append("PHASE 4 — EXPLICIT MAXSIM RETRIEVAL")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("Representation:")
report_lines.append("  Chunk -> multiple contextual vectors")
report_lines.append("  Query -> multiple contextual vectors")
report_lines.append("")
report_lines.append("Interaction:")
report_lines.append("  For each query vector:")
report_lines.append("      find strongest matching corpus vector inside each chunk")
report_lines.append("")
report_lines.append("Aggregation:")
report_lines.append("  MaxSim(Q,C) = SUM over query vectors of their maximum")
report_lines.append("  similarity against vectors belonging to chunk C")
report_lines.append("")
report_lines.append(f"Top-K reported: {TOP_K}")
report_lines.append("")
report_lines.append(
    "IMPORTANT: These are raw BGE-M3 contextual hidden states, "
    "not genuine pretrained ColBERT token embeddings."
)
report_lines.append(
    "This phase isolates the MaxSim mechanism in the simplified "
    "multi-vector reconstruction."
)
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("")


for query_id, query_data in queries.items():

    query_text = query_data["query"]
    query_vectors = query_data["vectors"]

    chunk_results = []

    for chunk_id, chunk_data in corpus.items():

        chunk_vectors = chunk_data["vectors"]

        # ----------------------------------------------------
        # Matrix containing all corpus vectors for this chunk
        # ----------------------------------------------------

        chunk_indices = [
            item["vector_index"]
            for item in chunk_vectors
        ]

        chunk_matrix = vectors_normalized[chunk_indices]

        # ----------------------------------------------------
        # Per-query-vector MaxSim contributions
        # ----------------------------------------------------

        contributions = []

        for query_item in query_vectors:

            query_index = query_item["vector_index"]

            query_vector = vectors_normalized[query_index]

            # ------------------------------------------------
            # Similarity of this query vector against every
            # vector belonging to this corpus chunk.
            # ------------------------------------------------

            similarities = chunk_matrix @ query_vector

            best_position = int(np.argmax(similarities))

            best_score = float(similarities[best_position])

            best_corpus_vector = chunk_vectors[best_position]

            contributions.append(
                {
                    "query_vector_id": query_item["vector_id"],
                    "query_token": query_item["token"],
                    "best_vector_id": best_corpus_vector["vector_id"],
                    "best_token": best_corpus_vector["token"],
                    "score": best_score,
                }
            )

        # ----------------------------------------------------
        # Standard MaxSim aggregation:
        #
        #       SUM(max similarity for each query vector)
        # ----------------------------------------------------

        maxsim_score = sum(
            item["score"]
            for item in contributions
        )

        # ----------------------------------------------------
        # Mean is retained only as a diagnostic.
        # Primary ranking remains SUM MaxSim.
        # ----------------------------------------------------

        maxsim_mean = (
            maxsim_score / len(contributions)
            if contributions
            else 0.0
        )

        chunk_results.append(
            {
                "query_id": query_id,
                "query": query_text,
                "chunk_id": chunk_id,
                "chunk_text": chunk_data["text"],
                "query_vector_count": len(query_vectors),
                "maxsim_score": maxsim_score,
                "maxsim_mean": maxsim_mean,
                "contributions": contributions,
            }
        )

    # --------------------------------------------------------
    # Rank chunks by MaxSim score
    # --------------------------------------------------------

    chunk_results.sort(
        key=lambda x: x["maxsim_score"],
        reverse=True
    )

    for rank, result in enumerate(chunk_results, start=1):
        result["rank"] = rank
        all_results.append(result)

    # --------------------------------------------------------
    # REPORT TOP-K
    # --------------------------------------------------------

    report_lines.append("-" * 80)
    report_lines.append(
        f"{query_id}: {query_text}"
    )
    report_lines.append("-" * 80)
    report_lines.append("")

    for result in chunk_results[:TOP_K]:

        report_lines.append(
            f"Rank {result['rank']} | "
            f"{result['chunk_id']} | "
            f"MaxSim = {result['maxsim_score']:.6f} | "
            f"Mean = {result['maxsim_mean']:.6f}"
        )

        report_lines.append(
            f"Chunk: {result['chunk_text']}"
        )

        report_lines.append("")
        report_lines.append(
            "Query-vector -> strongest corpus-vector matches:"
        )

        for contribution in result["contributions"]:

            report_lines.append(
                f"  {contribution['query_token']!r}"
                f" -> "
                f"{contribution['best_token']!r}"
                f" | "
                f"{contribution['score']:.6f}"
            )

        report_lines.append("")
        report_lines.append(
            f"MaxSim SUM = {result['maxsim_score']:.6f}"
        )
        report_lines.append("")


# ============================================================
# WRITE CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "query_id",
            "query",
            "rank",
            "chunk_id",
            "maxsim_score",
            "maxsim_mean",
            "query_vector_count",
        ]
    )

    for result in all_results:

        writer.writerow(
            [
                result["query_id"],
                result["query"],
                result["rank"],
                result["chunk_id"],
                f"{result['maxsim_score']:.8f}",
                f"{result['maxsim_mean']:.8f}",
                result["query_vector_count"],
            ]
        )


# ============================================================
# WRITE TEXT REPORT
# ============================================================

with open(
    OUTPUT_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write("\n".join(report_lines))


# ============================================================
# FINAL CONSOLE OUTPUT
# ============================================================

print("")
print("=" * 80)
print("PHASE 4 COMPLETE")
print("=" * 80)
print("")
print(f"CSV report : {OUTPUT_CSV}")
print(f"TXT report : {OUTPUT_TXT}")
print("")
print("MaxSim ranking generated for all queries.")
print(f"Top-{TOP_K} interaction traces written to the TXT report.")
print("")