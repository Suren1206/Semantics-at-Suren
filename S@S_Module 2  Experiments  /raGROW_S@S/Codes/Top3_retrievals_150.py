# ============================================================
# FIX IMPORT PATH FOR rag_engine
# ============================================================

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# ============================================================
# IMPORTS
# ============================================================

from rag_engine.retriever import Retriever

QUERY_FILE = PROJECT_ROOT / "REFRESH exercise" / "query_set_150.txt"
OUTPUT_FILE = PROJECT_ROOT / "REFRESH exercise" / "baseline_150_results.txt"

retriever = Retriever()

# --------------------------
# Load queries
# --------------------------
with open(QUERY_FILE, "r", encoding="utf-8") as f:
    queries = [line.strip() for line in f if line.strip()]

high = 0
mid = 0
low = 0
total_score = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    out.write(f"ACTIVE VERSION: {retriever.version_id}\n")
    out.write(f"TOTAL QUERIES: {len(queries)}\n")
    out.write("=" * 80 + "\n\n")

    for i, query in enumerate(queries, start=1):

        result = retriever.retrieve(query=query, top_k=3)

        out.write(f"Q{i}\n")
        out.write(f"Query: {query}\n\n")

        # -------------------------------------------------
        # No retrieval case
        # -------------------------------------------------
        if not result["results"]:

            score = 0
            band = "LOW"

            low += 1

            out.write("NO RETRIEVAL\n")
            out.write("-" * 80 + "\n")

            continue

        # -------------------------------------------------
        # Use top retrieval for band calculation
        # -------------------------------------------------
        top_score = result["results"][0]["score"]

        total_score += top_score

        if top_score >= 0.75:
            band = "HIGH"
            high += 1

        elif top_score >= 0.60:
            band = "MID"
            mid += 1

        else:
            band = "LOW"
            low += 1

        out.write(f"Top Score: {top_score:.4f}\n")
        out.write(f"Band: {band}\n\n")

        # -------------------------------------------------
        # Print Top 3 Retrievals
        # -------------------------------------------------
        for rank, item in enumerate(result["results"], start=1):

            score = item["score"]
            chunk_text = item["text"]

            out.write(f"RETRIEVAL #{rank}\n")
            out.write(f"Cosine Similarity Score: {score:.4f}\n")
            out.write(f"Chunk:\n{chunk_text[:700]}\n")
            out.write("-" * 60 + "\n")

        out.write("=" * 80 + "\n\n")

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------
    avg_score = total_score / len(queries)

    out.write("\n")
    out.write("=" * 80 + "\n")
    out.write("SUMMARY\n")
    out.write("=" * 80 + "\n")
    out.write(f"HIGH: {high}\n")
    out.write(f"MID: {mid}\n")
    out.write(f"LOW: {low}\n")
    out.write(f"AVERAGE SCORE: {avg_score:.4f}\n")

print(f"\nResults saved to:\n{OUTPUT_FILE}")