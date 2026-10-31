from rag_engine.retriever import Retriever
from pathlib import Path

QUERY_FILE = "query_set_150.txt"
OUTPUT_FILE = "baseline_150_results.txt"

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

        result = retriever.retrieve(query=query, top_k=1)

        if not result["results"]:
            score = 0
            band = "LOW"
            chunk_text = "NO RETRIEVAL"
            low += 1

        else:
            top_result = result["results"][0]
            score = top_result["score"]
            chunk_text = top_result["text"]

            total_score += score

            if score >= 0.75:
                band = "HIGH"
                high += 1
            elif score >= 0.60:
                band = "MID"
                mid += 1
            else:
                band = "LOW"
                low += 1

        out.write(f"Q{i}\n")
        out.write(f"Query: {query}\n")
        out.write(f"Score: {score:.4f}\n")
        out.write(f"Band: {band}\n")
        out.write(f"Top Chunk: {chunk_text[:500]}\n")
        out.write("-" * 80 + "\n")

    avg_score = total_score / len(queries)

    out.write("\n")
    out.write("=" * 80 + "\n")
    out.write("SUMMARY\n")
    out.write("=" * 80 + "\n")
    out.write(f"HIGH: {high}\n")
    out.write(f"MID: {mid}\n")
    out.write(f"LOW: {low}\n")
    out.write(f"AVG SCORE: {avg_score:.4f}\n")