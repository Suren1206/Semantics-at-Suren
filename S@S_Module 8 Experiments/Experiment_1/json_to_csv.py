from pathlib import Path
import json
import csv


BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "results" / "top5_results.json"
OUTPUT_FILE = BASE_DIR / "results" / "top5_results.csv"


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


rows = []

for query_id, query_data in data.items():
    query_text = query_data["query"]

    for result in query_data["results"]:
        rows.append({
            "query_id": query_id,
            "query": query_text,
            "rank": result["rank"],
            "similarity": result["similarity"],
            "corpus_id": result["corpus_id"],
            "title": result["title"],
            "text": result["text"],
        })


with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "query_id",
            "query",
            "rank",
            "similarity",
            "corpus_id",
            "title",
            "text",
        ],
    )

    writer.writeheader()
    writer.writerows(rows)


print(f"Created: {OUTPUT_FILE}")
print(f"Rows   : {len(rows)}")