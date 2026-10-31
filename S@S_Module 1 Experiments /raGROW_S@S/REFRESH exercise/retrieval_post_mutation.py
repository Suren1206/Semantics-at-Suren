from rag_engine.retriever import Retriever

OUTPUT_FILE = "mutation_check_6_results.txt"

questions = {
    "Q108": "Are employees allowed to combine annual leave with unpaid leave to extend a vacation period?",
    "Q104": "What escalation occurs if an employee fails to maintain contact during extended medical absence?",
    "Q97": "What mid-year learning completion threshold applies to employees in regulated roles?",
    "Q146": "What happens if employees refuse to complete CPD activities required for their role?",
    "Q88": "When must whistleblowing complaints be acknowledged?",
    "Q75": "What must occur if conflict disrupts critical delivery deadlines beyond a certain duration?",
}

retriever = Retriever()

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    out.write(f"ACTIVE VERSION: {retriever.version_id}\n")
    out.write("=" * 90 + "\n\n")

    for ref, question in questions.items():

        result = retriever.retrieve(
            query=question,
            top_k=5,
            score_threshold=0.0,
            return_metadata=True
        )

        out.write(f"{ref}\n")
        out.write(f"QUESTION: {question}\n")
        out.write("-" * 90 + "\n")

        if not result.get("results"):
            out.write("NO RESULTS FOUND\n\n")
            continue

        for item in result["results"]:
            metadata = item.get("metadata", {})

            out.write(f"Rank: {item['rank']}\n")
            out.write(f"Score: {item['score']:.4f}\n")
            out.write(f"Chunk ID: {item['chunk_id']}\n")
            out.write(f"Chunk Index: {item.get('chunk_index')}\n")
            out.write(f"Header: {metadata.get('header')}\n")
            out.write(f"Text: {item['text'][:800]}\n")
            out.write("-" * 90 + "\n")

        out.write("\n\n")

print(f"Done. Output written to {OUTPUT_FILE}")