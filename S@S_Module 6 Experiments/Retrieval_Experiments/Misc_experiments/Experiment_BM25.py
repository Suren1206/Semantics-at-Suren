from rank_bm25 import BM25Okapi

# --------------------------------------------------
# 1. Controlled corpus
# --------------------------------------------------

chunks = {
    "C1": "Employees may carry forward up to 10 days of unused annual leave.",
    "C2": "Staff must obtain manager approval before taking annual leave.",
    "C3": "Unused vacation entitlement can be transferred to the following year, subject to a maximum of ten days.",
    "C4": "Sick leave requires a medical certificate after three consecutive days of absence.",
    "C5": "Employees receive 20 days of annual leave each calendar year.",
    "C6": "Managers are responsible for approving employee time-off requests.",
}

# --------------------------------------------------
# 2. Query
# --------------------------------------------------

query = "How many vacation days can I carry over?"

# --------------------------------------------------
# 3. Tokenize
# --------------------------------------------------

tokenized_chunks = [
    text.lower().split()
    for text in chunks.values()
]

tokenized_query = query.lower().split()

# --------------------------------------------------
# 4. Build BM25 index
# --------------------------------------------------

bm25 = BM25Okapi(tokenized_chunks)

# --------------------------------------------------
# Inspect query-term statistics
# --------------------------------------------------

print("\nQUERY TERM ANALYSIS")
print("=" * 80)

for term in tokenized_query:
    df = sum(term in doc for doc in tokenized_chunks)
    idf = bm25.idf.get(term, 0)

    matching_chunks = [
        chunk_id
        for chunk_id, doc in zip(chunks.keys(), tokenized_chunks)
        if term in doc
    ]

    print(f"\nTerm: {term}")
    print(f"Document frequency (df): {df}")
    print(f"BM25 IDF: {idf:.4f}")
    print(f"Matching chunks: {matching_chunks}")


# --------------------------------------------------
# 5. Calculate BM25 scores
# --------------------------------------------------

scores = bm25.get_scores(tokenized_query)

# --------------------------------------------------
# 6. Rank chunks
# --------------------------------------------------

results = sorted(
    zip(chunks.keys(), chunks.values(), scores),
    key=lambda x: x[2],
    reverse=True
)

# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

print("\nQUERY:")
print(query)

print("\nBM25 RESULTS")
print("=" * 80)

for rank, (chunk_id, text, score) in enumerate(results, start=1):

    print(f"\nRank {rank}")
    print(f"Chunk ID : {chunk_id}")
    print(f"Score    : {score:.4f}")
    print(f"Text     : {text}")

print("\n" + "=" * 80)