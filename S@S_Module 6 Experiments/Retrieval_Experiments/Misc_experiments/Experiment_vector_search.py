import requests
import numpy as np

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
    "C7": "Unused entitlement does not expire immediately; remaining allowance rolls into the subsequent calendar period.",
    } 

# --------------------------------------------------
# 2. Query
# --------------------------------------------------

#query = "How many vacation days can I carry over?"
#query = "Can unused vacation entitlement be transferred to next year?"
#query = "What is the maximum annual leave rollover?"
#query = "What is the medical certificate requirement after three consecutive days?"
query = "What happens to unused vacation at the end of the year?"
# --------------------------------------------------
# 3. Embedding configuration
# --------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "mxbai-embed-large:335m"


def get_embedding(text):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": text
        }
    )

    response.raise_for_status()

    return np.array(
        response.json()["embedding"],
        dtype=np.float32
    )


# --------------------------------------------------
# 4. Create chunk embeddings
# --------------------------------------------------

print("Creating chunk embeddings...")

chunk_vectors = {}

for chunk_id, text in chunks.items():
    chunk_vectors[chunk_id] = get_embedding(text)

# --------------------------------------------------
# 5. Create query embedding
# --------------------------------------------------

print("Creating query embedding...")

query_vector = get_embedding(query)


# --------------------------------------------------
# 6. Cosine similarity
# --------------------------------------------------

def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# --------------------------------------------------
# 7. Compare query against every chunk
# --------------------------------------------------

results = []

for chunk_id, vector in chunk_vectors.items():

    score = cosine_similarity(
        query_vector,
        vector
    )

    results.append(
        (chunk_id, chunks[chunk_id], score)
    )


# --------------------------------------------------
# 8. Rank by cosine similarity
# --------------------------------------------------

results.sort(
    key=lambda x: x[2],
    reverse=True
)


# --------------------------------------------------
# 9. Display results
# --------------------------------------------------

print("\nQUERY:")
print(query)

print("\nDENSE RETRIEVAL RESULTS")
print("=" * 80)

for rank, (chunk_id, text, score) in enumerate(
    results,
    start=1
):

    print(f"\nRank {rank}")
    print(f"Chunk ID : {chunk_id}")
    print(f"Cosine  : {score:.4f}")
    print(f"Text    : {text}")

print("\n" + "=" * 80)

# --------------------------------------------------
# 10. Inspect vector dimensions
# --------------------------------------------------

first_vector = next(iter(chunk_vectors.values()))

print("\nVECTOR INFORMATION")
print("=" * 80)
print(f"Embedding model : {MODEL}")
print(f"Vector dimension: {len(first_vector)}")
print(f"Query dimension : {len(query_vector)}")