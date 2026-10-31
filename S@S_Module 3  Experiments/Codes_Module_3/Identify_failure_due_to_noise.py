from pathlib import Path
import re
import numpy as np
import requests


# =============================================================================
# Semantics @ Suren — Module 4
# Atomic Chunk Retrieval Experiment
# =============================================================================
#
# Purpose:
#   Split the supplied HR text into sentence-level atomic chunks and compare
#   a query against ALL chunks using cosine similarity.
#
# No PCA
# No mean-centering
# No whitening
# No dimensionality reduction
#
# The original text is embedded directly in this script.
# =============================================================================


# =============================================================================
# BLOCK 1 — CONFIGURATION
# =============================================================================

EMBEDDING_MODEL = "mxbai-embed-large:335m"

OLLAMA_URL = "http://localhost:11434/api/embeddings"

QUERY = "What if my manager only tried to contact me once?"

# The temporary NPY file will be created beside this Python script.
BASE_DIR = Path(__file__).resolve().parent

OUTPUT_FILE = BASE_DIR / "atomic_hr_chunks_embeddings.npy"


# =============================================================================
# BLOCK 2 — SOURCE TEXT
# =============================================================================

SOURCE_TEXT = """
Support
from HR must be provided within 5 working days of formal request.
Managers must address absence threshold triggers in accordance with
policy timelines. Line managers must not investigate complaints in
which they are personally implicated. Manager performance metrics may
include team engagement outcomes.

Employees are expected to maintain reasonable contact during
extended absence, typically not less than once every 14 calendar days
unless otherwise agreed. Adjusted or phased return arrangements may
be granted for a period not normally exceeding 20 working days,
unless extended by management discretion.

Escalated employee concerns must be acknowledged within 3 working
days. Managers must document key performance conversations within 5
working days of occurrence. Repeated managerial non-compliance
exceeding three documented instances in 12 months may trigger
capability review. Managers must ensure team training completion
rates meet organisational standards.

Three separate absence instances within a rolling 90-day period
may trigger a formal attendance review meeting. Where total absence
exceeds 8 working days within a rolling 6-month window, management
may initiate an attendance improvement plan. Absence of 10
consecutive working days without contact may be treated as voluntary
abandonment of employment, subject to documented outreach attempts on
at least two separate days.

Managers must review flexible working arrangements every 6 months. Managers must identify high-potential talent annually. Failure to fulfil managerial duties may result in reassignment of supervisory responsibility.
"""


# =============================================================================
# BLOCK 3 — ATOMIC CHUNKING
# =============================================================================

def create_atomic_chunks(text):
    """
    Split the source text at full stops.

    Each resulting sentence is treated as one atomic chunk.
    """

    # Convert line breaks and repeated whitespace into single spaces.
    cleaned_text = re.sub(r"\s+", " ", text).strip()

    # Split ONLY on full stop.
    raw_chunks = cleaned_text.split(".")

    # Remove empty fragments and surrounding whitespace.
    chunks = [
        chunk.strip()
        for chunk in raw_chunks
        if chunk.strip()
    ]

    return chunks


# =============================================================================
# BLOCK 4 — OLLAMA EMBEDDING FUNCTION
# =============================================================================

def get_embedding(text):

    payload = {
        "model": EMBEDDING_MODEL,
        "prompt": text
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return np.asarray(data["embedding"], dtype=np.float32)


# =============================================================================
# BLOCK 5 — COSINE SIMILARITY
# =============================================================================

def cosine_similarity(a, b):

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# =============================================================================
# BLOCK 6 — CREATE ATOMIC CHUNKS
# =============================================================================

print("=" * 100)
print("Semantics @ Suren — Module 4")
print("ATOMIC CHUNK QUERY SIMILARITY EXPERIMENT")
print("=" * 100)

print("\nQuery:")
print(QUERY)

print("\nCreating atomic chunks...")

chunks = create_atomic_chunks(SOURCE_TEXT)

print(f"Atomic chunks created : {len(chunks)}")


# =============================================================================
# BLOCK 7 — GENERATE EMBEDDINGS
# =============================================================================

print("\nGenerating embeddings...")
print("-" * 100)

embeddings = []

for i, chunk in enumerate(chunks, start=1):

    print(f"Embedding chunk {i:02d}/{len(chunks)}...")

    vector = get_embedding(chunk)

    embeddings.append(vector)


embeddings = np.vstack(embeddings)


# =============================================================================
# BLOCK 8 — SAVE TEMPORARY NPY
# =============================================================================

np.save(OUTPUT_FILE, embeddings)

print("\nEmbedding matrix created:")
print(f"Shape : {embeddings.shape}")

print(f"\nTemporary NPY file:")
print(OUTPUT_FILE)


# =============================================================================
# BLOCK 9 — EMBED QUERY
# =============================================================================

print("\nGenerating query embedding...")

query_vector = get_embedding(QUERY)


# =============================================================================
# BLOCK 10 — CALCULATE QUERY vs ALL CHUNKS
# =============================================================================

results = []

for i, chunk_vector in enumerate(embeddings):

    score = cosine_similarity(
        query_vector,
        chunk_vector
    )

    results.append(
        {
            "chunk_number": i + 1,
            "cosine": score,
            "text": chunks[i]
        }
    )


# =============================================================================
# BLOCK 11 — SORT HIGHEST → LOWEST
# =============================================================================

results.sort(
    key=lambda x: x["cosine"],
    reverse=True
)


# =============================================================================
# BLOCK 12 — DISPLAY RESULTS
# =============================================================================

print("\n")
print("=" * 100)
print("QUERY vs ALL ATOMIC CHUNKS")
print("=" * 100)

print(f"\nQuery:")
print(QUERY)

print("\n")
print(f"{'Rank':<8}{'Chunk':<10}{'Cosine Similarity':<22}Chunk")
print("-" * 100)

for rank, result in enumerate(results, start=1):

    print(
        f"{rank:<8}"
        f"{result['chunk_number']:<10}"
        f"{result['cosine']:<22.6f}"
        f"{result['text']}"
    )


# =============================================================================
# BLOCK 13 — SUMMARY
# =============================================================================

scores = np.array(
    [result["cosine"] for result in results],
    dtype=np.float64
)

print("\n")
print("=" * 100)
print("SIMILARITY SUMMARY")
print("=" * 100)

print(f"Number of chunks : {len(chunks)}")
print(f"Mean             : {scores.mean():.6f}")
print(f"Median           : {np.median(scores):.6f}")
print(f"Std Deviation    : {scores.std():.6f}")
print(f"Minimum          : {scores.min():.6f}")
print(f"Maximum          : {scores.max():.6f}")


# =============================================================================
# BLOCK 14 — EXPERIMENT COMPLETE
# =============================================================================

print("\n")
print("=" * 100)
print("EXPERIMENT COMPLETE")
print("=" * 100)

print("\nNo mean-centering was applied.")
print("No PCA was applied.")
print("No whitening was applied.")
print("No dimensionality reduction was applied.")
print("The original source text was not modified.")