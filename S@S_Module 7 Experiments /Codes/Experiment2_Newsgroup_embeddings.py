import json
import requests
import numpy as np
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "Data" / "experiment2_newsgroups.json"
embedding_path = BASE_DIR / "Data" / "experiment2_newsgroup_embeddings.npy"
metadata_path = BASE_DIR / "Data" / "experiment2_newsgroup_metadata.json"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL_NAME = "mxbai-embed-large:335m"
BATCH_SIZE = 32


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

with open(input_path, "r", encoding="utf-8") as f:
    records = json.load(f)

print(f"Loaded documents: {len(records)}")


# ---------------------------------------------------------
# Generate embeddings
# ---------------------------------------------------------

all_embeddings = []

for start in range(0, len(records), BATCH_SIZE):

    batch = records[start:start + BATCH_SIZE]
    texts = [record["text"] for record in batch]

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "input": texts,
        },
        timeout=300,
    )

    response.raise_for_status()

    result = response.json()
    batch_embeddings = result["embeddings"]

    all_embeddings.extend(batch_embeddings)

    end = min(start + BATCH_SIZE, len(records))

    print(f"Embedded {end}/{len(records)} documents")


# ---------------------------------------------------------
# Convert to NumPy array
# ---------------------------------------------------------

embeddings = np.asarray(all_embeddings, dtype=np.float32)

print()
print("Embedding generation complete.")
print(f"Embedding shape: {embeddings.shape}")
print(f"Embedding dimension: {embeddings.shape[1]}")


# ---------------------------------------------------------
# Verify alignment
# ---------------------------------------------------------

if len(embeddings) != len(records):
    raise ValueError(
        "Number of embeddings does not match number of documents."
    )

if embeddings.shape[1] != 1024:
    raise ValueError(
        f"Unexpected embedding dimension: {embeddings.shape[1]}"
    )


# ---------------------------------------------------------
# Save embeddings
# ---------------------------------------------------------

np.save(embedding_path, embeddings)


# ---------------------------------------------------------
# Save metadata separately
# ---------------------------------------------------------

metadata = []

for record in records:
    metadata.append(
        {
            "document_id": record["document_id"],
            "category_id": record["category_id"],
            "category": record["category"],
        }
    )

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------
# Final confirmation
# ---------------------------------------------------------

print()
print("Embeddings saved successfully.")
print(f"Embeddings: {embedding_path}")
print(f"Metadata:   {metadata_path}")
print(f"Final shape: {embeddings.shape}")