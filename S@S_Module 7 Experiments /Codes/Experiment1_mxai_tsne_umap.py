from pathlib import Path
import json

import requests
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

import umap


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_DIR = PROJECT_ROOT / "Outputs"

CORPUS_FILE = DATA_DIR / "corpus.json"
OUTPUT_FILE = OUTPUT_DIR / "mxbaI_tsne_umap.png"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "mxbai-embed-large:335m"

TOP_K = 5

QUERY = "HHow many annual leave days can I carry forward, and when do I need to submit a request to take leave?"


# ============================================================
# LOAD CORPUS
# ============================================================

with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    corpus = json.load(f)

chunks = corpus["chunks"]

print(f"Corpus chunks: {len(chunks)}")
print(f"Query: {QUERY}")


# ============================================================
# MXBAI EMBEDDING FUNCTION
# ============================================================

def embed_text(text):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": text
        },
        timeout=120
    )

    response.raise_for_status()

    embedding = np.asarray(
        response.json()["embedding"],
        dtype=np.float32
    )

    return embedding


# ============================================================
# EMBED 40 CHUNKS
# ============================================================

chunk_ids = []
chunk_texts = []
chunk_embeddings = []

for chunk in chunks:

    chunk_id = chunk["chunk_id"]
    text = chunk["text"]

    print(f"Embedding {chunk_id}...")

    vector = embed_text(text)

    chunk_ids.append(chunk_id)
    chunk_texts.append(text)
    chunk_embeddings.append(vector)


chunk_embeddings = np.vstack(chunk_embeddings)


# ============================================================
# EMBED QUERY
# ============================================================

print("Embedding query...")

query_embedding = embed_text(QUERY).reshape(1, -1)


# ============================================================
# COSINE SIMILARITY
# ============================================================

scores = cosine_similarity(
    query_embedding,
    chunk_embeddings
)[0]

ranking = np.argsort(scores)[::-1]

print("\n============================================================")
print("COSINE TOP-5")
print("============================================================")

for rank, index in enumerate(ranking[:TOP_K], start=1):

    print(
        f"{rank}. "
        f"{chunk_ids[index]}  "
        f"cosine={scores[index]:.6f}"
    )


top5_indices = ranking[:TOP_K]


# ============================================================
# COMBINE CHUNKS + QUERY
# ============================================================

all_embeddings = np.vstack(
    [
        chunk_embeddings,
        query_embedding
    ]
)

labels = chunk_ids + ["QUERY"]


# ============================================================
# t-SNE
# ============================================================

print("\nRunning t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=10,
    random_state=42,
    init="pca",
    learning_rate="auto"
)

tsne_coordinates = tsne.fit_transform(
    all_embeddings
)


# ============================================================
# UMAP
# ============================================================

print("Running UMAP...")

umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=10,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)

umap_coordinates = umap_model.fit_transform(
    all_embeddings
)


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(20, 9)
)


# ============================================================
# t-SNE PLOT
# ============================================================

ax = axes[0]

chunk_coordinates = tsne_coordinates[:-1]
query_coordinate = tsne_coordinates[-1]

ax.scatter(
    chunk_coordinates[:, 0],
    chunk_coordinates[:, 1],
    s=55
)

ax.scatter(
    query_coordinate[0],
    query_coordinate[1],
    s=180,
    marker="*",
    edgecolors="black",
    linewidths=1.2,
    label="Query"
)

for i, chunk_id in enumerate(chunk_ids):

    ax.annotate(
        chunk_id,
        (
            chunk_coordinates[i, 0],
            chunk_coordinates[i, 1]
        ),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8
    )

ax.annotate(
    "QUERY",
    (
        query_coordinate[0],
        query_coordinate[1]
    ),
    xytext=(8, 8),
    textcoords="offset points",
    fontsize=10,
    fontweight="bold"
)

ax.set_title("t-SNE — MXBAI Single-Vector Geometry")
ax.set_xlabel("t-SNE dimension 1")
ax.set_ylabel("t-SNE dimension 2")
ax.legend()


# ============================================================
# UMAP PLOT
# ============================================================

ax = axes[1]

chunk_coordinates = umap_coordinates[:-1]
query_coordinate = umap_coordinates[-1]

ax.scatter(
    chunk_coordinates[:, 0],
    chunk_coordinates[:, 1],
    s=55
)

ax.scatter(
    query_coordinate[0],
    query_coordinate[1],
    s=180,
    marker="*",
    edgecolors="black",
    linewidths=1.2,
    label="Query"
)

for i, chunk_id in enumerate(chunk_ids):

    ax.annotate(
        chunk_id,
        (
            chunk_coordinates[i, 0],
            chunk_coordinates[i, 1]
        ),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8
    )

ax.annotate(
    "QUERY",
    (
        query_coordinate[0],
        query_coordinate[1]
    ),
    xytext=(8, 8),
    textcoords="offset points",
    fontsize=10,
    fontweight="bold"
)

ax.set_title("UMAP — MXBAI Single-Vector Geometry")
ax.set_xlabel("UMAP dimension 1")
ax.set_ylabel("UMAP dimension 2")
ax.legend()


# ============================================================
# FIGURE TITLE
# ============================================================

fig.suptitle(
    "MXBAI Embedding Geometry: 40 Chunks + Query",
    fontsize=16,
    fontweight="bold"
)

fig.text(
    0.5,
    0.01,
    f"Query: {QUERY}",
    ha="center",
    fontsize=10
)

plt.tight_layout(
    rect=[0, 0.03, 1, 0.95]
)


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

print("\n============================================================")
print(f"Figure saved to: {OUTPUT_FILE}")
print("============================================================")

plt.show()