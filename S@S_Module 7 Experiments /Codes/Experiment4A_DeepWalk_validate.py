# Codes/Experiment4A_DeepWalk_validate.py

from pathlib import Path
import gzip
import json
import random

import numpy as np


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent

GRAPH_FILE = MODULE_DIR / "Data/experiment4/com-amazon.ungraph.txt.gz"
EMBEDDING_FILE = (
    MODULE_DIR
    / "Data/experiment4/deepwalk/experiment4A_deepwalk_embeddings.npy"
)
METADATA_FILE = (
    MODULE_DIR
    / "Data/experiment4/deepwalk/experiment4A_deepwalk_metadata.json"
)


# ============================================================
# Validation configuration
# ============================================================

RANDOM_SEED = 42
SAMPLE_SIZE = 5000


# ============================================================
# Utilities
# ============================================================

def cosine_similarity(a, b):
    """
    Cosine similarity between two vectors.
    """
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def summarize(values, label):
    """
    Print distribution statistics.
    """
    values = np.asarray(values, dtype=np.float64)

    print(f"\n{label}")
    print("-" * len(label))

    if len(values) == 0:
        print("No observations.")
        return

    print(f"Count   : {len(values)}")
    print(f"Mean    : {np.mean(values):.6f}")
    print(f"Median  : {np.median(values):.6f}")
    print(f"Std     : {np.std(values):.6f}")
    print(f"P10     : {np.percentile(values, 10):.6f}")
    print(f"P25     : {np.percentile(values, 25):.6f}")
    print(f"P75     : {np.percentile(values, 75):.6f}")
    print(f"P90     : {np.percentile(values, 90):.6f}")
    print(f"Min     : {np.min(values):.6f}")
    print(f"Max     : {np.max(values):.6f}")


# ============================================================
# Load embeddings and metadata
# ============================================================

print("=" * 70)
print("Experiment 4A — DeepWalk Validation")
print("Amazon Co-Purchase Graph")
print("=" * 70)

print("\nLoading embeddings...")

embeddings = np.load(EMBEDDING_FILE)

with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

node_ids = metadata["node_ids"]

print(f"Embedding shape : {embeddings.shape}")
print(f"Node IDs        : {len(node_ids)}")


# ============================================================
# Basic consistency checks
# ============================================================

assert embeddings.ndim == 2, "Embeddings must be a 2D matrix."
assert embeddings.shape[0] == len(node_ids), (
    "Number of embedding rows does not match number of node IDs."
)

assert np.isnan(embeddings).sum() == 0, "NaN values found."
assert np.isinf(embeddings).sum() == 0, "Inf values found."

node_to_row = {
    int(node_id): row
    for row, node_id in enumerate(node_ids)
}

print("Embedding ↔ node mapping: OK")


# ============================================================
# Load graph edges
# ============================================================

print("\nLoading graph edges...")

edges = []

with gzip.open(GRAPH_FILE, "rt") as f:
    for line in f:

        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        parts = line.split()

        if len(parts) < 2:
            continue

        u = int(parts[0])
        v = int(parts[1])

        edges.append((u, v))


print(f"Graph edges loaded: {len(edges):,}")


# ============================================================
# 1. Directly connected pairs
# ============================================================

print("\nSampling direct graph edges...")

rng = random.Random(RANDOM_SEED)

if len(edges) > SAMPLE_SIZE:
    sampled_edges = rng.sample(edges, SAMPLE_SIZE)
else:
    sampled_edges = edges

direct_edge_similarities = []

for u, v in sampled_edges:

    if u not in node_to_row or v not in node_to_row:
        continue

    u_vec = embeddings[node_to_row[u]]
    v_vec = embeddings[node_to_row[v]]

    similarity = cosine_similarity(u_vec, v_vec)

    direct_edge_similarities.append(similarity)


# ============================================================
# 2. Random non-edge pairs
# ============================================================

print("Sampling random non-edge pairs...")

edge_set = {
    tuple(sorted((u, v)))
    for u, v in edges
}

all_nodes = list(node_to_row.keys())

random_non_edge_similarities = []

attempts = 0
max_attempts = SAMPLE_SIZE * 20

while (
    len(random_non_edge_similarities) < SAMPLE_SIZE
    and attempts < max_attempts
):

    attempts += 1

    u, v = rng.sample(all_nodes, 2)

    pair = tuple(sorted((u, v)))

    if pair in edge_set:
        continue

    u_vec = embeddings[node_to_row[u]]
    v_vec = embeddings[node_to_row[v]]

    similarity = cosine_similarity(u_vec, v_vec)

    random_non_edge_similarities.append(similarity)


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION RESULTS")
print("=" * 70)

summarize(
    direct_edge_similarities,
    "Directly Connected Products"
)

summarize(
    random_non_edge_similarities,
    "Random Non-Connected Products"
)


# ============================================================
# Difference in means
# ============================================================

if direct_edge_similarities and random_non_edge_similarities:

    direct_mean = np.mean(direct_edge_similarities)
    random_mean = np.mean(random_non_edge_similarities)

    difference = direct_mean - random_mean

    print("\nMean cosine-similarity difference")
    print("--------------------------------")
    print(f"Direct edges : {direct_mean:.6f}")
    print(f"Random pairs : {random_mean:.6f}")
    print(f"Difference   : {difference:.6f}")


# ============================================================
# Norm inspection
# ============================================================

norms = np.linalg.norm(embeddings, axis=1)

print("\nEmbedding norm distribution")
print("---------------------------")
print(f"Min    : {np.min(norms):.6f}")
print(f"Median : {np.median(norms):.6f}")
print(f"Mean   : {np.mean(norms):.6f}")
print(f"Max    : {np.max(norms):.6f}")


# ============================================================
# Completion
# ============================================================

print("\n" + "=" * 70)
print("DeepWalk validation completed.")
print("=" * 70)