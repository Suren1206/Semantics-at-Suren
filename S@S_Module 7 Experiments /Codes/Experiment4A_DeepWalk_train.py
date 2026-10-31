# Codes/Experiment4A_DeepWalk_train.py

from pathlib import Path
from collections import Counter
import gzip
import json
import random
import time

import numpy as np
from gensim.models import Word2Vec


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent

GRAPH_FILE = MODULE_DIR / "Data/experiment4/com-amazon.ungraph.txt.gz"

OUTPUT_DIR = MODULE_DIR / "Data/experiment4/deepwalk"

EMBEDDING_FILE = (
    OUTPUT_DIR / "experiment4A_deepwalk_embeddings.npy"
)

METADATA_FILE = (
    OUTPUT_DIR / "experiment4A_deepwalk_metadata.json"
)

COVERAGE_FILE = (
    OUTPUT_DIR / "experiment4A_deepwalk_coverage.json"
)


# ============================================================
# Frozen DeepWalk configuration
# ============================================================

EMBEDDING_DIM = 128

WALKS_PER_NODE = 5

WALK_LENGTH = 20

CONTEXT_WINDOW = 5

NEGATIVE_SAMPLES = 5

EPOCHS = 3

RANDOM_SEED = 42

WORKERS = 8


# ============================================================
# Graph loading
# ============================================================

def load_graph():
    """
    Load the undirected Amazon co-purchase graph.

    Returns:
        adjacency: dictionary
            node -> list of neighboring nodes
    """

    adjacency = {}

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

            adjacency.setdefault(u, []).append(v)
            adjacency.setdefault(v, []).append(u)

    return adjacency


# ============================================================
# Random walk
# ============================================================

def random_walk(
    start_node,
    adjacency,
    walk_length,
    rng,
    unique_nodes_visited,
    unique_edges_traversed,
    edge_traversal_counts,
):
    """
    Generate one uniform random walk and simultaneously
    record coverage statistics for the exact walk.

    The graph is undirected, so an edge is stored canonically
    as (smaller_node, larger_node).
    """

    walk = [start_node]

    unique_nodes_visited.add(start_node)

    current = start_node

    for _ in range(walk_length - 1):

        neighbors = adjacency[current]

        if not neighbors:
            break

        next_node = rng.choice(neighbors)

        walk.append(next_node)

        # Node exposure
        unique_nodes_visited.add(next_node)

        # Edge exposure
        edge = (
            (current, next_node)
            if current < next_node
            else (next_node, current)
        )

        unique_edges_traversed.add(edge)

        edge_traversal_counts[edge] += 1

        current = next_node

    return walk


# ============================================================
# Main
# ============================================================

print("=" * 70)
print("Experiment 4A — DeepWalk")
print("Amazon Co-Purchase Graph")
print("Coverage-Aware Training Run")
print("=" * 70)


# ============================================================
# Load graph
# ============================================================

print("\nLoading Amazon graph...")

graph_start = time.time()

adjacency = load_graph()

graph_load_time = time.time() - graph_start

nodes = sorted(adjacency.keys())

node_count = len(nodes)

edge_count = (
    sum(len(neighbors) for neighbors in adjacency.values())
    // 2
)

print("\nGraph loaded.")
print(f"Nodes: {node_count:,}")
print(f"Edges: {edge_count:,}")
print(f"Graph loading time: {graph_load_time:.2f} seconds")


# ============================================================
# Coverage trackers
# ============================================================

unique_nodes_visited = set()

unique_edges_traversed = set()

edge_traversal_counts = Counter()

walk_lengths = []

total_walks = 0

total_node_visits = 0

total_edge_traversals = 0


# ============================================================
# Generate walks
# ============================================================

print("\nGenerating DeepWalk sequences...")
print(f"Nodes: {node_count:,}")
print(f"Walks per node: {WALKS_PER_NODE}")
print(f"Walk length: {WALK_LENGTH}")

rng = random.Random(RANDOM_SEED)

walk_generation_start = time.time()


def walk_generator():

    global total_walks
    global total_node_visits
    global total_edge_traversals

    for round_number in range(WALKS_PER_NODE):

        round_start = time.time()

        for start_node in nodes:

            walk = random_walk(
                start_node=start_node,
                adjacency=adjacency,
                walk_length=WALK_LENGTH,
                rng=rng,
                unique_nodes_visited=unique_nodes_visited,
                unique_edges_traversed=unique_edges_traversed,
                edge_traversal_counts=edge_traversal_counts,
            )

            total_walks += 1

            walk_length = len(walk)

            walk_lengths.append(walk_length)

            total_node_visits += walk_length

            total_edge_traversals += max(
                0,
                walk_length - 1
            )

            # Gensim accepts sequences of tokens.
            yield [str(node) for node in walk]

        round_elapsed = time.time() - round_start

        print(
            f"Completed walk round "
            f"{round_number + 1}/{WALKS_PER_NODE} "
            f"in {round_elapsed:.2f} seconds"
        )


walks = walk_generator()

# Materialize the walks because the original training
# design uses the generated corpus for Word2Vec training.
#
# The coverage counters above are updated as the generator
# produces the walks.

walk_corpus = list(walks)

walk_generation_time = time.time() - walk_generation_start

print(
    f"\nGenerated {total_walks:,} walks "
    f"in {walk_generation_time:.2f} seconds."
)


# ============================================================
# Coverage summary BEFORE training
# ============================================================

walk_lengths_array = np.asarray(
    walk_lengths,
    dtype=np.int64
)

unique_node_count = len(unique_nodes_visited)

unique_edge_count = len(unique_edges_traversed)

edges_never_traversed = (
    edge_count - unique_edge_count
)

node_coverage_percent = (
    unique_node_count / node_count * 100
)

edge_coverage_percent = (
    unique_edge_count / edge_count * 100
)

print("\n" + "=" * 70)
print("EXACT WALK COVERAGE")
print("=" * 70)

print("\nNode coverage")
print("-------------")
print(f"Unique nodes visited : {unique_node_count:,}")
print(f"Total graph nodes    : {node_count:,}")
print(f"Coverage             : {node_coverage_percent:.4f}%")

print("\nEdge coverage")
print("-------------")
print(f"Unique edges traversed : {unique_edge_count:,}")
print(f"Total graph edges      : {edge_count:,}")
print(f"Edges never traversed  : {edges_never_traversed:,}")
print(f"Coverage               : {edge_coverage_percent:.4f}%")

print("\nWalk exposure")
print("-------------")
print(f"Total walks            : {total_walks:,}")
print(f"Total node visits      : {total_node_visits:,}")
print(f"Total edge traversals  : {total_edge_traversals:,}")

print("\nWalk length")
print("-----------")
print(f"Minimum : {np.min(walk_lengths_array)}")
print(f"Median  : {np.median(walk_lengths_array):.2f}")
print(f"Mean    : {np.mean(walk_lengths_array):.2f}")
print(f"Maximum : {np.max(walk_lengths_array)}")


# ============================================================
# Edge exposure distribution
# ============================================================

traversal_counts = np.asarray(
    list(edge_traversal_counts.values()),
    dtype=np.int64
)

edge_exposure_stats = {
    "min": int(np.min(traversal_counts)),
    "median": float(np.median(traversal_counts)),
    "mean": float(np.mean(traversal_counts)),
    "p25": float(np.percentile(traversal_counts, 25)),
    "p75": float(np.percentile(traversal_counts, 75)),
    "p90": float(np.percentile(traversal_counts, 90)),
    "max": int(np.max(traversal_counts)),
}

print("\nEdge exposure frequency")
print("-----------------------")
print(
    f"Minimum traversals/edge : "
    f"{edge_exposure_stats['min']}"
)
print(
    f"Median traversals/edge  : "
    f"{edge_exposure_stats['median']:.2f}"
)
print(
    f"Mean traversals/edge    : "
    f"{edge_exposure_stats['mean']:.2f}"
)
print(
    f"P25 traversals/edge     : "
    f"{edge_exposure_stats['p25']:.2f}"
)
print(
    f"P75 traversals/edge     : "
    f"{edge_exposure_stats['p75']:.2f}"
)
print(
    f"P90 traversals/edge     : "
    f"{edge_exposure_stats['p90']:.2f}"
)
print(
    f"Maximum traversals/edge : "
    f"{edge_exposure_stats['max']}"
)


# ============================================================
# Context-pair calculation
# ============================================================

total_context_pairs = 0

for walk_length in walk_lengths:

    for center_position in range(walk_length):

        left = max(
            0,
            center_position - CONTEXT_WINDOW
        )

        right = min(
            walk_length,
            center_position + CONTEXT_WINDOW + 1
        )

        total_context_pairs += (
            right - left - 1
        )


print("\nContext generation")
print("------------------")
print(f"Context window          : {CONTEXT_WINDOW}")
print(f"Total context pairs     : {total_context_pairs:,}")


# ============================================================
# Save coverage report
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

coverage_metadata = {

    "experiment":
        "Experiment 4A — DeepWalk",

    "coverage_type":
        "exact_training_run",

    "graph": {
        "nodes": node_count,
        "edges": edge_count,
    },

    "deepwalk_configuration": {
        "embedding_dimension": EMBEDDING_DIM,
        "walks_per_node": WALKS_PER_NODE,
        "walk_length": WALK_LENGTH,
        "context_window": CONTEXT_WINDOW,
        "negative_samples": NEGATIVE_SAMPLES,
        "epochs": EPOCHS,
        "random_seed": RANDOM_SEED,
        "workers": WORKERS,
    },

    "walk_generation": {
        "total_walks": total_walks,
        "total_node_visits": total_node_visits,
        "total_edge_traversals": total_edge_traversals,
        "minimum_walk_length":
            int(np.min(walk_lengths_array)),
        "median_walk_length":
            float(np.median(walk_lengths_array)),
        "mean_walk_length":
            float(np.mean(walk_lengths_array)),
        "maximum_walk_length":
            int(np.max(walk_lengths_array)),
    },

    "node_coverage": {
        "unique_nodes_visited": unique_node_count,
        "coverage_percent": node_coverage_percent,
    },

    "edge_coverage": {
        "unique_edges_traversed": unique_edge_count,
        "edges_never_traversed": edges_never_traversed,
        "coverage_percent": edge_coverage_percent,
    },

    "edge_exposure": edge_exposure_stats,

    "context_generation": {
        "context_window": CONTEXT_WINDOW,
        "total_context_pairs": total_context_pairs,
    },

    "timing": {
        "graph_load_seconds": graph_load_time,
        "walk_generation_seconds": walk_generation_time,
    },
}

with open(COVERAGE_FILE, "w") as f:

    json.dump(
        coverage_metadata,
        f,
        indent=2
    )

print("\nSaved exact coverage report:")
print(COVERAGE_FILE)


# ============================================================
# Train Skip-gram
# ============================================================

print("\n" + "=" * 70)
print("Training DeepWalk embeddings...")
print("=" * 70)

training_start = time.time()

model = Word2Vec(
    sentences=walk_corpus,
    vector_size=EMBEDDING_DIM,
    window=CONTEXT_WINDOW,
    min_count=0,
    sg=1,
    negative=NEGATIVE_SAMPLES,
    workers=WORKERS,
    epochs=EPOCHS,
    seed=RANDOM_SEED,
)

training_time = time.time() - training_start

print(
    f"\nTraining completed in "
    f"{training_time:.2f} seconds."
)


# ============================================================
# Build embedding matrix
# ============================================================

embedding_matrix = np.vstack(
    [
        model.wv[str(node_id)]
        for node_id in nodes
    ]
)


# ============================================================
# Save embeddings
# ============================================================

np.save(
    EMBEDDING_FILE,
    embedding_matrix
)

print("\nSaved embeddings:")
print(EMBEDDING_FILE)


# ============================================================
# Save metadata
# ============================================================

metadata = {

    "experiment":
        "Experiment 4A — DeepWalk",

    "graph_file":
        str(GRAPH_FILE),

    "embedding_file":
        str(EMBEDDING_FILE),

    "coverage_file":
        str(COVERAGE_FILE),

    "nodes":
        node_count,

    "edges":
        edge_count,

    "embedding_dimension":
        EMBEDDING_DIM,

    "walks_per_node":
        WALKS_PER_NODE,

    "walk_length":
        WALK_LENGTH,

    "context_window":
        CONTEXT_WINDOW,

    "negative_samples":
        NEGATIVE_SAMPLES,

    "epochs":
        EPOCHS,

    "random_seed":
        RANDOM_SEED,

    "workers":
        WORKERS,

    "walk_strategy":
        "uniform_random_walk",

    "training_method":
        "Skip-gram",

    "node_ids":
        nodes,
}

with open(METADATA_FILE, "w") as f:

    json.dump(
        metadata,
        f,
        indent=2
    )

print("\nSaved metadata:")
print(METADATA_FILE)


# ============================================================
# Basic embedding validation
# ============================================================

print("\nRunning basic embedding validation...")

nan_count = int(
    np.isnan(embedding_matrix).sum()
)

inf_count = int(
    np.isinf(embedding_matrix).sum()
)

norms = np.linalg.norm(
    embedding_matrix,
    axis=1
)

print(
    f"Shape: {embedding_matrix.shape}"
)

print(
    f"NaN count: {nan_count}"
)

print(
    f"Inf count: {inf_count}"
)

print(
    f"Norm min: {np.min(norms):.6f}"
)

print(
    f"Norm max: {np.max(norms):.6f}"
)

print(
    f"Norm mean: {np.mean(norms):.6f}"
)


# ============================================================
# Final
# ============================================================

print("\n" + "=" * 70)
print("DeepWalk experiment completed.")
print("=" * 70)