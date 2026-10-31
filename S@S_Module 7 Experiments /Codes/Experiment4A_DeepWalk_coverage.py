# Codes/Experiment4A_DeepWalk_coverage.py

from pathlib import Path
from collections import Counter
import gzip
import json
import random
import time

import numpy as np


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent

GRAPH_FILE = MODULE_DIR / "Data/experiment4/com-amazon.ungraph.txt.gz"

OUTPUT_DIR = MODULE_DIR / "Data/experiment4/deepwalk"

OUTPUT_FILE = OUTPUT_DIR / "experiment4A_deepwalk_coverage.json"


# ============================================================
# Frozen DeepWalk configuration
# ============================================================

WALKS_PER_NODE = 5
WALK_LENGTH = 20
CONTEXT_WINDOW = 5
RANDOM_SEED = 42


# ============================================================
# Helper functions
# ============================================================

def load_graph():
    """
    Load the undirected Amazon graph as an adjacency dictionary.
    """

    adjacency = {}

    with gzip.open(GRAPH_FILE, "rt") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            u, v = map(int, line.split()[:2])

            adjacency.setdefault(u, []).append(v)
            adjacency.setdefault(v, []).append(u)

    return adjacency


def random_walk(start_node, adjacency, walk_length, rng):
    """
    Generate one uniform random walk.

    The walk starts at start_node and repeatedly chooses
    one neighbor uniformly at random.
    """

    walk = [start_node]
    current = start_node

    for _ in range(walk_length - 1):

        neighbors = adjacency[current]

        if not neighbors:
            break

        current = rng.choice(neighbors)

        walk.append(current)

    return walk


# ============================================================
# Main
# ============================================================

print("=" * 70)
print("Experiment 4A — DeepWalk Coverage Analysis")
print("Amazon Co-Purchase Graph")
print("=" * 70)

print("\nThis is a reconstructed coverage profile.")
print("It replays the frozen DeepWalk configuration using seed 42.")
print("The original walk corpus was not saved during training.")


# ============================================================
# Load graph
# ============================================================

print("\nLoading graph...")

start_time = time.time()

adjacency = load_graph()

graph_load_time = time.time() - start_time

nodes = sorted(adjacency.keys())

node_set = set(nodes)

print(f"Nodes loaded : {len(nodes):,}")

total_edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2

print(f"Edges loaded : {total_edges:,}")
print(f"Graph load time: {graph_load_time:.2f} seconds")


# ============================================================
# Generate reconstructed walks
# ============================================================

print("\nGenerating reconstructed DeepWalk walks...")

rng = random.Random(RANDOM_SEED)

unique_nodes_visited = set()

# Each undirected edge represented canonically as (smaller, larger)
unique_edges_traversed = set()

edge_traversal_counts = Counter()

walk_lengths = []

total_walks = 0
total_node_visits = 0
total_edge_traversals = 0

# Keep track of unique walk sequences.
# We store tuples because the graph is large; this is still
# potentially memory-heavy, so we only use it for exact
# duplicate counting.
unique_walks = set()


for round_number in range(WALKS_PER_NODE):

    round_start = time.time()

    for start_node in nodes:

        walk = random_walk(
            start_node,
            adjacency,
            WALK_LENGTH,
            rng
        )

        total_walks += 1

        walk_lengths.append(len(walk))

        total_node_visits += len(walk)

        unique_nodes_visited.update(walk)

        # Store walk identity for duplicate measurement.
        unique_walks.add(tuple(walk))

        # Count actual graph transitions.
        for i in range(len(walk) - 1):

            u = walk[i]
            v = walk[i + 1]

            edge = (u, v) if u < v else (v, u)

            unique_edges_traversed.add(edge)

            edge_traversal_counts[edge] += 1

            total_edge_traversals += 1

    elapsed = time.time() - round_start

    print(
        f"Completed coverage round "
        f"{round_number + 1}/{WALKS_PER_NODE} "
        f"in {elapsed:.2f} seconds"
    )


# ============================================================
# Node coverage
# ============================================================

unique_node_count = len(unique_nodes_visited)

node_coverage_percent = (
    unique_node_count / len(nodes) * 100
)


# ============================================================
# Edge coverage
# ============================================================

unique_edge_count = len(unique_edges_traversed)

edge_coverage_percent = (
    unique_edge_count / total_edges * 100
)

edges_never_traversed = total_edges - unique_edge_count


# ============================================================
# Walk statistics
# ============================================================

walk_lengths_array = np.asarray(
    walk_lengths,
    dtype=np.int64
)

unique_walk_count = len(unique_walks)

duplicate_walk_count = total_walks - unique_walk_count

average_walk_length = float(
    np.mean(walk_lengths_array)
)

min_walk_length = int(
    np.min(walk_lengths_array)
)

max_walk_length = int(
    np.max(walk_lengths_array)
)


# ============================================================
# Edge exposure distribution
# ============================================================

traversal_counts = np.asarray(
    list(edge_traversal_counts.values()),
    dtype=np.int64
)

if len(traversal_counts) > 0:

    edge_exposure_stats = {
        "min": int(np.min(traversal_counts)),
        "median": float(np.median(traversal_counts)),
        "mean": float(np.mean(traversal_counts)),
        "p25": float(np.percentile(traversal_counts, 25)),
        "p75": float(np.percentile(traversal_counts, 75)),
        "p90": float(np.percentile(traversal_counts, 90)),
        "max": int(np.max(traversal_counts)),
    }

else:

    edge_exposure_stats = {}


# ============================================================
# Context-pair estimate
# ============================================================

# For a full-length walk of 20 nodes and window=5:
#
# each node can generate up to 5 neighbors on either side.
#
# We calculate the actual number of context pairs from the
# reconstructed walks rather than assuming every walk is full.

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


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 70)
print("COVERAGE RESULTS")
print("=" * 70)

print("\n1. Walk exposure")
print("----------------")
print(f"Total walks              : {total_walks:,}")
print(f"Configured walks/node    : {WALKS_PER_NODE}")
print(f"Configured walk length   : {WALK_LENGTH}")
print(f"Average actual length    : {average_walk_length:.4f}")
print(f"Minimum actual length    : {min_walk_length}")
print(f"Maximum actual length    : {max_walk_length}")

print("\n2. Node coverage")
print("----------------")
print(f"Graph nodes              : {len(nodes):,}")
print(f"Unique nodes visited     : {unique_node_count:,}")
print(f"Node coverage            : {node_coverage_percent:.4f}%")

print("\n3. Edge coverage")
print("----------------")
print(f"Graph edges              : {total_edges:,}")
print(f"Unique edges traversed   : {unique_edge_count:,}")
print(f"Edges never traversed    : {edges_never_traversed:,}")
print(f"Edge coverage            : {edge_coverage_percent:.4f}%")

print("\n4. Edge exposure")
print("----------------")
print(f"Total edge traversals    : {total_edge_traversals:,}")
print(
    f"Unique traversed edges   : "
    f"{unique_edge_count:,}"
)

if edge_exposure_stats:

    print(
        f"Minimum traversals/edge  : "
        f"{edge_exposure_stats['min']}"
    )

    print(
        f"Median traversals/edge   : "
        f"{edge_exposure_stats['median']:.2f}"
    )

    print(
        f"Mean traversals/edge     : "
        f"{edge_exposure_stats['mean']:.2f}"
    )

    print(
        f"P25 traversals/edge      : "
        f"{edge_exposure_stats['p25']:.2f}"
    )

    print(
        f"P75 traversals/edge      : "
        f"{edge_exposure_stats['p75']:.2f}"
    )

    print(
        f"P90 traversals/edge      : "
        f"{edge_exposure_stats['p90']:.2f}"
    )

    print(
        f"Maximum traversals/edge  : "
        f"{edge_exposure_stats['max']}"
    )

print("\n5. Walk repetition")
print("------------------")
print(f"Unique walks             : {unique_walk_count:,}")
print(f"Duplicate walks          : {duplicate_walk_count:,}")

duplicate_percent = (
    duplicate_walk_count / total_walks * 100
)

print(f"Duplicate walk %         : {duplicate_percent:.4f}%")

print("\n6. Context generation")
print("---------------------")
print(f"Context window           : {CONTEXT_WINDOW}")
print(f"Estimated context pairs  : {total_context_pairs:,}")


# ============================================================
# Save results
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

results = {
    "experiment": "Experiment 4A — DeepWalk Coverage Analysis",
    "coverage_type": "reconstructed_sampling_profile",
    "random_seed": RANDOM_SEED,

    "graph": {
        "nodes": len(nodes),
        "edges": total_edges,
    },

    "deepwalk_configuration": {
        "walks_per_node": WALKS_PER_NODE,
        "walk_length": WALK_LENGTH,
        "context_window": CONTEXT_WINDOW,
    },

    "walk_exposure": {
        "total_walks": total_walks,
        "total_node_visits": total_node_visits,
        "average_walk_length": average_walk_length,
        "min_walk_length": min_walk_length,
        "max_walk_length": max_walk_length,
    },

    "node_coverage": {
        "unique_nodes_visited": unique_node_count,
        "coverage_percent": node_coverage_percent,
    },

    "edge_coverage": {
        "total_edges": total_edges,
        "unique_edges_traversed": unique_edge_count,
        "edges_never_traversed": edges_never_traversed,
        "coverage_percent": edge_coverage_percent,
    },

    "edge_exposure": edge_exposure_stats,

    "walk_repetition": {
        "unique_walks": unique_walk_count,
        "duplicate_walks": duplicate_walk_count,
        "duplicate_percent": duplicate_percent,
    },

    "context_generation": {
        "context_window": CONTEXT_WINDOW,
        "total_context_pairs": total_context_pairs,
    },
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(
        results,
        f,
        indent=2
    )


print("\n" + "=" * 70)
print("Coverage analysis completed.")
print("=" * 70)

print(f"\nSaved coverage report:")
print(OUTPUT_FILE)