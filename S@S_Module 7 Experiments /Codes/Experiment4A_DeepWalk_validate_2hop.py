from pathlib import Path
import json
import random

import numpy as np
import networkx as nx


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent

GRAPH_FILE = (
    MODULE_DIR
    / "Data"
    / "experiment4"
    / "com-amazon.ungraph.txt.gz"
)

EMBEDDING_FILE = (
    MODULE_DIR
    / "Data"
    / "experiment4"
    / "deepwalk"
    / "experiment4A_deepwalk_embeddings.npy"
)

METADATA_FILE = (
    MODULE_DIR
    / "Data"
    / "experiment4"
    / "deepwalk"
    / "experiment4A_deepwalk_metadata.json"
)

OUTPUT_FILE = (
    MODULE_DIR
    / "Data"
    / "experiment4"
    / "deepwalk"
    / "experiment4A_deepwalk_2hop_validation.json"
)

RANDOM_SEED = 42
SAMPLE_SIZE = 5000


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def cosine_similarity(vec_a, vec_b):
    """
    Compute cosine similarity between two vectors.
    """

    denominator = (
        np.linalg.norm(vec_a)
        * np.linalg.norm(vec_b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(vec_a, vec_b) / denominator
    )


def percentile(values, p):
    """
    Return percentile as a regular Python float.
    """

    return float(
        np.percentile(values, p)
    )


def summarize(values):
    """
    Produce summary statistics for a similarity distribution.
    """

    values = np.asarray(
        values,
        dtype=np.float64
    )

    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p10": percentile(values, 10),
        "p25": percentile(values, 25),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


# ============================================================
# LOAD GRAPH
# ============================================================

print("\nLoading Amazon graph...")

G = nx.read_edgelist(
    GRAPH_FILE,
    nodetype=int,
    data=False,
)

print(
    f"Graph nodes : "
    f"{G.number_of_nodes():,}"
)

print(
    f"Graph edges : "
    f"{G.number_of_edges():,}"
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("\nLoading DeepWalk embeddings...")

embeddings = np.load(
    EMBEDDING_FILE
)

print(
    f"Embedding shape : "
    f"{embeddings.shape}"
)


# ============================================================
# LOAD AUTHORITATIVE NODE MAPPING
# ============================================================

print("\nLoading DeepWalk metadata...")

with open(METADATA_FILE, "r") as f:
    metadata = json.load(f)

node_ids = metadata["node_ids"]

print(
    f"Node IDs in metadata : "
    f"{len(node_ids):,}"
)


# ============================================================
# VALIDATE EMBEDDING / NODE MAPPING
# ============================================================

if len(node_ids) != len(embeddings):

    raise ValueError(
        "Mismatch between number of node IDs "
        "and embedding rows."
    )


node_to_row = {
    node_id: row
    for row, node_id in enumerate(node_ids)
}


graph_nodes = set(G.nodes())
metadata_nodes = set(node_ids)

if graph_nodes != metadata_nodes:

    missing_from_metadata = (
        graph_nodes - metadata_nodes
    )

    extra_in_metadata = (
        metadata_nodes - graph_nodes
    )

    raise ValueError(
        "Graph node IDs and metadata node IDs do not match.\n"
        f"Missing from metadata: "
        f"{len(missing_from_metadata):,}\n"
        f"Extra in metadata: "
        f"{len(extra_in_metadata):,}"
    )


print(
    "Graph / embedding node mapping : PASSED"
)

print(
    "Embedding row order is taken "
    "from the original DeepWalk metadata."
)


# ============================================================
# GENERATE EXACTLY-2-HOP NON-EDGE PAIRS
# ============================================================

print(
    "\nSearching for exactly-2-hop "
    "non-edge pairs..."
)

random.seed(RANDOM_SEED)

two_hop_pairs = set()

nodes = list(G.nodes())

random.shuffle(nodes)

for node_a in nodes:

    neighbors_a = set(
        G.neighbors(node_a)
    )

    for intermediate in neighbors_a:

        neighbors_b = G.neighbors(
            intermediate
        )

        for node_c in neighbors_b:

            # Same node is not a pair.
            if node_c == node_a:
                continue

            # Exclude direct edges.
            if G.has_edge(
                node_a,
                node_c
            ):
                continue

            pair = (
                min(node_a, node_c),
                max(node_a, node_c),
            )

            two_hop_pairs.add(pair)

            if len(two_hop_pairs) >= SAMPLE_SIZE:
                break

        if len(two_hop_pairs) >= SAMPLE_SIZE:
            break

    if len(two_hop_pairs) >= SAMPLE_SIZE:
        break


two_hop_pairs = list(
    two_hop_pairs
)

print(
    "Exactly-2-hop non-edge pairs "
    f"collected : {len(two_hop_pairs):,}"
)


if len(two_hop_pairs) < SAMPLE_SIZE:

    print(
        "WARNING: fewer than requested "
        "2-hop pairs were found."
    )


# ============================================================
# RANDOM NON-EDGE CONTROL
# ============================================================

print(
    "\nGenerating random non-edge "
    "control pairs..."
)

random_non_edges = set()

while (
    len(random_non_edges)
    < len(two_hop_pairs)
):

    node_a = random.choice(nodes)
    node_b = random.choice(nodes)

    if node_a == node_b:
        continue

    if G.has_edge(
        node_a,
        node_b
    ):
        continue

    pair = (
        min(node_a, node_b),
        max(node_a, node_b),
    )

    random_non_edges.add(pair)


random_non_edges = list(
    random_non_edges
)

print(
    "Random non-edge pairs "
    f"collected : {len(random_non_edges):,}"
)


# ============================================================
# COMPUTE 2-HOP SIMILARITIES
# ============================================================

print(
    "\nComputing 2-hop cosine similarities..."
)

two_hop_similarities = []

for node_a, node_b in two_hop_pairs:

    row_a = node_to_row[node_a]
    row_b = node_to_row[node_b]

    similarity = cosine_similarity(
        embeddings[row_a],
        embeddings[row_b],
    )

    two_hop_similarities.append(
        similarity
    )


# ============================================================
# COMPUTE RANDOM NON-EDGE SIMILARITIES
# ============================================================

print(
    "Computing random non-edge "
    "cosine similarities..."
)

random_non_edge_similarities = []

for node_a, node_b in random_non_edges:

    row_a = node_to_row[node_a]
    row_b = node_to_row[node_b]

    similarity = cosine_similarity(
        embeddings[row_a],
        embeddings[row_b],
    )

    random_non_edge_similarities.append(
        similarity
    )


# ============================================================
# SUMMARIZE RESULTS
# ============================================================

two_hop_summary = summarize(
    two_hop_similarities
)

random_non_edge_summary = summarize(
    random_non_edge_similarities
)

mean_difference = (
    two_hop_summary["mean"]
    - random_non_edge_summary["mean"]
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "experiment":
        "Experiment 4A - DeepWalk "
        "Higher-Order Relationship Validation",

    "question":
        "Does the DeepWalk Euclidean embedding "
        "reflect higher-order graph relationships?",

    "definition": {

        "two_hop_pair":
            "Two nodes connected through at least "
            "one intermediate node, with no direct "
            "edge between the two nodes.",

        "random_non_edge":
            "Two nodes with no direct edge, sampled "
            "as a control population.",
    },

    "configuration": {

        "random_seed":
            RANDOM_SEED,

        "sample_size":
            SAMPLE_SIZE,

        "embedding_dimension":
            int(embeddings.shape[1]),
    },

    "graph": {

        "nodes":
            int(G.number_of_nodes()),

        "edges":
            int(G.number_of_edges()),
    },

    "results": {

        "exactly_2_hop_non_edge":
            two_hop_summary,

        "random_non_edge":
            random_non_edge_summary,

        "mean_difference_2hop_minus_random":
            mean_difference,
    },
}


with open(
    OUTPUT_FILE,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2,
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print(
    "DEEPWALK HIGHER-ORDER "
    "RELATIONSHIP VALIDATION"
)
print("=" * 65)


print(
    "\nExactly-2-hop non-edge pairs:"
)

print(
    f"  Count   : "
    f"{two_hop_summary['count']:,}"
)

print(
    f"  Mean    : "
    f"{two_hop_summary['mean']:.6f}"
)

print(
    f"  Median  : "
    f"{two_hop_summary['median']:.6f}"
)

print(
    f"  Std     : "
    f"{two_hop_summary['std']:.6f}"
)

print(
    f"  P10     : "
    f"{two_hop_summary['p10']:.6f}"
)

print(
    f"  P25     : "
    f"{two_hop_summary['p25']:.6f}"
)

print(
    f"  P75     : "
    f"{two_hop_summary['p75']:.6f}"
)

print(
    f"  P90     : "
    f"{two_hop_summary['p90']:.6f}"
)

print(
    f"  Min     : "
    f"{two_hop_summary['min']:.6f}"
)

print(
    f"  Max     : "
    f"{two_hop_summary['max']:.6f}"
)


print(
    "\nRandom non-edge pairs:"
)

print(
    f"  Count   : "
    f"{random_non_edge_summary['count']:,}"
)

print(
    f"  Mean    : "
    f"{random_non_edge_summary['mean']:.6f}"
)

print(
    f"  Median  : "
    f"{random_non_edge_summary['median']:.6f}"
)

print(
    f"  Std     : "
    f"{random_non_edge_summary['std']:.6f}"
)

print(
    f"  P10     : "
    f"{random_non_edge_summary['p10']:.6f}"
)

print(
    f"  P25     : "
    f"{random_non_edge_summary['p25']:.6f}"
)

print(
    f"  P75     : "
    f"{random_non_edge_summary['p75']:.6f}"
)

print(
    f"  P90     : "
    f"{random_non_edge_summary['p90']:.6f}"
)

print(
    f"  Min     : "
    f"{random_non_edge_summary['min']:.6f}"
)

print(
    f"  Max     : "
    f"{random_non_edge_summary['max']:.6f}"
)


print(
    "\nMean similarity difference "
    "(2-hop - random): "
    f"{mean_difference:.6f}"
)


print("\nResults saved to:")
print(OUTPUT_FILE)

print("\n" + "=" * 65)