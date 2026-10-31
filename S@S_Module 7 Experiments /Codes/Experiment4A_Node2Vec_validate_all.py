from pathlib import Path
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

GRAPH_FILE = (
    MODULE_DIR
    / "Data/experiment4/com-amazon.ungraph.txt.gz"
)

EMBEDDING_DIR = (
    MODULE_DIR
    / "Data/experiment4/node2vec"
)

OUTPUT_FILE = (
    EMBEDDING_DIR
    / "experiment4A_Node2Vec_validation_all.json"
)


EMBEDDING_FILES = {
    "N2V-1": EMBEDDING_DIR / "experiment4A_N2V-1_embeddings.npy",
    "N2V-2": EMBEDDING_DIR / "experiment4A_N2V-2_embeddings.npy",
    "N2V-3": EMBEDDING_DIR / "experiment4A_N2V-3_embeddings.npy",
}


# ============================================================
# Frozen validation configuration
# ============================================================

RANDOM_SEED = 42

SAMPLE_SIZE = 5000


# ============================================================
# Load graph
# ============================================================

def load_graph():

    adjacency = {}

    edges = set()

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

            edge = (
                (u, v)
                if u < v
                else (v, u)
            )

            edges.add(edge)

    return adjacency, edges


# ============================================================
# Similarity
# ============================================================

def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# ============================================================
# Statistics
# ============================================================

def calculate_statistics(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


# ============================================================
# Node / embedding mapping
# ============================================================

def load_embedding_metadata():

    metadata_files = {
        "N2V-1":
            EMBEDDING_DIR
            / "experiment4A_N2V-1_metadata.json",

        "N2V-2":
            EMBEDDING_DIR
            / "experiment4A_N2V-2_metadata.json",

        "N2V-3":
            EMBEDDING_DIR
            / "experiment4A_N2V-3_metadata.json",
    }

    mappings = {}

    for name, metadata_file in metadata_files.items():

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        node_ids = metadata["node_ids"]

        mappings[name] = {
            node_id: row
            for row, node_id
            in enumerate(node_ids)
        }

    return mappings


# ============================================================
# Direct-edge sample
# ============================================================

def sample_direct_edges(
    edges,
    sample_size,
    rng
):

    edge_list = list(edges)

    rng.shuffle(edge_list)

    return edge_list[:sample_size]


# ============================================================
# Random non-edge sample
# ============================================================

def sample_random_non_edges(
    nodes,
    edges,
    sample_size,
    rng
):

    sampled = []

    edge_set = edges

    node_count = len(nodes)

    while len(sampled) < sample_size:

        u = nodes[
            rng.randrange(node_count)
        ]

        v = nodes[
            rng.randrange(node_count)
        ]

        if u == v:
            continue

        edge = (
            (u, v)
            if u < v
            else (v, u)
        )

        if edge in edge_set:
            continue

        sampled.append(edge)

    return sampled


# ============================================================
# Exactly-2-hop sample
# ============================================================

def sample_two_hop_pairs(
    nodes,
    adjacency,
    edges,
    sample_size,
    rng
):

    sampled = []

    shuffled_nodes = list(nodes)

    rng.shuffle(shuffled_nodes)

    for u in shuffled_nodes:

        if len(sampled) >= sample_size:
            break

        neighbors_u = adjacency[u]

        candidates = set()

        for middle in neighbors_u:

            for v in adjacency[middle]:

                if v == u:
                    continue

                edge = (
                    (u, v)
                    if u < v
                    else (v, u)
                )

                if edge in edges:
                    continue

                candidates.add(v)

        candidates = list(candidates)

        rng.shuffle(candidates)

        for v in candidates:

            sampled.append(
                (u, v)
            )

            if len(sampled) >= sample_size:
                break

    return sampled


# ============================================================
# Evaluate pairs
# ============================================================

def evaluate_pairs(
    pairs,
    embedding,
    node_to_row
):

    similarities = []

    for u, v in pairs:

        row_u = node_to_row[u]
        row_v = node_to_row[v]

        similarity = cosine_similarity(
            embedding[row_u],
            embedding[row_v]
        )

        similarities.append(similarity)

    return similarities


# ============================================================
# Main
# ============================================================

print("=" * 75)
print("Experiment 4A — Node2Vec Comparative Validation")
print("N2V-1 vs N2V-2 vs N2V-3")
print("=" * 75)


# ============================================================
# Load graph
# ============================================================

print("\nLoading Amazon graph...")

start = time.time()

adjacency, edges = load_graph()

graph_time = time.time() - start

nodes = sorted(adjacency.keys())

print(
    f"Graph loaded in {graph_time:.2f} seconds."
)

print(
    f"Nodes : {len(nodes):,}"
)

print(
    f"Edges : {len(edges):,}"
)


# ============================================================
# Load metadata mappings
# ============================================================

print("\nLoading embedding metadata...")

node_mappings = load_embedding_metadata()

for name in ["N2V-1", "N2V-2", "N2V-3"]:

    print(
        f"{name}: "
        f"{len(node_mappings[name]):,} node mappings"
    )


# ============================================================
# Load embeddings
# ============================================================

print("\nLoading embeddings...")

embeddings = {}

for name, path in EMBEDDING_FILES.items():

    embedding = np.load(path)

    embeddings[name] = embedding

    print(
        f"{name}: "
        f"shape={embedding.shape}, "
        f"dtype={embedding.dtype}"
    )


# ============================================================
# Validate matrix integrity
# ============================================================

print("\nEmbedding integrity")

for name, embedding in embeddings.items():

    nan_count = int(
        np.isnan(embedding).sum()
    )

    inf_count = int(
        np.isinf(embedding).sum()
    )

    norms = np.linalg.norm(
        embedding,
        axis=1
    )

    print(
        f"{name}: "
        f"NaN={nan_count}, "
        f"Inf={inf_count}, "
        f"mean_norm={np.mean(norms):.6f}"
    )


# ============================================================
# Create common validation populations
# ============================================================

rng = random.Random(
    RANDOM_SEED
)


print("\nCreating common validation populations...")

direct_pairs = sample_direct_edges(
    edges,
    SAMPLE_SIZE,
    rng
)

random_pairs = sample_random_non_edges(
    nodes,
    edges,
    SAMPLE_SIZE,
    rng
)

two_hop_pairs = sample_two_hop_pairs(
    nodes,
    adjacency,
    edges,
    SAMPLE_SIZE,
    rng
)

print(
    f"Direct-edge pairs       : {len(direct_pairs):,}"
)

print(
    f"Random non-edge pairs   : {len(random_pairs):,}"
)

print(
    f"Exactly-2-hop pairs     : {len(two_hop_pairs):,}"
)


# ============================================================
# Evaluate all three embeddings
# ============================================================

results = {}


for name in ["N2V-1", "N2V-2", "N2V-3"]:

    print("\n" + "=" * 75)

    print(
        f"Evaluating {name}"
    )

    print("=" * 75)

    embedding = embeddings[name]

    mapping = node_mappings[name]


    # --------------------------------------------------------
    # Direct relationships
    # --------------------------------------------------------

    direct_similarities = evaluate_pairs(
        direct_pairs,
        embedding,
        mapping
    )

    random_similarities = evaluate_pairs(
        random_pairs,
        embedding,
        mapping
    )

    direct_stats = calculate_statistics(
        direct_similarities
    )

    random_stats = calculate_statistics(
        random_similarities
    )

    direct_difference = (
        direct_stats["mean"]
        - random_stats["mean"]
    )


    # --------------------------------------------------------
    # Higher-order relationships
    # --------------------------------------------------------

    two_hop_similarities = evaluate_pairs(
        two_hop_pairs,
        embedding,
        mapping
    )

    two_hop_stats = calculate_statistics(
        two_hop_similarities
    )

    two_hop_difference = (
        two_hop_stats["mean"]
        - random_stats["mean"]
    )


    results[name] = {

        "direct_edge": {
            **direct_stats,
            "difference_vs_random_mean":
                direct_difference,
        },

        "random_non_edge": random_stats,

        "two_hop": {
            **two_hop_stats,
            "difference_vs_random_mean":
                two_hop_difference,
        },
    }


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\nDirect edges")

    print(
        f"Mean   : "
        f"{direct_stats['mean']:.6f}"
    )

    print(
        f"Median : "
        f"{direct_stats['median']:.6f}"
    )

    print(
        f"P10    : "
        f"{direct_stats['p10']:.6f}"
    )

    print(
        f"P90    : "
        f"{direct_stats['p90']:.6f}"
    )


    print("\nRandom non-edges")

    print(
        f"Mean   : "
        f"{random_stats['mean']:.6f}"
    )

    print(
        f"Median : "
        f"{random_stats['median']:.6f}"
    )


    print(
        "\nDirect − Random mean difference: "
        f"{direct_difference:.6f}"
    )


    print("\nExactly-2-hop pairs")

    print(
        f"Mean   : "
        f"{two_hop_stats['mean']:.6f}"
    )

    print(
        f"Median : "
        f"{two_hop_stats['median']:.6f}"
    )

    print(
        f"P10    : "
        f"{two_hop_stats['p10']:.6f}"
    )

    print(
        f"P90    : "
        f"{two_hop_stats['p90']:.6f}"
    )


    print(
        "\n2-hop − Random mean difference: "
        f"{two_hop_difference:.6f}"
    )


# ============================================================
# Comparative summary
# ============================================================

print("\n\n")
print("=" * 100)
print("COMPARATIVE SUMMARY")
print("=" * 100)

print(
    "\n"
    f"{'Metric':<32}"
    f"{'N2V-1':>14}"
    f"{'N2V-2':>14}"
    f"{'N2V-3':>14}"
)

print("-" * 100)


metrics = [

    (
        "Direct edge mean",
        "direct_edge",
        "mean"
    ),

    (
        "Direct edge median",
        "direct_edge",
        "median"
    ),

    (
        "Random non-edge mean",
        "random_non_edge",
        "mean"
    ),

    (
        "Direct − Random",
        "direct_edge",
        "difference_vs_random_mean"
    ),

    (
        "2-hop mean",
        "two_hop",
        "mean"
    ),

    (
        "2-hop median",
        "two_hop",
        "median"
    ),

    (
        "2-hop − Random",
        "two_hop",
        "difference_vs_random_mean"
    ),
]


for label, section, key in metrics:

    print(
        f"{label:<32}"
        f"{results['N2V-1'][section][key]:>14.6f}"
        f"{results['N2V-2'][section][key]:>14.6f}"
        f"{results['N2V-3'][section][key]:>14.6f}"
    )


# ============================================================
# Save results
# ============================================================

validation_metadata = {

    "experiment":
        "Experiment 4A — Node2Vec Comparative Validation",

    "random_seed":
        RANDOM_SEED,

    "sample_size":
        SAMPLE_SIZE,

    "graph": {
        "nodes":
            len(nodes),

        "edges":
            len(edges),
    },

    "validation_populations": {
        "direct_edges":
            len(direct_pairs),

        "random_non_edges":
            len(random_pairs),

        "exactly_two_hop":
            len(two_hop_pairs),
    },

    "results":
        results,
}


with open(
    OUTPUT_FILE,
    "w"
) as f:

    json.dump(
        validation_metadata,
        f,
        indent=2
    )


print("\n" + "=" * 75)

print(
    "Validation completed."
)

print(
    f"Results saved to:\n{OUTPUT_FILE}"
)

print("=" * 75)