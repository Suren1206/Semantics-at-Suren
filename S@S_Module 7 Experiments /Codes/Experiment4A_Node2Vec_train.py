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

GRAPH_FILE = (
    MODULE_DIR
    / "Data/experiment4/com-amazon.ungraph.txt.gz"
)

OUTPUT_DIR = (
    MODULE_DIR
    / "Data/experiment4/node2vec"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Frozen common configuration
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
# Node2Vec configurations
# ============================================================

CONFIGURATIONS = {

    "N2V-1": {
        "p": 0.25,
        "q": 4.0,
        "description": "local-biased"
    },

    "N2V-2": {
        "p": 1.0,
        "q": 1.0,
        "description": "unbiased"
    },

    "N2V-3": {
        "p": 4.0,
        "q": 0.25,
        "description": "outward-biased"
    },
}


# ============================================================
# Graph loading
# ============================================================

def load_graph():

    adjacency = {}

    with gzip.open(
        GRAPH_FILE,
        "rt"
    ) as f:

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

            adjacency.setdefault(
                u,
                []
            ).append(v)

            adjacency.setdefault(
                v,
                []
            ).append(u)

    return adjacency


# ============================================================
# Node2Vec transition probability
# ============================================================

def node2vec_next_node(
    previous_node,
    current_node,
    adjacency,
    p,
    q,
    rng
):
    """
    Select the next node using Node2Vec's
    second-order biased transition rule.
    """

    neighbors = adjacency[current_node]

    if not neighbors:
        return None

    weights = []

    previous_neighbors = set(
        adjacency[previous_node]
    )

    for candidate in neighbors:

        if candidate == previous_node:

            # Return to previous node.
            weight = 1.0 / p

        elif candidate in previous_neighbors:

            # Candidate is connected to the
            # previous node: local exploration.
            weight = 1.0

        else:

            # Candidate moves outward.
            weight = 1.0 / q

        weights.append(weight)

    total_weight = sum(weights)

    threshold = (
        rng.random()
        * total_weight
    )

    cumulative = 0.0

    for candidate, weight in zip(
        neighbors,
        weights
    ):

        cumulative += weight

        if cumulative >= threshold:
            return candidate

    return neighbors[-1]


# ============================================================
# Node2Vec random walk
# ============================================================

def node2vec_walk(
    start_node,
    adjacency,
    walk_length,
    p,
    q,
    rng,
    unique_nodes_visited,
    unique_edges_traversed,
    edge_traversal_counts
):

    walk = [
        start_node
    ]

    unique_nodes_visited.add(
        start_node
    )

    current = start_node

    # First step is an ordinary random choice.
    neighbors = adjacency[current]

    if not neighbors:
        return walk

    next_node = rng.choice(
        neighbors
    )

    walk.append(next_node)

    unique_nodes_visited.add(
        next_node
    )

    edge = (
        (current, next_node)
        if current < next_node
        else (next_node, current)
    )

    unique_edges_traversed.add(
        edge
    )

    edge_traversal_counts[edge] += 1

    previous = current
    current = next_node

    # Remaining steps use Node2Vec bias.
    for _ in range(
        walk_length - 2
    ):

        next_node = node2vec_next_node(
            previous_node=previous,
            current_node=current,
            adjacency=adjacency,
            p=p,
            q=q,
            rng=rng
        )

        if next_node is None:
            break

        walk.append(
            next_node
        )

        unique_nodes_visited.add(
            next_node
        )

        edge = (
            (current, next_node)
            if current < next_node
            else (next_node, current)
        )

        unique_edges_traversed.add(
            edge
        )

        edge_traversal_counts[edge] += 1

        previous = current
        current = next_node

    return walk


# ============================================================
# Train one Node2Vec configuration
# ============================================================

def train_configuration(
    config_name,
    config,
    adjacency,
    nodes,
    node_count,
    edge_count
):

    p = config["p"]
    q = config["q"]
    description = config["description"]

    print("\n" + "=" * 70)
    print(
        f"Node2Vec configuration: "
        f"{config_name}"
    )
    print("=" * 70)

    print(
        f"p = {p}"
    )

    print(
        f"q = {q}"
    )

    print(
        f"Behaviour = {description}"
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    embedding_file = (
        OUTPUT_DIR
        / f"experiment4A_{config_name}_embeddings.npy"
    )

    metadata_file = (
        OUTPUT_DIR
        / f"experiment4A_{config_name}_metadata.json"
    )

    coverage_file = (
        OUTPUT_DIR
        / f"experiment4A_{config_name}_coverage.json"
    )

    # --------------------------------------------------------
    # Coverage trackers
    # --------------------------------------------------------

    unique_nodes_visited = set()

    unique_edges_traversed = set()

    edge_traversal_counts = Counter()

    walk_lengths = []

    total_walks = 0

    total_node_visits = 0

    total_edge_traversals = 0

    # --------------------------------------------------------
    # Random generator
    # --------------------------------------------------------

    rng = random.Random(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Generate walks
    # --------------------------------------------------------

    print(
        "\nGenerating Node2Vec walks..."
    )

    walk_generation_start = time.time()

    walk_corpus = []

    for round_number in range(
        WALKS_PER_NODE
    ):

        round_start = time.time()

        for start_node in nodes:

            walk = node2vec_walk(
                start_node=start_node,
                adjacency=adjacency,
                walk_length=WALK_LENGTH,
                p=p,
                q=q,
                rng=rng,
                unique_nodes_visited=
                    unique_nodes_visited,
                unique_edges_traversed=
                    unique_edges_traversed,
                edge_traversal_counts=
                    edge_traversal_counts
            )

            total_walks += 1

            walk_length = len(walk)

            walk_lengths.append(
                walk_length
            )

            total_node_visits += (
                walk_length
            )

            total_edge_traversals += max(
                0,
                walk_length - 1
            )

            walk_corpus.append(
                [
                    str(node)
                    for node in walk
                ]
            )

        round_elapsed = (
            time.time()
            - round_start
        )

        print(
            f"Completed walk round "
            f"{round_number + 1}/"
            f"{WALKS_PER_NODE} "
            f"in {round_elapsed:.2f} seconds"
        )

    walk_generation_time = (
        time.time()
        - walk_generation_start
    )

    print(
        f"\nGenerated "
        f"{total_walks:,} walks "
        f"in "
        f"{walk_generation_time:.2f} seconds."
    )

    # --------------------------------------------------------
    # Coverage summary
    # --------------------------------------------------------

    walk_lengths_array = np.asarray(
        walk_lengths,
        dtype=np.int64
    )

    unique_node_count = len(
        unique_nodes_visited
    )

    unique_edge_count = len(
        unique_edges_traversed
    )

    edges_never_traversed = (
        edge_count
        - unique_edge_count
    )

    node_coverage_percent = (
        unique_node_count
        / node_count
        * 100
    )

    edge_coverage_percent = (
        unique_edge_count
        / edge_count
        * 100
    )

    traversal_counts = np.asarray(
        list(
            edge_traversal_counts.values()
        ),
        dtype=np.int64
    )

    edge_exposure_stats = {

        "min":
            int(
                np.min(
                    traversal_counts
                )
            ),

        "median":
            float(
                np.median(
                    traversal_counts
                )
            ),

        "mean":
            float(
                np.mean(
                    traversal_counts
                )
            ),

        "p25":
            float(
                np.percentile(
                    traversal_counts,
                    25
                )
            ),

        "p75":
            float(
                np.percentile(
                    traversal_counts,
                    75
                )
            ),

        "p90":
            float(
                np.percentile(
                    traversal_counts,
                    90
                )
            ),

        "max":
            int(
                np.max(
                    traversal_counts
                )
            ),
    }

    # --------------------------------------------------------
    # Context pair calculation
    # --------------------------------------------------------

    total_context_pairs = 0

    for walk_length in walk_lengths:

        for center_position in range(
            walk_length
        ):

            left = max(
                0,
                center_position
                - CONTEXT_WINDOW
            )

            right = min(
                walk_length,
                center_position
                + CONTEXT_WINDOW
                + 1
            )

            total_context_pairs += (
                right
                - left
                - 1
            )

    # --------------------------------------------------------
    # Print coverage
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        f"{config_name} WALK COVERAGE"
    )

    print(
        "-" * 70
    )

    print(
        f"Unique nodes visited : "
        f"{unique_node_count:,}"
    )

    print(
        f"Node coverage        : "
        f"{node_coverage_percent:.4f}%"
    )

    print(
        f"Unique edges traversed : "
        f"{unique_edge_count:,}"
    )

    print(
        f"Edges never traversed   : "
        f"{edges_never_traversed:,}"
    )

    print(
        f"Edge coverage           : "
        f"{edge_coverage_percent:.4f}%"
    )

    print(
        f"Total walks             : "
        f"{total_walks:,}"
    )

    print(
        f"Total node visits       : "
        f"{total_node_visits:,}"
    )

    print(
        f"Total edge traversals   : "
        f"{total_edge_traversals:,}"
    )

    print(
        f"Median edge exposure    : "
        f"{edge_exposure_stats['median']:.2f}"
    )

    print(
        f"Mean edge exposure      : "
        f"{edge_exposure_stats['mean']:.2f}"
    )

    print(
        f"Total context pairs     : "
        f"{total_context_pairs:,}"
    )

    # --------------------------------------------------------
    # Save coverage
    # --------------------------------------------------------

    coverage_metadata = {

        "experiment":
            "Experiment 4A — Node2Vec",

        "coverage_type":
            "exact_training_run",

        "configuration":
            config_name,

        "p":
            p,

        "q":
            q,

        "walk_behaviour":
            description,

        "graph": {

            "nodes":
                node_count,

            "edges":
                edge_count,
        },

        "node2vec_configuration": {

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

            "p":
                p,

            "q":
                q,
        },

        "walk_generation": {

            "total_walks":
                total_walks,

            "total_node_visits":
                total_node_visits,

            "total_edge_traversals":
                total_edge_traversals,

            "minimum_walk_length":
                int(
                    np.min(
                        walk_lengths_array
                    )
                ),

            "median_walk_length":
                float(
                    np.median(
                        walk_lengths_array
                    )
                ),

            "mean_walk_length":
                float(
                    np.mean(
                        walk_lengths_array
                    )
                ),

            "maximum_walk_length":
                int(
                    np.max(
                        walk_lengths_array
                    )
                ),
        },

        "node_coverage": {

            "unique_nodes_visited":
                unique_node_count,

            "coverage_percent":
                node_coverage_percent,
        },

        "edge_coverage": {

            "unique_edges_traversed":
                unique_edge_count,

            "edges_never_traversed":
                edges_never_traversed,

            "coverage_percent":
                edge_coverage_percent,
        },

        "edge_exposure":
            edge_exposure_stats,

        "context_generation": {

            "context_window":
                CONTEXT_WINDOW,

            "total_context_pairs":
                total_context_pairs,
        },
    }

    with open(
        coverage_file,
        "w"
    ) as f:

        json.dump(
            coverage_metadata,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Train Skip-gram
    # --------------------------------------------------------

    print(
        "\nTraining Skip-gram..."
    )

    training_start = time.time()

    model = Word2Vec(

        sentences=walk_corpus,

        vector_size=
            EMBEDDING_DIM,

        window=
            CONTEXT_WINDOW,

        min_count=0,

        sg=1,

        negative=
            NEGATIVE_SAMPLES,

        workers=
            WORKERS,

        epochs=
            EPOCHS,

        seed=
            RANDOM_SEED,
    )

    training_time = (
        time.time()
        - training_start
    )

    print(
        f"Training completed in "
        f"{training_time:.2f} seconds."
    )

    # --------------------------------------------------------
    # Build embedding matrix
    # --------------------------------------------------------

    embedding_matrix = np.vstack(
        [
            model.wv[str(node_id)]
            for node_id in nodes
        ]
    )

    # --------------------------------------------------------
    # Save embedding
    # --------------------------------------------------------

    np.save(
        embedding_file,
        embedding_matrix
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {

        "experiment":
            "Experiment 4A — Node2Vec",

        "configuration":
            config_name,

        "walk_behaviour":
            description,

        "graph_file":
            str(GRAPH_FILE),

        "embedding_file":
            str(embedding_file),

        "coverage_file":
            str(coverage_file),

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

        "p":
            p,

        "q":
            q,

        "walk_strategy":
            "node2vec_biased_random_walk",

        "training_method":
            "Skip-gram",

        "node_ids":
            nodes,
    }

    with open(
        metadata_file,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print(
        "\nSaved embedding:"
    )

    print(
        embedding_file
    )

    print(
        "\nSaved metadata:"
    )

    print(
        metadata_file
    )

    print(
        "\nSaved coverage:"
    )

    print(
        coverage_file
    )

    return {
        "configuration":
            config_name,

        "p":
            p,

        "q":
            q,

        "description":
            description,

        "embedding_file":
            str(embedding_file),

        "metadata_file":
            str(metadata_file),

        "coverage_file":
            str(coverage_file),
    }


# ============================================================
# Main
# ============================================================

print("=" * 70)
print("Experiment 4A — Node2Vec")
print("Amazon Co-Purchase Graph")
print("=" * 70)

print(
    "\nLoading Amazon graph..."
)

graph_start = time.time()

adjacency = load_graph()

graph_load_time = (
    time.time()
    - graph_start
)

nodes = sorted(
    adjacency.keys()
)

node_count = len(nodes)

edge_count = (
    sum(
        len(neighbors)
        for neighbors in adjacency.values()
    )
    // 2
)

print(
    f"Nodes: {node_count:,}"
)

print(
    f"Edges: {edge_count:,}"
)

print(
    f"Graph loading time: "
    f"{graph_load_time:.2f} seconds"
)


# ============================================================
# Run all three configurations
# ============================================================

results = []

for config_name, config in CONFIGURATIONS.items():

    result = train_configuration(
        config_name=config_name,
        config=config,
        adjacency=adjacency,
        nodes=nodes,
        node_count=node_count,
        edge_count=edge_count
    )

    results.append(
        result
    )


# ============================================================
# Save experiment registry
# ============================================================

registry_file = (
    OUTPUT_DIR
    / "experiment4A_Node2Vec_registry.json"
)

registry = {

    "experiment":
        "Experiment 4A — Node2Vec",

    "graph":
        str(GRAPH_FILE),

    "common_configuration": {

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
    },

    "configurations":
        results,
}


with open(
    registry_file,
    "w"
) as f:

    json.dump(
        registry,
        f,
        indent=2
    )


print(
    "\n" + "=" * 70
)

print(
    "ALL NODE2VEC CONFIGURATIONS COMPLETED"
)

print(
    "=" * 70
)

print(
    "\nRegistry saved to:"
)

print(
    registry_file
)