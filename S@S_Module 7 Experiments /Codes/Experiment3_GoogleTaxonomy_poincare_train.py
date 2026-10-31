from pathlib import Path
import json
import random

import numpy as np
import torch
import geoopt


# ============================================================
# Experiment 3 - Google Product Taxonomy
# Step 4: Poincare Hyperbolic Embedding
#
# Run from:
# ~/LangChain/Semantics@Suren/S@S_Module 7 Experiments
#
# Purpose:
# Learn a hyperbolic representation directly from the
# hierarchical structure of the Google Product Taxonomy.
#
# No semantic text embeddings are used.
# ============================================================


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data"

INPUT_FILE = (
    DATA_DIR /
    "experiment3_google_taxonomy_hierarchy.json"
)

EMBEDDING_FILE = (
    DATA_DIR /
    "experiment3_google_taxonomy_poincare_embeddings.npy"
)

METADATA_FILE = (
    DATA_DIR /
    "experiment3_google_taxonomy_poincare_metadata.json"
)


# ------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------

EMBEDDING_DIM = 10

CURVATURE = 1.0

LEARNING_RATE = 0.01

EPOCHS = 300

NEGATIVE_SAMPLES = 10

BATCH_SIZE = 512

MARGIN = 1.0


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ------------------------------------------------------------
# Load hierarchy
# ------------------------------------------------------------

def load_hierarchy():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Hierarchy file not found:\n{INPUT_FILE}"
        )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data


# ------------------------------------------------------------
# Build node information
# ------------------------------------------------------------

def build_node_information(data):

    nodes = data["nodes"]

    node_ids = [
        node["id"]
        for node in nodes
    ]

    node_id_to_index = {
        node_id: index
        for index, node_id in enumerate(node_ids)
    }

    index_to_node = {
        index: node
        for index, node in enumerate(nodes)
    }

    return (
        node_ids,
        node_id_to_index,
        index_to_node
    )


# ------------------------------------------------------------
# Build parent lookup
# ------------------------------------------------------------

def build_parent_lookup(data):

    parent_of = {}

    for edge in data["edges"]:

        parent_id = edge["parent_id"]
        child_id = edge["child_id"]

        parent_of[child_id] = parent_id

    return parent_of


# ------------------------------------------------------------
# Build ancestor-descendant relationships
#
# For every node:
#
# child
#   ↓
# parent
#   ↓
# grandparent
#   ↓
# ...
#
# We create a positive relationship between the node and
# every ancestor.
# ------------------------------------------------------------

def build_ancestor_pairs(
    node_ids,
    parent_of,
    node_id_to_index
):

    pairs = []

    for node_id in node_ids:

        current = node_id

        while current in parent_of:

            parent = parent_of[current]

            child_index = node_id_to_index[node_id]
            ancestor_index = node_id_to_index[parent]

            pairs.append(
                (
                    ancestor_index,
                    child_index
                )
            )

            current = parent

    return pairs


# ------------------------------------------------------------
# Convert pairs to tensors
# ------------------------------------------------------------

def create_pair_tensors(pairs):

    parents = torch.tensor(
        [pair[0] for pair in pairs],
        dtype=torch.long,
        device=DEVICE
    )

    children = torch.tensor(
        [pair[1] for pair in pairs],
        dtype=torch.long,
        device=DEVICE
    )

    return parents, children


# ------------------------------------------------------------
# Hyperbolic distance
# ------------------------------------------------------------

def poincare_distance(
    manifold,
    x,
    y
):

    return manifold.dist(
        x,
        y
    )


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

def train_embeddings(
    num_nodes,
    parents,
    children
):

    manifold = geoopt.PoincareBall(
        c=CURVATURE
    )

    # Small initial coordinates inside the Poincare ball.
    initial = (
        torch.randn(
            num_nodes,
            EMBEDDING_DIM,
            device=DEVICE
        )
        * 0.001
    )

    embeddings = geoopt.ManifoldParameter(
        initial,
        manifold=manifold
    )

    optimizer = geoopt.optim.RiemannianAdam(
        [embeddings],
        lr=LEARNING_RATE
    )

    num_pairs = len(parents)

    print("\nTraining configuration:")
    print(f"  Nodes:             {num_nodes:,}")
    print(f"  Positive pairs:    {num_pairs:,}")
    print(f"  Dimensions:        {EMBEDDING_DIM}")
    print(f"  Curvature:         -{CURVATURE}")
    print(f"  Negative samples:  {NEGATIVE_SAMPLES}")
    print(f"  Batch size:        {BATCH_SIZE}")
    print(f"  Epochs:            {EPOCHS}")
    print(f"  Learning rate:     {LEARNING_RATE}")
    print(f"  Device:            {DEVICE}")

    for epoch in range(1, EPOCHS + 1):

        permutation = torch.randperm(
            num_pairs,
            device=DEVICE
        )

        epoch_loss = 0.0
        batch_count = 0

        for start in range(
            0,
            num_pairs,
            BATCH_SIZE
        ):

            batch_indices = permutation[
                start:start + BATCH_SIZE
            ]

            batch_parents = parents[
                batch_indices
            ]

            batch_children = children[
                batch_indices
            ]

            positive_parent = embeddings[
                batch_parents
            ]

            positive_child = embeddings[
                batch_children
            ]

            positive_distance = poincare_distance(
                manifold,
                positive_parent,
                positive_child
            )

            # ------------------------------------------------
            # Negative sampling
            #
            # Keep the parent fixed and replace the child
            # with randomly selected nodes.
            # ------------------------------------------------

            negative_indices = torch.randint(
                0,
                num_nodes,
                (
                    len(batch_indices),
                    NEGATIVE_SAMPLES
                ),
                device=DEVICE
            )

            negative_nodes = embeddings[
                negative_indices
            ]

            repeated_parents = (
                positive_parent
                .unsqueeze(1)
                .expand(
                    -1,
                    NEGATIVE_SAMPLES,
                    -1
                )
            )

            negative_distances = poincare_distance(
                manifold,
                repeated_parents.reshape(
                    -1,
                    EMBEDDING_DIM
                ),
                negative_nodes.reshape(
                    -1,
                    EMBEDDING_DIM
                )
            )

            negative_distances = negative_distances.reshape(
                -1,
                NEGATIVE_SAMPLES
            )

            # ------------------------------------------------
            # Margin ranking loss
            #
            # Positive ancestor-descendant pairs should have
            # smaller hyperbolic distance than negatives.
            # ------------------------------------------------

            loss = torch.relu(
                positive_distance.unsqueeze(1)
                - negative_distances
                + MARGIN
            ).mean()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        average_loss = epoch_loss / batch_count

        if (
            epoch == 1
            or epoch % 25 == 0
            or epoch == EPOCHS
        ):

            print(
                f"Epoch {epoch:4d}/{EPOCHS} "
                f"| Loss: {average_loss:.6f}"
            )

    return embeddings.detach(), manifold


# ------------------------------------------------------------
# Save embeddings
# ------------------------------------------------------------

def save_embeddings(
    embeddings,
    nodes,
    data
):

    # Move from GPU to CPU.
    embedding_array = (
        embeddings
        .detach()
        .cpu()
        .numpy()
    )

    np.save(
        EMBEDDING_FILE,
        embedding_array
    )

    metadata = {
        "experiment": (
            "Google Product Taxonomy "
            "Poincare Hyperbolic Embedding"
        ),
        "source_file": INPUT_FILE.name,
        "taxonomy_version": data.get(
            "taxonomy_version"
        ),
        "num_nodes": len(nodes),
        "embedding_dimension": EMBEDDING_DIM,
        "curvature_parameter": CURVATURE,
        "geometric_curvature": -CURVATURE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "negative_samples": NEGATIVE_SAMPLES,
        "batch_size": BATCH_SIZE,
        "margin": MARGIN,
        "random_seed": SEED,
        "device": str(DEVICE),
        "node_mapping": [
            {
                "row": index,
                "category_id": node["id"],
                "category_name": node["name"],
                "category_path": node["path"],
                "depth": node["depth"],
            }
            for index, node in enumerate(nodes)
        ],
    }

    with METADATA_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False
        )

    return embedding_array


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print(
        "GOOGLE PRODUCT TAXONOMY - "
        "POINCARE HYPERBOLIC EMBEDDING"
    )
    print("=" * 70)

    print(f"\nDevice selected: {DEVICE}")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_hierarchy()

    nodes = data["nodes"]

    print(
        f"\nLoaded taxonomy hierarchy:"
    )

    print(
        f"  Nodes: {len(nodes):,}"
    )

    print(
        f"  Immediate parent-child edges: "
        f"{len(data['edges']):,}"
    )

    # --------------------------------------------------------
    # Node mappings
    # --------------------------------------------------------

    (
        node_ids,
        node_id_to_index,
        index_to_node
    ) = build_node_information(data)

    parent_of = build_parent_lookup(data)

    # --------------------------------------------------------
    # Ancestor relationships
    # --------------------------------------------------------

    pairs = build_ancestor_pairs(
        node_ids,
        parent_of,
        node_id_to_index
    )

    print(
        f"\nAncestor-descendant relationships: "
        f"{len(pairs):,}"
    )

    # --------------------------------------------------------
    # Tensor conversion
    # --------------------------------------------------------

    parents, children = create_pair_tensors(
        pairs
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    embeddings, manifold = train_embeddings(
        len(nodes),
        parents,
        children
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    embedding_array = save_embeddings(
        embeddings,
        nodes,
        data
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nEmbedding shape: "
        f"{embedding_array.shape}"
    )

    print(
        f"\nSaved embedding matrix:"
    )

    print(
        f"  {EMBEDDING_FILE}"
    )

    print(
        f"\nSaved metadata:"
    )

    print(
        f"  {METADATA_FILE}"
    )

    print(
        "\nThe .npy file contains the learned numerical "
        "coordinates."
    )

    print(
        "Those coordinates must be interpreted using "
        "Poincare-ball geometry."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()