from pathlib import Path
import json
import random
import numpy as np
import torch
import geoopt


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDING_FILE = (
    BASE_DIR
    / "Data"
    / "experiment3_google_taxonomy_poincare_embeddings.npy"
)

METADATA_FILE = (
    BASE_DIR
    / "Data"
    / "experiment3_google_taxonomy_poincare_metadata.json"
)

OUTPUT_FILE = (
    BASE_DIR
    / "Data"
    / "experiment3_google_taxonomy_poincare_validation.json"
)


# ============================================================
# Configuration
# ============================================================

CURVATURE = 1.0

# Fixed seed for reproducible sampling where sampling is needed.
RANDOM_SEED = 42

# Maximum number of pairs used when a complete pair population
# would become unnecessarily large.
MAX_SAMPLED_PAIRS = 5000

# For depth-specific cross-branch and cross-level groups.
MAX_DEPTH_GROUP_PAIRS = 5000


# ============================================================
# Reproducibility
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# Helper functions
# ============================================================

def distance_statistics(distances, population_type):
    """
    Calculate descriptive statistics for a distance distribution.
    """

    distances = np.asarray(distances, dtype=np.float64)

    if len(distances) == 0:
        return {
            "count": 0,
            "population_type": population_type,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "max": None,
        }

    return {
        "count": int(len(distances)),
        "population_type": population_type,
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "std": float(np.std(distances)),
        "min": float(np.min(distances)),
        "p10": float(np.percentile(distances, 10)),
        "p25": float(np.percentile(distances, 25)),
        "p75": float(np.percentile(distances, 75)),
        "p90": float(np.percentile(distances, 90)),
        "max": float(np.max(distances)),
    }


def pairwise_distances(embeddings, pairs, manifold):
    """
    Calculate Poincare distances for a list of index pairs.
    """

    if len(pairs) == 0:
        return np.array([], dtype=np.float64)

    idx_a = torch.tensor(
        [a for a, _ in pairs],
        dtype=torch.long,
    )

    idx_b = torch.tensor(
        [b for _, b in pairs],
        dtype=torch.long,
    )

    points_a = embeddings[idx_a]
    points_b = embeddings[idx_b]

    with torch.no_grad():
        distances = manifold.dist(points_a, points_b)

    return distances.cpu().numpy()


def sample_pairs(pairs, max_pairs, rng):
    """
    Return all pairs if small enough.
    Otherwise return a reproducible random sample.
    """

    if len(pairs) <= max_pairs:
        return pairs, "full_population"

    sampled = rng.sample(pairs, max_pairs)

    return sampled, "sampled"


def print_stats(label, stats):
    """
    Human-readable console output.
    """

    print(f"\n{label}")

    if stats["count"] == 0:
        print("  No pairs available.")
        return

    print(f"  Population type : {stats['population_type']}")
    print(f"  Count           : {stats['count']}")
    print(f"  Mean            : {stats['mean']:.6f}")
    print(f"  Median          : {stats['median']:.6f}")
    print(f"  Std             : {stats['std']:.6f}")
    print(f"  P10             : {stats['p10']:.6f}")
    print(f"  P25             : {stats['p25']:.6f}")
    print(f"  P75             : {stats['p75']:.6f}")
    print(f"  P90             : {stats['p90']:.6f}")
    print(f"  Min             : {stats['min']:.6f}")
    print(f"  Max             : {stats['max']:.6f}")


# ============================================================
# Load representation
# ============================================================

print("=" * 70)
print("Experiment 3 — Google Product Taxonomy")
print("Poincaré Representation Validation")
print("=" * 70)

embeddings_np = np.load(EMBEDDING_FILE)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

node_mapping = metadata["node_mapping"]

print(f"\nEmbedding shape: {embeddings_np.shape}")
print(f"Number of nodes: {len(node_mapping)}")


# ============================================================
# Convert embeddings to Poincaré manifold representation
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embeddings = torch.tensor(
    embeddings_np,
    dtype=torch.float32,
    device=device,
)

manifold = geoopt.PoincareBall(c=CURVATURE)


# ============================================================
# Build node metadata
# ============================================================

row_to_node = {}

for item in node_mapping:
    row = item["row"]

    row_to_node[row] = {
        "category_id": item["category_id"],
        "category_name": item["category_name"],
        "category_path": item["category_path"],
        "depth": item["depth"],
    }


# ============================================================
# Derive parent relationships from category paths
# ============================================================

path_to_row = {
    item["category_path"]: item["row"]
    for item in node_mapping
}

parent_of = {}

missing_parent_links = []

for row, node in row_to_node.items():

    path = node["category_path"]

    if " > " not in path:
        continue

    parent_path = path.rsplit(" > ", 1)[0]

    if parent_path not in path_to_row:
        missing_parent_links.append(
            {
                "row": row,
                "category_path": path,
                "expected_parent": parent_path,
            }
        )
    else:
        parent_of[row] = path_to_row[parent_path]


print(f"Missing parent links: {len(missing_parent_links)}")


# ============================================================
# Depth information
# ============================================================

depth_to_rows = {}

for row, node in row_to_node.items():

    depth = node["depth"]

    depth_to_rows.setdefault(depth, []).append(row)


available_depths = sorted(depth_to_rows.keys())
max_depth = max(available_depths)

print(f"Available depths: {available_depths}")
print(f"Maximum depth: {max_depth}")


# ============================================================
# Children / siblings
# ============================================================

children_of = {}

for child, parent in parent_of.items():

    children_of.setdefault(parent, []).append(child)


# ============================================================
# Random generator
# ============================================================

rng = random.Random(RANDOM_SEED)


# ============================================================
# Validation results
# ============================================================

results = {
    "experiment": "Google Product Taxonomy Poincare Representation Validation",
    "embedding_file": str(EMBEDDING_FILE),
    "metadata_file": str(METADATA_FILE),
    "num_nodes": len(node_mapping),
    "embedding_dimension": int(embeddings_np.shape[1]),
    "curvature_parameter": CURVATURE,
    "random_seed": RANDOM_SEED,
    "available_depths": available_depths,
    "maximum_depth": max_depth,
    "missing_parent_links": len(missing_parent_links),
}


# ============================================================
# 1. Parent -> Child
# ============================================================

parent_child_pairs = [
    (parent, child)
    for child, parent in parent_of.items()
]

parent_child_sample, parent_child_type = sample_pairs(
    parent_child_pairs,
    MAX_SAMPLED_PAIRS,
    rng,
)

parent_child_distances = pairwise_distances(
    embeddings,
    parent_child_sample,
    manifold,
)

stats = distance_statistics(
    parent_child_distances,
    parent_child_type,
)

results["parent_child"] = stats

print_stats(
    "1. Parent -> Child",
    stats,
)


# ============================================================
# 2. Sibling -> Sibling
# ============================================================

sibling_pairs = []

for parent, children in children_of.items():

    if len(children) < 2:
        continue

    for i in range(len(children)):
        for j in range(i + 1, len(children)):

            sibling_pairs.append(
                (children[i], children[j])
            )


sibling_sample, sibling_type = sample_pairs(
    sibling_pairs,
    MAX_SAMPLED_PAIRS,
    rng,
)

sibling_distances = pairwise_distances(
    embeddings,
    sibling_sample,
    manifold,
)

stats = distance_statistics(
    sibling_distances,
    sibling_type,
)

results["siblings"] = stats

print_stats(
    "2. Sibling -> Sibling",
    stats,
)


# ============================================================
# 3. Same depth -> different branch
# ============================================================

same_depth_cross_branch_results = {}

for depth in available_depths:

    rows = depth_to_rows[depth]

    if depth == 0:
        same_depth_cross_branch_results[str(depth)] = {
            "count": 0,
            "population_type": "not_applicable",
        }

        continue

    # Group nodes by immediate parent.
    parent_groups = {}

    for row in rows:

        parent = parent_of.get(row)

        if parent is None:
            continue

        parent_groups.setdefault(parent, []).append(row)

    parent_groups = list(parent_groups.values())

    cross_branch_pairs = []

    # Compare nodes belonging to different immediate-parent groups.
    for i in range(len(parent_groups)):

        for j in range(i + 1, len(parent_groups)):

            group_a = parent_groups[i]
            group_b = parent_groups[j]

            for a in group_a:
                for b in group_b:

                    cross_branch_pairs.append((a, b))

    sampled_pairs, pair_type = sample_pairs(
        cross_branch_pairs,
        MAX_DEPTH_GROUP_PAIRS,
        rng,
    )

    distances = pairwise_distances(
        embeddings,
        sampled_pairs,
        manifold,
    )

    stats = distance_statistics(
        distances,
        pair_type,
    )

    same_depth_cross_branch_results[str(depth)] = stats

    print_stats(
        f"3. Same Depth {depth} -> Different Branch",
        stats,
    )


results["same_depth_cross_branch"] = (
    same_depth_cross_branch_results
)


# ============================================================
# 4. Cross-level distances
#
# n <-> n-1
# n <-> n-2
#
# Stop two levels before maximum depth, as agreed.
# ============================================================

cross_level_results = {
    "n_to_n_minus_1": {},
    "n_to_n_minus_2": {},
}

cross_level_start_depth = max_depth - 2

print(
    f"\nCross-level start depth: {cross_level_start_depth}"
)


# ------------------------------------------------------------
# n <-> n-1
# ------------------------------------------------------------

for depth in range(1, cross_level_start_depth + 1):

    rows_a = depth_to_rows[depth]
    rows_b = depth_to_rows[depth - 1]

    pairs = [
        (a, b)
        for a in rows_a
        for b in rows_b
    ]

    sampled_pairs, pair_type = sample_pairs(
        pairs,
        MAX_DEPTH_GROUP_PAIRS,
        rng,
    )

    distances = pairwise_distances(
        embeddings,
        sampled_pairs,
        manifold,
    )

    stats = distance_statistics(
        distances,
        pair_type,
    )

    cross_level_results["n_to_n_minus_1"][
        f"{depth}_to_{depth - 1}"
    ] = stats

    print_stats(
        f"4a. Depth {depth} <-> Depth {depth - 1}",
        stats,
    )


# ------------------------------------------------------------
# n <-> n-2
# ------------------------------------------------------------

for depth in range(2, cross_level_start_depth + 1):

    rows_a = depth_to_rows[depth]
    rows_b = depth_to_rows[depth - 2]

    pairs = [
        (a, b)
        for a in rows_a
        for b in rows_b
    ]

    sampled_pairs, pair_type = sample_pairs(
        pairs,
        MAX_DEPTH_GROUP_PAIRS,
        rng,
    )

    distances = pairwise_distances(
        embeddings,
        sampled_pairs,
        manifold,
    )

    stats = distance_statistics(
        distances,
        pair_type,
    )

    cross_level_results["n_to_n_minus_2"][
        f"{depth}_to_{depth - 2}"
    ] = stats

    print_stats(
        f"4b. Depth {depth} <-> Depth {depth - 2}",
        stats,
    )


results["cross_level"] = cross_level_results


# ============================================================
# 5. Random non-parent/child baseline
# ============================================================

num_nodes = len(node_mapping)

random_pairs = []

while len(random_pairs) < MAX_SAMPLED_PAIRS:

    a = rng.randrange(num_nodes)
    b = rng.randrange(num_nodes)

    if a == b:
        continue

    # Exclude direct parent-child relationships.
    if parent_of.get(a) == b:
        continue

    if parent_of.get(b) == a:
        continue

    random_pairs.append((a, b))


random_distances = pairwise_distances(
    embeddings,
    random_pairs,
    manifold,
)

stats = distance_statistics(
    random_distances,
    "sampled",
)

results["random_non_parent_child"] = stats

print_stats(
    "5. Random Non-Parent/Child",
    stats,
)


# ============================================================
# 6. Distance from Poincaré origin by depth
# ============================================================

origin = torch.zeros(
    (1, embeddings.shape[1]),
    dtype=embeddings.dtype,
    device=device,
)

origin_distance_results = {}

for depth in available_depths:

    rows = depth_to_rows[depth]

    points = embeddings[
        torch.tensor(
            rows,
            dtype=torch.long,
            device=device,
        )
    ]

    with torch.no_grad():

        origin_distances = manifold.dist(
            points,
            origin.expand_as(points),
        )

    origin_distances = (
        origin_distances
        .cpu()
        .numpy()
    )

    stats = distance_statistics(
        origin_distances,
        "full_population",
    )

    origin_distance_results[str(depth)] = stats

    print_stats(
        f"6. Distance from Poincare Origin — Depth {depth}",
        stats,
    )


results["origin_distance_by_depth"] = (
    origin_distance_results
)


# ============================================================
# Methodological notes
# ============================================================

results["methodological_notes"] = {
    "interpretation": (
        "Population statistics describe aggregate geometric "
        "behaviour. They do not by themselves establish that "
        "every local neighbourhood behaves similarly."
    ),
    "variation": (
        "Standard deviation and percentile statistics are "
        "included to assess the spread of each distance "
        "distribution."
    ),
    "sampling": (
        "Where the complete pair population exceeds the "
        "configured limit, a reproducible random sample is used. "
        "Such results are explicitly labelled as sampled."
    ),
    "same_depth_cross_branch": (
        "Pairs are at the same taxonomy depth but belong to "
        "different immediate-parent groups."
    ),
    "cross_level": (
        "Pairs are sampled across the specified depth strata "
        "without requiring the nodes to belong to the same branch."
    ),
    "training_validation_note": (
        "The Poincare representation was trained using the "
        "previous experiment's ancestor-descendant training pairs. "
        "This validation includes direct parent-child structure "
        "and other structural comparisons."
    ),
}


# ============================================================
# Save
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        results,
        f,
        indent=2,
    )


print("\n" + "=" * 70)
print("Validation complete.")
print(f"Results saved to:")
print(OUTPUT_FILE)
print("=" * 70)