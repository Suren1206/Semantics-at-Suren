from pathlib import Path
import json
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


# ============================================================
# Configuration
# ============================================================

CURVATURE = 1.0


# ============================================================
# Load representation
# ============================================================

embeddings_np = np.load(EMBEDDING_FILE)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

node_mapping = metadata["node_mapping"]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

embeddings = torch.tensor(
    embeddings_np,
    dtype=torch.float32,
    device=device,
)

manifold = geoopt.PoincareBall(c=CURVATURE)


# ============================================================
# Build node structures
# ============================================================

row_to_node = {
    item["row"]: item
    for item in node_mapping
}

path_to_row = {
    item["category_path"]: item["row"]
    for item in node_mapping
}


# ============================================================
# Parent relationships
# ============================================================

parent_of = {}

for row, node in row_to_node.items():

    path = node["category_path"]

    if " > " not in path:
        continue

    parent_path = path.rsplit(" > ", 1)[0]

    if parent_path in path_to_row:
        parent_of[row] = path_to_row[parent_path]


# ============================================================
# Children relationships
# ============================================================

children_of = {}

for child, parent in parent_of.items():

    children_of.setdefault(parent, []).append(child)


# ============================================================
# Distance helper
# ============================================================

def poincare_distance(row_a, row_b):

    a = embeddings[row_a].unsqueeze(0)
    b = embeddings[row_b].unsqueeze(0)

    with torch.no_grad():
        distance = manifold.dist(a, b).item()

    return distance


# ============================================================
# Path reconstruction
# ============================================================

def get_path_to_root(row):

    path = []

    current = row

    while True:

        path.append(current)

        if current not in parent_of:
            break

        current = parent_of[current]

    path.reverse()

    return path


def print_node(row):

    node = row_to_node[row]

    print(
        f"Depth {node['depth']}: "
        f"{node['category_name']}"
    )


# ============================================================
# 1. SHALLOW HIERARCHY
#
# Root -> Level 1 -> Level 2
# ============================================================

print("=" * 80)
print("1. SHALLOW HIERARCHY")
print("=" * 80)

shallow_candidates = []

for row, node in row_to_node.items():

    if node["depth"] != 2:
        continue

    path_rows = get_path_to_root(row)

    if len(path_rows) == 3:
        shallow_candidates.append(row)

# Deterministic choice.
shallow_row = sorted(
    shallow_candidates,
    key=lambda r: row_to_node[r]["category_path"]
)[0]

shallow_path = get_path_to_root(shallow_row)

print("\nSelected path:\n")

for row in shallow_path:
    print_node(row)

print("\nConsecutive Poincaré distances:\n")

for i in range(len(shallow_path) - 1):

    parent = shallow_path[i]
    child = shallow_path[i + 1]

    print(
        f"{row_to_node[parent]['category_name']}"
        f" -> "
        f"{row_to_node[child]['category_name']}"
        f" : "
        f"{poincare_distance(parent, child):.6f}"
    )


# ============================================================
# 2. DEEP HIERARCHY
#
# Select a deepest available node and trace it to root.
# ============================================================

print("\n" + "=" * 80)
print("2. DEEP HIERARCHY")
print("=" * 80)

max_depth = max(
    node["depth"]
    for node in node_mapping
)

deep_candidates = [
    row
    for row, node in row_to_node.items()
    if node["depth"] == max_depth
]

# Deterministic deepest path.
deep_row = sorted(
    deep_candidates,
    key=lambda r: row_to_node[r]["category_path"]
)[0]

deep_path = get_path_to_root(deep_row)

print(
    f"\nSelected deepest path "
    f"(depth {max_depth}):\n"
)

for row in deep_path:
    print_node(row)

print("\nConsecutive Poincaré distances:\n")

for i in range(len(deep_path) - 1):

    parent = deep_path[i]
    child = deep_path[i + 1]

    print(
        f"Depth {row_to_node[parent]['depth']}"
        f" -> "
        f"Depth {row_to_node[child]['depth']}"
        f" | "
        f"{row_to_node[parent]['category_name']}"
        f" -> "
        f"{row_to_node[child]['category_name']}"
        f" : "
        f"{poincare_distance(parent, child):.6f}"
    )


# ============================================================
# 3. BRANCHING HIERARCHY
#
# Find a parent with several children where at least one child
# itself has descendants.
#
# Prefer a moderate branch size so the output remains readable.
# ============================================================

print("\n" + "=" * 80)
print("3. BRANCHING HIERARCHY")
print("=" * 80)

branch_candidates = []

for parent, children in children_of.items():

    child_count = len(children)

    if child_count < 3 or child_count > 8:
        continue

    has_descendant = any(
        child in children_of
        for child in children
    )

    if not has_descendant:
        continue

    branch_candidates.append(
        (parent, children)
    )


# Deterministic selection.
branch_candidates.sort(
    key=lambda item: (
        -len(item[1]),
        row_to_node[item[0]]["category_path"],
    )
)

branch_parent, branch_children = (
    branch_candidates[0]
)

branch_children = sorted(
    branch_children,
    key=lambda r: row_to_node[r]["category_name"]
)

print("\nSelected parent:\n")
print_node(branch_parent)

print("\nImmediate children:\n")

for child in branch_children:

    print(
        f"  {row_to_node[child]['category_name']}"
        f" | parent distance = "
        f"{poincare_distance(branch_parent, child):.6f}"
    )


# ============================================================
# Sibling distances within the selected branch
# ============================================================

print("\nSibling-to-sibling distances:\n")

for i in range(len(branch_children)):

    for j in range(i + 1, len(branch_children)):

        a = branch_children[i]
        b = branch_children[j]

        print(
            f"{row_to_node[a]['category_name']}"
            f" <-> "
            f"{row_to_node[b]['category_name']}"
            f" : "
            f"{poincare_distance(a, b):.6f}"
        )


# ============================================================
# One branch continuation
#
# Pick the first child that has descendants and show one
# additional level.
# ============================================================

continuing_child = None

for child in branch_children:

    if child in children_of:

        continuing_child = child
        break


if continuing_child is not None:

    grandchildren = sorted(
        children_of[continuing_child],
        key=lambda r: row_to_node[r]["category_name"]
    )

    print(
        "\nOne branch continuation:\n"
    )

    print(
        f"{row_to_node[branch_parent]['category_name']}"
        f" -> "
        f"{row_to_node[continuing_child]['category_name']}"
    )

    for grandchild in grandchildren:

        print(
            f"  -> "
            f"{row_to_node[grandchild]['category_name']}"
            f" | distance from parent = "
            f"{poincare_distance(continuing_child, grandchild):.6f}"
        )


# ============================================================
# End
# ============================================================

print("\n" + "=" * 80)
print("Concrete path inspection complete.")
print("=" * 80)