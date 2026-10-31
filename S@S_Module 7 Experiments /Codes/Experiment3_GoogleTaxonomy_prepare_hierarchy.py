from pathlib import Path
import json


# ============================================================
# Experiment 3 - Google Product Taxonomy
# Step 3: Prepare hierarchical parent-child structure
#
# Run from:
# ~/LangChain/Semantics@Suren/S@S_Module 7 Experiments
# ============================================================


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data"

INPUT_FILE = DATA_DIR / "google_product_taxonomy.txt"
OUTPUT_FILE = DATA_DIR / "experiment3_google_taxonomy_hierarchy.json"


# ------------------------------------------------------------
# Load taxonomy
# ------------------------------------------------------------

def load_taxonomy():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Taxonomy file not found:\n{INPUT_FILE}"
        )

    categories = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            category_id, path = line.split(" - ", 1)

            parts = [
                part.strip()
                for part in path.split(">")
            ]

            categories.append(
                {
                    "id": int(category_id),
                    "path": path,
                    "parts": parts,
                    "depth": len(parts) - 1,
                }
            )

    return categories


# ------------------------------------------------------------
# Build hierarchy
# ------------------------------------------------------------

def build_hierarchy(categories):

    path_to_id = {
        category["path"]: category["id"]
        for category in categories
    }

    nodes = []
    edges = []

    for category in categories:

        category_id = category["id"]
        path = category["path"]
        parts = category["parts"]

        node = {
            "id": category_id,
            "path": path,
            "name": parts[-1],
            "depth": category["depth"],
        }

        nodes.append(node)

        # Root categories have no parent.
        if category["depth"] == 0:
            continue

        parent_path = " > ".join(parts[:-1])

        parent_id = path_to_id.get(parent_path)

        if parent_id is None:
            raise ValueError(
                f"Parent not found for:\n"
                f"Child : {path}\n"
                f"Parent: {parent_path}"
            )

        edges.append(
            {
                "parent_id": parent_id,
                "child_id": category_id,
            }
        )

    return nodes, edges


# ------------------------------------------------------------
# Save prepared hierarchy
# ------------------------------------------------------------

def save_hierarchy(nodes, edges):

    output = {
        "source_file": INPUT_FILE.name,
        "taxonomy_version": "2021-09-21",
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("GOOGLE PRODUCT TAXONOMY - HIERARCHY PREPARATION")
    print("=" * 70)

    categories = load_taxonomy()

    print(f"\nLoaded categories: {len(categories):,}")

    nodes, edges = build_hierarchy(categories)

    print(f"Nodes: {len(nodes):,}")
    print(f"Parent-child edges: {len(edges):,}")

    save_hierarchy(nodes, edges)

    print("\nSaved:")
    print(f"  {OUTPUT_FILE}")

    print("\nValidation:")

    if len(edges) == len(nodes) - 21:
        print(
            "  ✓ Edge count is consistent with "
            "21 root categories."
        )
    else:
        print(
            "  ! Edge count does not match "
            "the expected root count."
        )

    print(
        "\nThe taxonomy has now been converted into "
        "an explicit node + parent-child representation."
    )

    print(
        "\nNo embeddings or geometric transformations "
        "have been performed."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()