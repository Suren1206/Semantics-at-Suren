from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# Experiment 3 - Google Product Taxonomy
# Step 2: Inspect the actual hierarchy
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


# ------------------------------------------------------------
# Load taxonomy
# ------------------------------------------------------------

def load_taxonomy():
    if not INPUT_FILE.exists():
        print(f"ERROR: File not found:\n  {INPUT_FILE}")
        raise SystemExit(1)

    categories = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            # Skip blank lines
            if not line:
                continue

            # Skip Google metadata/comment lines
            if line.startswith("#"):
                continue

            # Expected format:
            # category_id - category path
            try:
                category_id, path = line.split(" - ", 1)
            except ValueError:
                print(f"WARNING: Could not parse line:\n{line}")
                continue

            parts = [part.strip() for part in path.split(">")]

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
# Analyse hierarchy
# ------------------------------------------------------------

def analyse_hierarchy(categories):

    print("\n" + "=" * 70)
    print("GOOGLE PRODUCT TAXONOMY - HIERARCHY INSPECTION")
    print("=" * 70)

    print(f"\nTaxonomy file:")
    print(f"  {INPUT_FILE}")

    print(f"\nTotal taxonomy categories:")
    print(f"  {len(categories):,}")


    # --------------------------------------------------------
    # Depth distribution
    # --------------------------------------------------------

    depth_counts = Counter(category["depth"] for category in categories)

    print("\n" + "-" * 70)
    print("1. CATEGORY DEPTH DISTRIBUTION")
    print("-" * 70)

    print("\nDepth 0 = root category")

    for depth in sorted(depth_counts):
        print(
            f"Depth {depth}: "
            f"{depth_counts[depth]:,} categories"
        )


    # --------------------------------------------------------
    # Root categories
    # --------------------------------------------------------

    roots = [
        category
        for category in categories
        if category["depth"] == 0
    ]

    print("\n" + "-" * 70)
    print("2. ROOT CATEGORIES")
    print("-" * 70)

    print(f"\nNumber of root categories: {len(roots)}")

    for category in roots:
        print(
            f"  {category['id']} - "
            f"{category['path']}"
        )


    # --------------------------------------------------------
    # Maximum depth
    # --------------------------------------------------------

    max_depth = max(category["depth"] for category in categories)

    deepest = [
        category
        for category in categories
        if category["depth"] == max_depth
    ]

    print("\n" + "-" * 70)
    print("3. MAXIMUM DEPTH")
    print("-" * 70)

    print(f"\nMaximum depth: {max_depth}")

    print(
        f"Number of categories at maximum depth: "
        f"{len(deepest):,}"
    )

    print("\nExamples of deepest categories:")

    for category in deepest[:10]:
        print(
            f"  {category['id']} - "
            f"{category['path']}"
        )


    # --------------------------------------------------------
    # Parent -> children relationships
    # --------------------------------------------------------

    children = defaultdict(list)

    for category in categories:

        if category["depth"] == 0:
            continue

        parts = category["parts"]

        parent_path = " > ".join(parts[:-1])

        children[parent_path].append(category["path"])


    # --------------------------------------------------------
    # Branching statistics
    # --------------------------------------------------------

    child_counts = {
        parent: len(child_list)
        for parent, child_list in children.items()
    }

    print("\n" + "-" * 70)
    print("4. BRANCHING STRUCTURE")
    print("-" * 70)

    print(
        f"\nNumber of parent categories with children: "
        f"{len(child_counts):,}"
    )

    if child_counts:

        print(
            f"Minimum children for a parent: "
            f"{min(child_counts.values())}"
        )

        print(
            f"Maximum children for a parent: "
            f"{max(child_counts.values())}"
        )

        average_children = (
            sum(child_counts.values())
            / len(child_counts)
        )

        print(
            f"Average children per parent: "
            f"{average_children:.2f}"
        )


    # --------------------------------------------------------
    # Parents with the largest number of children
    # --------------------------------------------------------

    print("\nParents with the largest number of direct children:")

    largest_parents = sorted(
        child_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )[:10]

    for parent, count in largest_parents:
        print(f"  {count:4d} children -> {parent}")


    # --------------------------------------------------------
    # Example hierarchy paths
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("5. EXAMPLE HIERARCHY PATHS")
    print("-" * 70)

    examples = [
        category
        for category in categories
        if category["depth"] >= 4
    ][:10]

    for category in examples:
        print(
            f"\nDepth {category['depth']}:"
        )
        print(
            f"  {category['path']}"
        )


    # --------------------------------------------------------
    # Structural consistency checks
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("6. STRUCTURAL CHECKS")
    print("-" * 70)

    category_paths = {
        category["path"]
        for category in categories
    }

    missing_parents = []

    for category in categories:

        if category["depth"] == 0:
            continue

        parts = category["parts"]
        parent_path = " > ".join(parts[:-1])

        if parent_path not in category_paths:
            missing_parents.append(
                (category["path"], parent_path)
            )

    duplicate_ids = (
        len(categories)
        - len({category["id"] for category in categories})
    )

    duplicate_paths = (
        len(categories)
        - len(category_paths)
    )

    print(
        f"\nDuplicate category IDs: "
        f"{duplicate_ids}"
    )

    print(
        f"Duplicate category paths: "
        f"{duplicate_paths}"
    )

    print(
        f"Categories whose parent path is missing: "
        f"{len(missing_parents)}"
    )

    if missing_parents:

        print("\nExamples of missing parents:")

        for child, parent in missing_parents[:10]:
            print(f"  Child : {child}")
            print(f"  Parent: {parent}")


    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)

    print(
        "\nThe raw taxonomy has been inspected without "
        "modifying or transforming it."
    )

    print(
        "\nNo embeddings or geometric representations "
        "have been created."
    )

    print(
        "\nNext step: interpret the observed hierarchy "
        "before designing the hyperbolic experiment."
    )

    print("=" * 70)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":

    taxonomy = load_taxonomy()

    analyse_hierarchy(taxonomy)