from pathlib import Path
import gzip
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRAPH_FILE = (
    PROJECT_ROOT
    / "Data"
    / "experiment4"
    / "com-amazon.ungraph.txt.gz"
)

COMMUNITY_FILE = (
    PROJECT_ROOT
    / "Data"
    / "experiment4"
    / "com-amazon.all.dedup.cmty.txt.gz"
)

MYSTERY_NODE = 296858


def load_community_2():

    communities = []

    with gzip.open(COMMUNITY_FILE, "rt", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            communities.append(set(map(int, line.split())))

            if len(communities) == 3:
                break

    return communities[1], communities[2]


def load_graph_neighbors():

    neighbors = {}

    with gzip.open(GRAPH_FILE, "rt", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            u, v = map(int, line.split()[:2])

            neighbors.setdefault(u, set()).add(v)
            neighbors.setdefault(v, set()).add(u)

    return neighbors


def main():

    print("=" * 70)
    print("INVESTIGATION OF NODE 296858")
    print("=" * 70)

    community_2, community_3 = load_community_2()

    common = community_2 & community_3

    mystery_only = community_2 - community_3

    print("\nCommunity 2 size:", len(community_2))
    print("Community 3 size:", len(community_3))
    print("Common members:", len(common))
    print("Community-2-only:", mystery_only)

    print("\nLoading graph...")

    neighbors = load_graph_neighbors()

    mystery_neighbors = neighbors.get(MYSTERY_NODE, set())

    print("\nNode 296858:")
    print("  Total graph neighbours:", len(mystery_neighbors))

    internal_neighbors = mystery_neighbors & community_2

    print(
        "  Neighbours inside Community 2:",
        len(internal_neighbors)
    )

    print(
        "  Neighbours outside Community 2:",
        len(mystery_neighbors - community_2)
    )

    if mystery_neighbors:

        internal_ratio = (
            len(internal_neighbors)
            / len(mystery_neighbors)
        )

        print(
            f"  Internal-neighbour ratio: "
            f"{internal_ratio:.4f}"
        )

    print("\nNeighbours of 296858 that belong to Community 2:")

    for node in sorted(internal_neighbors):

        marker = " [COMMON MEMBER]" if node in common else ""

        print(f"  {node}{marker}")

    # -----------------------------------------------------
    # Compare every Community-2 member
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("COMMUNITY 2 — STRUCTURAL COMPARISON")
    print("=" * 70)

    rows = []

    for node in sorted(community_2):

        node_neighbors = neighbors.get(node, set())

        internal = node_neighbors & community_2

        rows.append(
            (
                node,
                len(node_neighbors),
                len(internal),
                (
                    len(internal) / len(node_neighbors)
                    if node_neighbors
                    else 0.0
                )
            )
        )

    rows.sort(key=lambda x: x[3], reverse=True)

    print(
        "\n"
        "Node       Total degree   Internal links   Internal ratio"
    )

    print("-" * 65)

    for node, degree, internal, ratio in rows:

        marker = "  <--- 296858" if node == MYSTERY_NODE else ""

        print(
            f"{node:8d}   "
            f"{degree:12d}   "
            f"{internal:14d}   "
            f"{ratio:14.4f}"
            f"{marker}"
        )

    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()