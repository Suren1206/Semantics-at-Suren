from pathlib import Path
import gzip


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "Data" / "experiment4"

GRAPH_FILE = DATA_DIR / "com-amazon.ungraph.txt.gz"
ALL_COMMUNITIES_FILE = DATA_DIR / "com-amazon.all.dedup.cmty.txt.gz"
TOP5000_FILE = DATA_DIR / "com-amazon.top5000.cmty.txt.gz"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

GRAPH_SAMPLE_SIZE = 20
COMMUNITY_SAMPLE_SIZE = 10


# ---------------------------------------------------------
# Graph inspection
# ---------------------------------------------------------

def inspect_graph_file(path):

    print("\n" + "=" * 70)
    print("1. GRAPH FILE — SAMPLE DATA")
    print("=" * 70)

    print(f"File: {path.name}")

    node_ids = set()
    edge_count = 0
    self_loops = 0

    sample_edges = []

    with gzip.open(path, "rt", encoding="utf-8") as f:

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

            edge_count += 1

            node_ids.add(u)
            node_ids.add(v)

            if u == v:
                self_loops += 1

            if len(sample_edges) < GRAPH_SAMPLE_SIZE:
                sample_edges.append((u, v))

    print("\nFirst graph edges:")

    for i, (u, v) in enumerate(sample_edges, start=1):
        print(f"  {i:2d}. Product {u}  <-->  Product {v}")

    print("\nGraph totals:")
    print(f"  Unique nodes : {len(node_ids):,}")
    print(f"  Edges        : {edge_count:,}")
    print(f"  Self-loops   : {self_loops:,}")

    return node_ids


# ---------------------------------------------------------
# Community inspection
# ---------------------------------------------------------

def inspect_community_file(path, title, graph_nodes):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(f"File: {path.name}")

    community_count = 0
    community_sizes = []

    sample_communities = []

    community_nodes_seen = set()

    with gzip.open(path, "rt", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            nodes = [int(x) for x in line.split()]

            if not nodes:
                continue

            community_count += 1
            community_sizes.append(len(nodes))

            community_nodes_seen.update(nodes)

            if len(sample_communities) < COMMUNITY_SAMPLE_SIZE:
                sample_communities.append(nodes)

    print("\nSample communities:")

    for i, nodes in enumerate(sample_communities, start=1):

        preview = nodes[:20]

        suffix = ""

        if len(nodes) > 20:
            suffix = f" ... (+{len(nodes) - 20} more)"

        print(
            f"  Community {i:2d} "
            f"(size={len(nodes):,}): "
            f"{preview}{suffix}"
        )

    print("\nCommunity totals:")
    print(f"  Communities : {community_count:,}")

    if community_sizes:

        print(f"  Minimum size : {min(community_sizes):,}")
        print(f"  Maximum size : {max(community_sizes):,}")
        print(
            f"  Mean size    : "
            f"{sum(community_sizes) / len(community_sizes):.2f}"
        )

        sorted_sizes = sorted(community_sizes)

        mid = len(sorted_sizes) // 2

        if len(sorted_sizes) % 2 == 0:
            median = (
                sorted_sizes[mid - 1]
                + sorted_sizes[mid]
            ) / 2
        else:
            median = sorted_sizes[mid]

        print(f"  Median size  : {median:,}")

    # -----------------------------------------------------
    # Compare community nodes against graph nodes
    # -----------------------------------------------------

    overlap = community_nodes_seen.intersection(graph_nodes)

    print("\nRelationship to graph:")
    print(f"  Unique community nodes : {len(community_nodes_seen):,}")
    print(f"  Present in graph       : {len(overlap):,}")
    print(
        f"  Not present in graph   : "
        f"{len(community_nodes_seen - graph_nodes):,}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("EXPERIMENT 4 — AMAZON GRAPH DATA INSPECTION")
    print("=" * 70)

    print(f"\nProject root:")
    print(f"  {PROJECT_ROOT}")

    print("\nData directory:")
    print(f"  {DATA_DIR}")

    # -----------------------------------------------------
    # File existence
    # -----------------------------------------------------

    required_files = [
        GRAPH_FILE,
        ALL_COMMUNITIES_FILE,
        TOP5000_FILE,
    ]

    print("\nChecking files:")

    missing = []

    for path in required_files:

        if path.exists():
            print(f"  FOUND   : {path.name}")
        else:
            print(f"  MISSING : {path.name}")
            missing.append(path)

    if missing:

        print("\nERROR: Required file(s) missing.")
        return

    # -----------------------------------------------------
    # Inspect graph
    # -----------------------------------------------------

    graph_nodes = inspect_graph_file(GRAPH_FILE)

    # -----------------------------------------------------
    # Inspect all communities
    # -----------------------------------------------------

    inspect_community_file(
        ALL_COMMUNITIES_FILE,
        "2. ALL COMMUNITIES — SAMPLE DATA",
        graph_nodes
    )

    # -----------------------------------------------------
    # Inspect top 5000
    # -----------------------------------------------------

    inspect_community_file(
        TOP5000_FILE,
        "3. TOP 5000 COMMUNITIES — SAMPLE DATA",
        graph_nodes
    )

    # -----------------------------------------------------
    # Final
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("RAW CONTENT INSPECTION COMPLETE")
    print("=" * 70)

    print("\nNo embedding or training was performed.")
    print("Community files remain validation/reference data.")


if __name__ == "__main__":
    main()