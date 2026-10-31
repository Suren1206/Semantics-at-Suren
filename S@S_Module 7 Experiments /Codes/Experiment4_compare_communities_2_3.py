from pathlib import Path
import gzip


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMMUNITY_FILE = (
    PROJECT_ROOT
    / "Data"
    / "experiment4"
    / "com-amazon.all.dedup.cmty.txt.gz"
)


def main():

    communities = []

    with gzip.open(COMMUNITY_FILE, "rt", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            nodes = [int(x) for x in line.split()]
            communities.append(set(nodes))

            if len(communities) >= 3:
                break

    community_2 = communities[1]
    community_3 = communities[2]

    only_2 = sorted(community_2 - community_3)
    only_3 = sorted(community_3 - community_2)
    common = sorted(community_2 & community_3)

    print("=" * 70)
    print("COMMUNITY 2 vs COMMUNITY 3")
    print("=" * 70)

    print(f"\nCommunity 2 size: {len(community_2)}")
    print(f"Community 3 size: {len(community_3)}")

    print("\nCommon nodes:")
    print(f"  {len(common)}")

    print("\nNodes ONLY in Community 2:")
    print(f"  {only_2}")

    print("\nNodes ONLY in Community 3:")
    print(f"  {only_3}")

    print("\nCommon nodes:")
    print(common)

    print("\nConclusion:")

    if len(only_2) == 1 and len(only_3) == 0:
        print(
            "  Community 3 is Community 2 with exactly one node removed."
        )

    elif len(only_2) == 0 and len(only_3) == 1:
        print(
            "  Community 3 is Community 2 with exactly one node added."
        )

    else:
        print(
            "  The two communities differ by more than one node."
        )


if __name__ == "__main__":
    main()