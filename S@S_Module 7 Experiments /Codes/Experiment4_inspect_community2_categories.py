from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CATEGORY_FILE = (
    PROJECT_ROOT
    / "Data"
    / "experiment4"
    / "amazon_product_categories.txt"
)


TARGET_IDS = {
    10001,
    27638,
    51569,
    88226,
    116422,
    126227,
    159947,
    162938,
    184977,
    188045,
    191044,
    246142,
    265214,
    290507,
    296858,   # mystery product
    300258,
    341525,
    348922,
    359832,
    365744,
    382502,
    390538,
    410857,
    433453,
    479170,
    489980,
    540746,
}


def main():

    print("=" * 70)
    print("COMMUNITY 2 — PRODUCT CATEGORY INVESTIGATION")
    print("=" * 70)

    print(f"\nCategory file:")
    print(f"  {CATEGORY_FILE}")

    if not CATEGORY_FILE.exists():
        print("\nERROR: Category file not found.")
        return

    # -----------------------------------------------------
    # First inspect the actual file format
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FIRST 20 RAW LINES")
    print("=" * 70)

    with open(CATEGORY_FILE, "r", encoding="utf-8", errors="replace") as f:

        shown = 0

        for line in f:

            line = line.rstrip("\n")

            if not line.strip():
                continue

            print(line)

            shown += 1

            if shown >= 20:
                break

    # -----------------------------------------------------
    # Extract target IDs
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("TARGET PRODUCT CATEGORIES")
    print("=" * 70)

    found = {}

    with open(CATEGORY_FILE, "r", encoding="utf-8", errors="replace") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split(None, 1)

            if len(parts) < 2:
                continue

            try:
                node_id = int(parts[0])
            except ValueError:
                continue

            if node_id in TARGET_IDS:
                found[node_id] = parts[1]

    print(f"\nTarget IDs : {len(TARGET_IDS)}")
    print(f"Found IDs  : {len(found)}")
    print(f"Missing IDs: {len(TARGET_IDS - found.keys())}")

    if TARGET_IDS - found.keys():
        print("\nMissing:")
        for node_id in sorted(TARGET_IDS - found.keys()):
            print(f"  {node_id}")

    print("\n" + "-" * 70)

    for node_id in sorted(TARGET_IDS):

        if node_id in found:

            marker = "  <--- MYSTERY PRODUCT" if node_id == 296858 else ""

            print(
                f"{node_id:>8} : "
                f"{found[node_id]}"
                f"{marker}"
            )

    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()