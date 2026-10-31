from pathlib import Path
from urllib.request import Request, urlopen


# ============================================================
# Experiment 3 - Google Product Taxonomy
# Step 1: Download the official taxonomy file
#
# Run this script from:
# ~/LangChain/Semantics@Suren/S@S_Module 7 Experiments
# ============================================================


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Google Product Taxonomy download URL
# ------------------------------------------------------------

TAXONOMY_URL = (
    "https://www.google.com/basepages/producttype/taxonomy-with-ids.en-US.txt"
)

OUTPUT_FILE = DATA_DIR / "google_product_taxonomy.txt"


# ------------------------------------------------------------
# Download
# ------------------------------------------------------------

def download_taxonomy():
    print("=" * 70)
    print("Google Product Taxonomy - Download")
    print("=" * 70)

    print(f"\nProject directory:")
    print(f"  {PROJECT_DIR}")

    print(f"\nOutput file:")
    print(f"  {OUTPUT_FILE}")

    print("\nDownloading official Google Product Taxonomy...")

    request = Request(
        TAXONOMY_URL,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    try:
        with urlopen(request, timeout=60) as response:
            content = response.read()

        OUTPUT_FILE.write_bytes(content)

    except Exception as exc:
        print("\nERROR: Download failed.")
        print(f"Reason: {exc}")
        raise SystemExit(1)

    print("\nDownload completed successfully.")

    print(f"\nFile size:")
    print(f"  {OUTPUT_FILE.stat().st_size:,} bytes")

    print("\nSaved to:")
    print(f"  {OUTPUT_FILE}")

    print("\nNo preprocessing has been performed.")
    print("The downloaded taxonomy is being preserved as the raw source.")

    print("\nNext step:")
    print("  Inspect the actual contents before designing the experiment.")

    print("=" * 70)


if __name__ == "__main__":
    download_taxonomy()