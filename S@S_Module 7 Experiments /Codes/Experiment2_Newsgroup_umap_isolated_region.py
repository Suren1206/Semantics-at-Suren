from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt
import umap


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "Outputs"

EMBEDDINGS_FILE = DATA_DIR / "experiment2_newsgroup_embeddings.npy"
DATA_FILE = DATA_DIR / "experiment2_newsgroups.json"

OUTPUT_PLOT = (
    OUTPUT_DIR / "experiment2_newsgroup_umap_original_vs_isolated.png"
)

OUTPUT_JSON = (
    DATA_DIR / "experiment2_newsgroup_umap_isolated_region.json"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load data
# --------------------------------------------------

embeddings = np.load(EMBEDDINGS_FILE)

with open(DATA_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)

print("=== INPUT ===")
print(f"Embeddings : {embeddings.shape}")
print(f"Documents  : {len(documents)}")


# --------------------------------------------------
# Reproduce the original UMAP
# --------------------------------------------------

print("\nRunning UMAP...")

umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=10,
    metric="cosine",
    random_state=42,
)

umap_result = umap_model.fit_transform(embeddings)

print(f"UMAP result: {umap_result.shape}")


# --------------------------------------------------
# Define isolated region
# --------------------------------------------------

X_THRESHOLD = -7.0

isolated_indices = np.where(
    umap_result[:, 0] < X_THRESHOLD
)[0]

print("\n=== ISOLATED REGION ===")
print(f"X threshold       : {X_THRESHOLD}")
print(f"Documents found   : {len(isolated_indices)}")


# --------------------------------------------------
# Extract documents from isolated region
# --------------------------------------------------

isolated_documents = []

for idx in isolated_indices:

    record = {
        "embedding_index": int(idx),
        "document_id": documents[idx]["document_id"],
        "category_id": documents[idx]["category_id"],
        "category": documents[idx]["category"],
        "umap_x": float(umap_result[idx, 0]),
        "umap_y": float(umap_result[idx, 1]),
        "text": documents[idx]["text"],
    }

    isolated_documents.append(record)


# Sort by UMAP X position
isolated_documents.sort(
    key=lambda x: x["umap_x"]
)


# --------------------------------------------------
# Print extracted documents
# --------------------------------------------------

print("\nDocuments in isolated region:\n")

for record in isolated_documents:

    print("-" * 80)

    print(
        f"Document ID : {record['document_id']}"
    )

    print(
        f"Category    : {record['category']}"
    )

    print(
        f"UMAP        : "
        f"({record['umap_x']:.3f}, "
        f"{record['umap_y']:.3f})"
    )

    text = (
        record["text"]
        .replace("\n", " ")
        .strip()
    )

    if len(text) > 500:
        text = text[:500] + "..."

    print(f"Text        : {text}")


# --------------------------------------------------
# Save extracted documents
# --------------------------------------------------

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        isolated_documents,
        f,
        indent=2,
        ensure_ascii=False,
    )


# --------------------------------------------------
# Create comparison figure
# --------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 7),
)


# --------------------------------------------------
# LEFT — Original UMAP
# --------------------------------------------------

axes[0].scatter(
    umap_result[:, 0],
    umap_result[:, 1],
    s=8,
    alpha=0.6,
)

axes[0].set_title(
    "UMAP — Original"
)

axes[0].set_xlabel(
    "Dimension 1"
)

axes[0].set_ylabel(
    "Dimension 2"
)


# --------------------------------------------------
# RIGHT — Same UMAP + isolated region highlighted
# --------------------------------------------------

# First draw the complete original UMAP
axes[1].scatter(
    umap_result[:, 0],
    umap_result[:, 1],
    s=8,
    alpha=0.6,
)

# Then overlay only the isolated points
axes[1].scatter(
    umap_result[isolated_indices, 0],
    umap_result[isolated_indices, 1],
    s=22,
    alpha=1.0,
)

axes[1].axvline(
    X_THRESHOLD,
    linestyle="--",
    linewidth=1,
)

axes[1].set_title(
    "UMAP — Isolated Region Highlighted"
)

axes[1].set_xlabel(
    "Dimension 1"
)

axes[1].set_ylabel(
    "Dimension 2"
)


# --------------------------------------------------
# Save figure
# --------------------------------------------------

plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

# --------------------------------------------------
# Final status
# --------------------------------------------------

print("\n=== COMPLETE ===")

print(
    f"Comparison plot : {OUTPUT_PLOT}"
)

print(
    f"Extracted data  : {OUTPUT_JSON}"
)