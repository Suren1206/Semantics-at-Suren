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

OUTPUT_FILE = (
    OUTPUT_DIR / "experiment2_newsgroup_umap_three_views.png"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load data
# --------------------------------------------------

embeddings = np.load(EMBEDDINGS_FILE)

with open(DATA_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)

categories = [
    record["category"]
    for record in documents
]

print("=== INPUT ===")
print(f"Embeddings : {embeddings.shape}")
print(f"Documents  : {len(documents)}")


# --------------------------------------------------
# Reproduce the same UMAP
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
# Identify isolated region
# --------------------------------------------------

X_THRESHOLD = -7.0

isolated_indices = np.where(
    umap_result[:, 0] < X_THRESHOLD
)[0]

print("\n=== ISOLATED REGION ===")
print(f"X threshold     : {X_THRESHOLD}")
print(f"Documents found : {len(isolated_indices)}")


# --------------------------------------------------
# Category mapping
# --------------------------------------------------

unique_categories = sorted(set(categories))


# --------------------------------------------------
# Create three-view comparison
# --------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(21, 7),
)


# ==================================================
# VIEW 1 — ORIGINAL UMAP
# ==================================================

axes[0].scatter(
    umap_result[:, 0],
    umap_result[:, 1],
    s=8,
    alpha=0.6,
)

axes[0].set_title(
    "1. UMAP — Original"
)

axes[0].set_xlabel(
    "Dimension 1"
)

axes[0].set_ylabel(
    "Dimension 2"
)


# ==================================================
# VIEW 2 — ISOLATED REGION HIGHLIGHTED
# ==================================================

# Complete UMAP remains visible
axes[1].scatter(
    umap_result[:, 0],
    umap_result[:, 1],
    s=8,
    alpha=0.6,
)

# Isolated region overlaid
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
    "2. UMAP — Isolated Region"
)

axes[1].set_xlabel(
    "Dimension 1"
)

axes[1].set_ylabel(
    "Dimension 2"
)


# ==================================================
# VIEW 3 — CATEGORIZED UMAP
# ==================================================

for category in unique_categories:

    mask = np.array([
        item == category
        for item in categories
    ])

    axes[2].scatter(
        umap_result[mask, 0],
        umap_result[mask, 1],
        s=8,
        alpha=0.6,
        label=category,
    )

axes[2].set_title(
    "3. UMAP — Categories Revealed"
)

axes[2].set_xlabel(
    "Dimension 1"
)

axes[2].set_ylabel(
    "Dimension 2"
)

axes[2].legend(
    markerscale=2,
    fontsize=8,
)


# --------------------------------------------------
# Save figure
# --------------------------------------------------

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# Final status
# --------------------------------------------------

print("\n=== COMPLETE ===")

print(
    f"Three-view comparison saved to:"
)

print(
    OUTPUT_FILE
)