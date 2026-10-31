from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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
    OUTPUT_DIR / "experiment2_newsgroup_tsne_umap_final_comparison.png"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load frozen embeddings and metadata
# --------------------------------------------------

embeddings = np.load(EMBEDDINGS_FILE)

with open(DATA_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)

categories = np.array([
    record["category"]
    for record in documents
])

unique_categories = sorted(set(categories))


print("=== INPUT ===")
print(f"Embeddings : {embeddings.shape}")
print(f"Documents  : {len(documents)}")


# --------------------------------------------------
# t-SNE
# --------------------------------------------------

print("\nRunning t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=10,
    metric="cosine",
    random_state=42,
    init="pca",
    learning_rate="auto",
)

tsne_result = tsne.fit_transform(embeddings)

print(f"t-SNE result: {tsne_result.shape}")


# --------------------------------------------------
# UMAP
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
# Create 2 x 2 comparison
# --------------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(17, 14),
)


# ==================================================
# 1. t-SNE — ORIGINAL
# ==================================================

axes[0, 0].scatter(
    tsne_result[:, 0],
    tsne_result[:, 1],
    s=8,
    alpha=0.6,
)

axes[0, 0].set_title(
    "t-SNE — Original"
)

axes[0, 0].set_xlabel(
    "Dimension 1"
)

axes[0, 0].set_ylabel(
    "Dimension 2"
)


# ==================================================
# 2. t-SNE — CATEGORIES
# ==================================================

for category in unique_categories:

    mask = categories == category

    axes[0, 1].scatter(
        tsne_result[mask, 0],
        tsne_result[mask, 1],
        s=8,
        alpha=0.6,
        label=category,
    )

axes[0, 1].set_title(
    "t-SNE — Categories Revealed"
)

axes[0, 1].set_xlabel(
    "Dimension 1"
)

axes[0, 1].set_ylabel(
    "Dimension 2"
)

axes[0, 1].legend(
    markerscale=2,
    fontsize=8,
)


# ==================================================
# 3. UMAP — ORIGINAL
# ==================================================

axes[1, 0].scatter(
    umap_result[:, 0],
    umap_result[:, 1],
    s=8,
    alpha=0.6,
)

axes[1, 0].set_title(
    "UMAP — Original"
)

axes[1, 0].set_xlabel(
    "Dimension 1"
)

axes[1, 0].set_ylabel(
    "Dimension 2"
)


# ==================================================
# 4. UMAP — CATEGORIES
# ==================================================

for category in unique_categories:

    mask = categories == category

    axes[1, 1].scatter(
        umap_result[mask, 0],
        umap_result[mask, 1],
        s=8,
        alpha=0.6,
        label=category,
    )

axes[1, 1].set_title(
    "UMAP — Categories Revealed"
)

axes[1, 1].set_xlabel(
    "Dimension 1"
)

axes[1, 1].set_ylabel(
    "Dimension 2"
)

axes[1, 1].legend(
    markerscale=2,
    fontsize=8,
)


# --------------------------------------------------
# Overall title
# --------------------------------------------------

fig.suptitle(
    "20 Newsgroups — t-SNE vs UMAP",
    fontsize=16,
)


# --------------------------------------------------
# Save
# --------------------------------------------------

plt.tight_layout(
    rect=[0, 0, 1, 0.97]
)

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
    f"Output saved to:\n{OUTPUT_FILE}"
)