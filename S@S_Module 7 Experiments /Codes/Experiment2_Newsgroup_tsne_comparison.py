from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "Outputs"

EMBEDDINGS_FILE = DATA_DIR / "experiment2_newsgroup_embeddings.npy"
DATA_FILE = DATA_DIR / "experiment2_newsgroups.json"

OUTPUT_FILE = (
    OUTPUT_DIR / "experiment2_newsgroup_tsne_comparison.png"
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
# Reproduce the original t-SNE
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
# Categories
# --------------------------------------------------

unique_categories = sorted(set(categories))


# --------------------------------------------------
# Create comparison
# --------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 7),
)


# ==================================================
# VIEW 1 — ORIGINAL t-SNE
# ==================================================

axes[0].scatter(
    tsne_result[:, 0],
    tsne_result[:, 1],
    s=8,
    alpha=0.6,
)

axes[0].set_title(
    "1. t-SNE — Original"
)

axes[0].set_xlabel(
    "Dimension 1"
)

axes[0].set_ylabel(
    "Dimension 2"
)


# ==================================================
# VIEW 2 — CATEGORIES REVEALED
# ==================================================

for category in unique_categories:

    mask = np.array([
        item == category
        for item in categories
    ])

    axes[1].scatter(
        tsne_result[mask, 0],
        tsne_result[mask, 1],
        s=8,
        alpha=0.6,
        label=category,
    )

axes[1].set_title(
    "2. t-SNE — Categories Revealed"
)

axes[1].set_xlabel(
    "Dimension 1"
)

axes[1].set_ylabel(
    "Dimension 2"
)

axes[1].legend(
    markerscale=2,
    fontsize=8,
)


# --------------------------------------------------
# Save
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
print(f"Output saved to: {OUTPUT_FILE}")