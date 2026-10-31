from pathlib import Path

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
OUTPUT_FILE = OUTPUT_DIR / "experiment2_newsgroup_tsne_umap.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load embeddings
# --------------------------------------------------

embeddings = np.load(EMBEDDINGS_FILE)

print("=== INPUT ===")
print(f"Embedding shape: {embeddings.shape}")


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

print(f"t-SNE shape: {tsne_result.shape}")


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

print(f"UMAP shape: {umap_result.shape}")


# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].scatter(
    tsne_result[:, 0],
    tsne_result[:, 1],
    s=8,
    alpha=0.6,
)

axes[0].set_title("t-SNE — Unlabeled")
axes[0].set_xlabel("Dimension 1")
axes[0].set_ylabel("Dimension 2")


axes[1].scatter(
    umap_result[:, 0],
    umap_result[:, 1],
    s=8,
    alpha=0.6,
)

axes[1].set_title("UMAP — Unlabeled")
axes[1].set_xlabel("Dimension 1")
axes[1].set_ylabel("Dimension 2")


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