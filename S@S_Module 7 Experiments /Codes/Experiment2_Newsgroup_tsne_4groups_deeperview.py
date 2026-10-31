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
    OUTPUT_DIR / "experiment2_tSNE_4groups_deeperview_REAL.png"
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

print("\nCategories:")
for category in unique_categories:
    print(
        f"  {category}: "
        f"{np.sum(categories == category)}"
    )


# --------------------------------------------------
# Run EXACT SAME t-SNE
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
# Create four real category-focused views
# --------------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(17, 14),
)

axes = axes.ravel()


for ax, category in zip(
    axes,
    unique_categories
):

    # ----------------------------------------------
    # All other categories in grey
    # ----------------------------------------------

    other_mask = categories != category

    ax.scatter(
        tsne_result[other_mask, 0],
        tsne_result[other_mask, 1],
        s=8,
        alpha=0.25,
        label="Other categories",
    )


    # ----------------------------------------------
    # Current category highlighted
    # ----------------------------------------------

    category_mask = categories == category

    ax.scatter(
        tsne_result[category_mask, 0],
        tsne_result[category_mask, 1],
        s=12,
        alpha=0.85,
        label=category,
    )


    # ----------------------------------------------
    # Labels
    # ----------------------------------------------

    ax.set_title(
        f"{category} — In Focus"
    )

    ax.set_xlabel(
        "Dimension 1"
    )

    ax.set_ylabel(
        "Dimension 2"
    )

    ax.legend(
        markerscale=2,
        fontsize=8,
    )


# --------------------------------------------------
# Overall title
# --------------------------------------------------

fig.suptitle(
    "20 Newsgroups — t-SNE: Each Category in Focus",
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
    f"Real data-derived plot saved to:\n"
    f"{OUTPUT_FILE}"
)