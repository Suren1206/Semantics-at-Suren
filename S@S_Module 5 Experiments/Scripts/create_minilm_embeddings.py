from pathlib import Path
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "Inputs"
OUTPUT_DIR = BASE_DIR / "Outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# B01
# ------------------------------------------------------------

B01_DIR = INPUT_DIR / "20260424_B01"

B01_METADATA_FILE = B01_DIR / "chunk_id_to_metadata.json"

B01_OUTPUT_FILE = (
    OUTPUT_DIR / "B01_all-MiniLM-L6-v2_embeddings.npy"
)


# ------------------------------------------------------------
# SYNTHETIC
# ------------------------------------------------------------

SYNTHETIC_CORPUS_FILE = INPUT_DIR / "Synthetic_Corpus.csv"

SYNTHETIC_OUTPUT_FILE = (
    OUTPUT_DIR / "Synthetic_all-MiniLM-L6-v2_embeddings.npy"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

print("=" * 70)
print("ALL-MiniLM-L6-v2 CORPUS EMBEDDING")
print("=" * 70)

print(f"\nLoading model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully.")


# ============================================================
# FUNCTION: EMBED TEXTS
# ============================================================

def create_embeddings(texts, output_file, corpus_name):

    print("\n" + "=" * 70)
    print(f"PROCESSING: {corpus_name}")
    print("=" * 70)

    print(f"Number of texts: {len(texts)}")

    # --------------------------------------------------------
    # Generate embeddings
    # normalize_embeddings=False is intentional.
    #
    # We are creating the raw MiniLM embeddings here.
    # Normalization can be applied later during retrieval.
    # --------------------------------------------------------

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    print(f"Embedding shape: {embeddings.shape}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(
        output_file,
        embeddings
    )

    print(f"Saved to: {output_file}")

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    saved = np.load(output_file)

    print(f"Reloaded shape: {saved.shape}")

    if saved.shape != embeddings.shape:
        raise ValueError(
            "Verification failed: saved shape differs "
            "from original embedding shape."
        )

    print("Verification: PASSED")

    return embeddings


# ============================================================
# 1. B01 CORPUS
# ============================================================

print("\n")
print("Loading B01 metadata...")

with open(
    B01_METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    b01_metadata = json.load(f)


# ------------------------------------------------------------
# Preserve the JSON's chunk ordering.
#
# The existing metadata is keyed by chunk ID.
# We sort naturally by chunk number where possible.
# ------------------------------------------------------------

def chunk_sort_key(chunk_id):

    try:
        return int(str(chunk_id).split("_")[-1])
    except ValueError:
        return str(chunk_id)


b01_chunk_ids = sorted(
    b01_metadata.keys(),
    key=chunk_sort_key
)


# ------------------------------------------------------------
# Extract text
# ------------------------------------------------------------

b01_texts = []

for chunk_id in b01_chunk_ids:

    metadata = b01_metadata[chunk_id]

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Unexpected metadata format for {chunk_id}"
        )

    text = metadata.get("text", "")

    if not text:
        raise ValueError(
            f"No text found for B01 {chunk_id}"
        )

    b01_texts.append(str(text))


print(f"B01 chunks found: {len(b01_texts)}")

b01_embeddings = create_embeddings(
    texts=b01_texts,
    output_file=B01_OUTPUT_FILE,
    corpus_name="B01"
)


# ============================================================
# 2. SYNTHETIC CORPUS
# ============================================================

print("\n")
print("Loading Synthetic corpus...")

synthetic_df = pd.read_csv(
    SYNTHETIC_CORPUS_FILE
)

print(
    f"Synthetic corpus shape: {synthetic_df.shape}"
)

print(
    f"Synthetic columns: {list(synthetic_df.columns)}"
)


# ------------------------------------------------------------
# Identify corpus text column.
#
# Based on the established Synthetic corpus structure,
# the actual corpus text is the fourth column.
# ------------------------------------------------------------

if synthetic_df.shape[1] < 4:

    raise ValueError(
        "Synthetic corpus does not contain at least "
        "four columns."
    )


synthetic_text_column = synthetic_df.columns[3]

print(
    f"Synthetic text column: {synthetic_text_column}"
)


synthetic_texts = (
    synthetic_df[synthetic_text_column]
    .astype(str)
    .tolist()
)


# ------------------------------------------------------------
# Check for empty texts
# ------------------------------------------------------------

empty_indices = [
    i for i, text in enumerate(synthetic_texts)
    if not text.strip()
]

if empty_indices:

    raise ValueError(
        f"Synthetic corpus contains empty text at rows: "
        f"{empty_indices}"
    )


print(
    f"Synthetic corpus texts found: "
    f"{len(synthetic_texts)}"
)


synthetic_embeddings = create_embeddings(
    texts=synthetic_texts,
    output_file=SYNTHETIC_OUTPUT_FILE,
    corpus_name="Synthetic"
)


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

print(
    f"\nB01:"
    f"\n  Corpus vectors : {b01_embeddings.shape[0]}"
    f"\n  Dimensions     : {b01_embeddings.shape[1]}"
    f"\n  File           : {B01_OUTPUT_FILE}"
)

print(
    f"\nSynthetic:"
    f"\n  Corpus vectors : {synthetic_embeddings.shape[0]}"
    f"\n  Dimensions     : {synthetic_embeddings.shape[1]}"
    f"\n  File           : {SYNTHETIC_OUTPUT_FILE}"
)

print("\n" + "=" * 70)
print("ALL EMBEDDINGS CREATED SUCCESSFULLY")
print("=" * 70)