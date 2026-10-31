from pathlib import Path
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_DIR = PROJECT_ROOT / "Outputs"

CORPUS_FILE = DATA_DIR / "corpus.json"
QUERIES_FILE = DATA_DIR / "queries.json"

VECTOR_FILE = OUTPUT_DIR / "phase2_multivector_representations.npz"
METADATA_FILE = OUTPUT_DIR / "phase2_multivector_metadata.json"
REPORT_FILE = OUTPUT_DIR / "phase2_representation_report.txt"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "BAAI/bge-m3"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# LOAD DATA
# ============================================================

with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    corpus_data = json.load(f)

with open(QUERIES_FILE, "r", encoding="utf-8") as f:
    queries_data = json.load(f)

chunks = corpus_data["chunks"]
queries = queries_data


# ============================================================
# LOAD MODEL
# ============================================================

print(f"Loading model: {MODEL_NAME}")
print(f"Device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    local_files_only=True
)

model.to(DEVICE)
model.eval()


# ============================================================
# REPRESENTATION EXTRACTION
# ============================================================

def extract_multivectors(text):
    """
    Extract contextual token representations.

    Each retained token produces one contextual vector.

    Special tokens and padding tokens are excluded.
    """

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
        return_special_tokens_mask=True
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)
    special_tokens_mask = encoded["special_tokens_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    # Last hidden state:
    # [batch, sequence_length, hidden_size]
    hidden_states = outputs.last_hidden_state[0]

    input_ids_cpu = input_ids[0].cpu().tolist()
    attention_cpu = attention_mask[0].cpu().tolist()
    special_cpu = special_tokens_mask[0].cpu().tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids_cpu)

    vectors = []
    retained_tokens = []

    for i, token in enumerate(tokens):

        # Ignore padding
        if attention_cpu[i] == 0:
            continue

        # Ignore special tokens such as <s> and </s>
        if special_cpu[i] == 1:
            continue

        vector = hidden_states[i].cpu().numpy().astype(np.float32)

        vectors.append(vector)
        retained_tokens.append(token)

    vectors = np.stack(vectors)

    return retained_tokens, vectors


# ============================================================
# EXTRACT CORPUS REPRESENTATIONS
# ============================================================

metadata = {
    "model": MODEL_NAME,
    "device": DEVICE,
    "representation_type": "contextual token representations",
    "special_tokens_removed": True,
    "padding_removed": True,
    "corpus": {},
    "queries": {}
}

all_vectors = {}
vector_counter = 0

print("\nExtracting corpus representations...")

for chunk in chunks:

    chunk_id = chunk["chunk_id"]
    text = chunk["text"]

    tokens, vectors = extract_multivectors(text)

    vector_ids = []

    for local_index in range(vectors.shape[0]):

        vector_id = f"V{vector_counter:05d}"

        all_vectors[vector_id] = vectors[local_index]

        vector_ids.append(vector_id)

        vector_counter += 1

    metadata["corpus"][chunk_id] = {
        "text": text,
        "tokens": tokens,
        "vector_ids": vector_ids,
        "vector_count": int(vectors.shape[0]),
        "vector_dimension": int(vectors.shape[1])
    }


# ============================================================
# EXTRACT QUERY REPRESENTATIONS
# ============================================================

print("Extracting query representations...")

for query in queries:

    query_id = query["query_id"]
    text = query["query"]

    tokens, vectors = extract_multivectors(text)

    vector_ids = []

    for local_index in range(vectors.shape[0]):

        vector_id = f"V{vector_counter:05d}"

        all_vectors[vector_id] = vectors[local_index]

        vector_ids.append(vector_id)

        vector_counter += 1

    metadata["queries"][query_id] = {
        "query": text,
        "tokens": tokens,
        "vector_ids": vector_ids,
        "vector_count": int(vectors.shape[0]),
        "vector_dimension": int(vectors.shape[1])
    }


# ============================================================
# SAVE VECTOR MATRIX
# ============================================================

vector_ids = list(all_vectors.keys())

vector_matrix = np.stack(
    [all_vectors[vector_id] for vector_id in vector_ids]
)

np.savez_compressed(
    VECTOR_FILE,
    vector_ids=np.array(vector_ids),
    vectors=vector_matrix
)


# ============================================================
# SAVE METADATA
# ============================================================

with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# BUILD REPORT
# ============================================================

corpus_counts = [
    item["vector_count"]
    for item in metadata["corpus"].values()
]

query_counts = [
    item["vector_count"]
    for item in metadata["queries"].values()
]

corpus_dimensions = sorted(
    set(
        item["vector_dimension"]
        for item in metadata["corpus"].values()
    )
)

query_dimensions = sorted(
    set(
        item["vector_dimension"]
        for item in metadata["queries"].values()
    )
)

report_lines = []

report_lines.append("PHASE 2 — MULTI-VECTOR REPRESENTATION")
report_lines.append("=" * 60)
report_lines.append("")

report_lines.append("Purpose")
report_lines.append("-------")
report_lines.append(
    "Inspect the structural change from a single-vector "
    "representation to multiple contextual vectors."
)
report_lines.append("")

report_lines.append("Phase 1 representation")
report_lines.append("----------------------")
report_lines.append(
    "Chunk -> ONE vector"
)
report_lines.append(
    "mxbai-embed-large:335m -> 1024 dimensions"
)
report_lines.append("")

report_lines.append("Phase 2 representation")
report_lines.append("----------------------")
report_lines.append(
    "Chunk -> MULTIPLE contextual vectors"
)
report_lines.append(
    "BGE-M3 contextual hidden states -> 1024 dimensions per vector"
)
report_lines.append("")

report_lines.append("Corpus")
report_lines.append("------")
report_lines.append(f"Number of chunks: {len(chunks)}")
report_lines.append(
    f"Vector count per chunk: "
    f"min={min(corpus_counts)}, "
    f"max={max(corpus_counts)}, "
    f"avg={np.mean(corpus_counts):.2f}"
)
report_lines.append(
    f"Vector dimensions observed: {corpus_dimensions}"
)
report_lines.append("")

report_lines.append("Queries")
report_lines.append("-------")
report_lines.append(f"Number of queries: {len(queries)}")
report_lines.append(
    f"Vector count per query: "
    f"min={min(query_counts)}, "
    f"max={max(query_counts)}, "
    f"avg={np.mean(query_counts):.2f}"
)
report_lines.append(
    f"Vector dimensions observed: {query_dimensions}"
)
report_lines.append("")

report_lines.append("Token/vector correspondence")
report_lines.append("---------------------------")

for chunk_id, item in metadata["corpus"].items():

    report_lines.append(
        f"{chunk_id}: "
        f"{item['vector_count']} vectors | "
        f"{item['vector_dimension']} dimensions"
    )

    report_lines.append(
        "  tokens: " + " | ".join(item["tokens"])
    )

report_lines.append("")

report_lines.append("Query token/vector correspondence")
report_lines.append("---------------------------------")

for query_id, item in metadata["queries"].items():

    report_lines.append(
        f"{query_id}: "
        f"{item['vector_count']} vectors | "
        f"{item['vector_dimension']} dimensions"
    )

    report_lines.append(
        "  tokens: " + " | ".join(item["tokens"])
    )

report_lines.append("")

report_lines.append("Interpretation boundary")
report_lines.append("-----------------------")
report_lines.append(
    "This phase does not perform retrieval, MaxSim, ranking, "
    "or relevance evaluation."
)
report_lines.append(
    "The purpose is to observe the structural representation "
    "change only."
)
report_lines.append(
    "The numerical vector spaces of mxbai and BGE-M3 are not "
    "directly compared because they are different models."
)

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\nPhase 2 complete.")
print(f"Vector artifact : {VECTOR_FILE}")
print(f"Metadata        : {METADATA_FILE}")
print(f"Report          : {REPORT_FILE}")
print(f"Total vectors   : {len(vector_ids)}")
print(f"Vector dimension: {vector_matrix.shape[1]}")