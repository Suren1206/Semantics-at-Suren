import numpy as np
from langchain_ollama import OllamaEmbeddings


# ================================================================
# MODEL
# ================================================================

MODEL = "mxbai-embed-large:335m"

embedding_model = OllamaEmbeddings(
    model=MODEL
)


# ================================================================
# FIVE ANSWERS
# ================================================================

ANSWERS = {

    "A1": (
        "Visualise embeddings as points or vectors and inspect their "
        "distances and directions."
    ),

    "A2": (
        "Study mathematical properties such as cosine similarity and "
        "Euclidean distance."
    ),

    "A3": (
        "Use controlled experiments that perturb text and observe how "
        "its embedding changes."
    ),

    "A4": (
        "a " 
    ),

    "A5": (
        "SSeetha is the wife of Rama; Ravana got into love with her and forcefully took her away  "
       
    )
}


# ================================================================
# GENERATE EMBEDDINGS
# ================================================================

print("=" * 80)
print("Generating MXBAI embeddings for five answers")
print("=" * 80)

ids = list(ANSWERS.keys())

vectors = []

for answer_id in ids:

    print(f"Embedding {answer_id}...")

    vector = embedding_model.embed_query(
        ANSWERS[answer_id]
    )

    vectors.append(vector)


# ================================================================
# CONVERT TO NUMPY MATRIX
# ================================================================

embeddings = np.asarray(
    vectors,
    dtype=np.float32
)


# ================================================================
# SAVE
# ================================================================

output_file = "synthetic_five_answers_mxbai_embeddings.npy"

np.save(
    output_file,
    embeddings
)


# ================================================================
# VERIFY
# ================================================================

print()
print("=" * 80)
print("COMPLETE")
print("=" * 80)

print(f"Model           : {MODEL}")
print(f"Number of answers: {len(ids)}")
print(f"Embedding shape : {embeddings.shape}")
print(f"Saved file      : {output_file}")

print()
print("Answer order in the NPY file:")
for i, answer_id in enumerate(ids):
    print(f"{i} -> {answer_id}")

print("=" * 80)