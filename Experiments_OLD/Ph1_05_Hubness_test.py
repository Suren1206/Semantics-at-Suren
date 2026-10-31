import numpy as np
from MAIN_embedding_wrapper_dev import embed_texts

# ==============================
# CONFIG
# ==============================

MODEL_KEY = "ollama_mxbai"
DOC_PATH = "/home/surendran/LangChain/Notebooks/Projects/Semantics@Suren_Phase_I/ML_Question_Bank.csv"

# ==============================
# LOAD QUESTIONS
# ==============================

def load_questions(path):
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                questions.append(text)
    return questions

texts = load_questions(DOC_PATH)
print("Loaded documents:", len(texts))

doc_vectors = embed_texts(MODEL_KEY, texts).astype("float32")


# ==============================
# HUBNESS TEST (TOP-5 COSINE)
# ==============================

# Normalize embeddings
doc_norm = doc_vectors / np.linalg.norm(doc_vectors, axis=1, keepdims=True)

# Cosine similarity matrix (190 x 190)
similarity_matrix = np.dot(doc_norm, doc_norm.T)

# Set diagonal to -inf so a vector is not retrieved as its own neighbor
np.fill_diagonal(similarity_matrix, -np.inf)

TOP_K = 5
n_docs = similarity_matrix.shape[0]

# Count how many times each document appears in top-5 of others
hub_counts = np.zeros(n_docs, dtype=int)

for i in range(n_docs):
    top_k_indices = np.argsort(-similarity_matrix[i])[:TOP_K]
    for idx in top_k_indices:
        hub_counts[idx] += 1

print("\n================ HUBNESS SUMMARY ================")
print("Min hub count :", np.min(hub_counts))
print("Max hub count :", np.max(hub_counts))
print("Mean hub count:", np.mean(hub_counts))
print("Std hub count :", np.std(hub_counts))

# Show top hubs
top_hubs = np.argsort(-hub_counts)[:10]

print("\nTop 10 Hubs:")
for idx in top_hubs:
    print(f"Doc {idx} | Appears {hub_counts[idx]} times")