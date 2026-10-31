import numpy as np
from MAIN_embedding_wrapper_dev import embed_texts

# ==============================
# CONFIG
# ==============================

MODEL_KEY = "ollama_mxbai"
DOC_PATH = "ML_Question_Bank.csv"
TOP_K = 5

query = "what is autocorrelation"

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

# ==============================
# EMBEDDINGS
# ==============================

doc_vectors = embed_texts(MODEL_KEY, texts).astype("float32")
query_vector = embed_texts(MODEL_KEY, query).astype("float32")

# Make sure query shape is (1, dim)
if query_vector.ndim == 1:
    query_vector = query_vector.reshape(1, -1)

# ==============================
# A) RAW DOT PRODUCT
# ==============================

raw_dot_scores = np.dot(doc_vectors, query_vector.T).flatten()
raw_dot_idx = np.argsort(-raw_dot_scores)[:TOP_K]

print("\n================ RAW DOT PRODUCT ================")
for idx in raw_dot_idx:
    print(f"{raw_dot_scores[idx]:.6f}  |  {texts[idx]}")

# ==============================
# B) COSINE SIMILARITY
# ==============================

doc_norm = doc_vectors / np.linalg.norm(doc_vectors, axis=1, keepdims=True)
query_norm = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)

cos_scores = np.dot(doc_norm, query_norm.T).flatten()
cos_idx = np.argsort(-cos_scores)[:TOP_K]

print("\n================ COSINE SIMILARITY ================")
for idx in cos_idx:
    print(f"{cos_scores[idx]:.6f}  |  {texts[idx]}")

# ==============================
# C) EUCLIDEAN DISTANCE (RAW)
# ==============================

raw_l2 = np.linalg.norm(doc_vectors - query_vector, axis=1)
raw_l2_idx = np.argsort(raw_l2)[:TOP_K]

print("\n================ EUCLIDEAN (RAW) ================")
for idx in raw_l2_idx:
    print(f"{raw_l2[idx]:.6f}  |  {texts[idx]}")

# ==============================
# D) EUCLIDEAN (NORMALIZED)
# ==============================

norm_l2 = np.linalg.norm(doc_norm - query_norm, axis=1)
norm_l2_idx = np.argsort(norm_l2)[:TOP_K]

print("\n================ EUCLIDEAN (NORMALIZED) ================")
for idx in norm_l2_idx:
    print(f"{norm_l2[idx]:.6f}  |  {texts[idx]}")
