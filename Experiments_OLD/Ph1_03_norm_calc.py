import numpy as np
from MAIN_embedding_wrapper_dev import embed_texts

# ==============================
# CONFIG
# ==============================

MODEL_KEY = "ollama_mxbai"
DOC_PATH = "/home/surendran/LangChain/Notebooks/Projects/Semantics@Suren_Phase_I/ML_Question_Bank.csv"
TOP_K = 5

query = "autocorrelation"

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
# L2 NORMS OF DOCUMENT EMBEDDINGS
# ==============================

doc_l2_norms = np.linalg.norm(doc_vectors, axis=1)

print("\n================ L2 NORMS (DOCUMENTS) ================")
for i, norm in enumerate(doc_l2_norms):
    print(f"Doc {i}: {norm:.6f}")

print("Min norm:", np.min(doc_l2_norms))
print("Max norm:", np.max(doc_l2_norms))
print("Mean norm:", np.mean(doc_l2_norms))
print("Std norm:", np.std(doc_l2_norms))