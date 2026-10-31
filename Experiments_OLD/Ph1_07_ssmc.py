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

global_mean_vector = np.mean(doc_vectors, axis=0)

# ==============================
# MEAN CENTERING
# ==============================

# Step 1: Mean Centering
E_centered = doc_vectors - global_mean_vector

# Step 2: Compute L2 norms (avoid divide-by-zero)
norms = np.linalg.norm(E_centered, axis=1, keepdims=True)

# Add small epsilon for numerical safety
epsilon = 1e-10
norms = np.maximum(norms, epsilon)

# Step 3: Normalize to unit length
E_centered_normalized = E_centered / norms

print("Centered matrix shape:", E_centered.shape)
print("Centered + normalized shape:", E_centered_normalized.shape)

