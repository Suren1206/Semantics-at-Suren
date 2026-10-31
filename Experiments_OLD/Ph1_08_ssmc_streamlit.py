import streamlit as st
import numpy as np
from MAIN_embedding_wrapper_dev import embed_texts

# ==============================
# CONFIG
# ==============================

MODEL_KEY = "ollama_mxbai"
DOC_PATH = "/home/surendran/LangChain/Notebooks/Projects/Semantics@Suren_Phase_I/ML_Question_Bank.csv"
TOP_K = 5

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

st.title("Mean-Centered Retrieval")

st.write(f"Loaded documents: {len(texts)}")

# ==============================
# EMBEDDINGS
# ==============================

doc_vectors = embed_texts(MODEL_KEY, texts).astype("float32")

# ==============================
# MEAN CENTERING + NORMALIZATION
# ==============================

global_mean_vector = np.mean(doc_vectors, axis=0)

E_centered = doc_vectors - global_mean_vector

norms = np.linalg.norm(E_centered, axis=1, keepdims=True)
epsilon = 1e-10
norms = np.maximum(norms, epsilon)

E_centered_normalized = E_centered / norms

# ==============================
# QUERY INPUT
# ==============================

query = st.text_input("Enter your query:")

if query:

    query_vector = embed_texts(MODEL_KEY, query).astype("float32")

    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    # Mean center query using SAME mean
    query_centered = query_vector - global_mean_vector

    query_norm = query_centered / np.linalg.norm(query_centered)

    # Cosine similarity
    scores = np.dot(E_centered_normalized, query_norm.T).flatten()

    top_k_idx = np.argsort(-scores)[:TOP_K]

    st.subheader("Top Results (Mean-Centered Cosine)")

    for idx in top_k_idx:
        st.write(f"{scores[idx]:.6f}  |  {texts[idx]}")