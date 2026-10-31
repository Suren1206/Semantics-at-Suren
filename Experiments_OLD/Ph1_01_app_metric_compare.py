import streamlit as st
import numpy as np
import os
from MAIN_embedding_wrapper_dev import embed_texts

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = "embedding_dbs"
DEFAULT_MODEL = "ollama_mxbai"
TOP_K = 5

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("Single Model – Metric Comparison")

model_key = st.selectbox(
    "Select embedding model:",
    sorted(os.listdir(BASE_DIR)),
    index=sorted(os.listdir(BASE_DIR)).index(DEFAULT_MODEL)
)

query = st.text_input(
    "Enter your query:",
    "Adding an irrelevant feature to a linear regression model always increases training error."
)

if st.button("Search"):

    MODEL_DIR = os.path.join(BASE_DIR, model_key)

    # Load texts
    with open(os.path.join(MODEL_DIR, "texts.txt"), "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    # Embed corpus
    vectors = embed_texts(model_key, texts).astype("float32")

    # Embed query
    query_vec = embed_texts(model_key, query).astype("float32")

    st.write(f"Loaded {len(texts)} questions")

    # ----------------------------
    # A) RAW DOT PRODUCT
    # ----------------------------
    dot_scores = (vectors @ query_vec.T).flatten()
    top_dot_idx = np.argsort(-dot_scores)[:TOP_K]

    st.subheader("A) Raw Dot Product")
    for idx in top_dot_idx:
        st.write(float(dot_scores[idx]), "|", texts[idx])

    # ----------------------------
    # B) COSINE SIMILARITY
    # ----------------------------
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    query_norm = query_vec / np.linalg.norm(query_vec)

    cos_scores = (vectors_norm @ query_norm.T).flatten()
    top_cos_idx = np.argsort(-cos_scores)[:TOP_K]

    st.subheader("B) Cosine Similarity (L2 Normalized Dot)")
    for idx in top_cos_idx:
        st.write(float(cos_scores[idx]), "|", texts[idx])

    # ----------------------------
    # C) RAW EUCLIDEAN
    # ----------------------------
    euclid_scores = np.linalg.norm(vectors - query_vec, axis=1)
    top_euclid_idx = np.argsort(euclid_scores)[:TOP_K]

    st.subheader("C) Raw Euclidean Distance")
    for idx in top_euclid_idx:
        st.write(float(euclid_scores[idx]), "|", texts[idx])

    # ----------------------------
    # D) EUCLIDEAN AFTER NORMALIZATION
    # ----------------------------
    euclid_norm_scores = np.linalg.norm(vectors_norm - query_norm, axis=1)
    top_euclid_norm_idx = np.argsort(euclid_norm_scores)[:TOP_K]

    st.subheader("D) Euclidean Distance (After L2 Normalization)")
    for idx in top_euclid_norm_idx:
        st.write(float(euclid_norm_scores[idx]), "|", texts[idx])
