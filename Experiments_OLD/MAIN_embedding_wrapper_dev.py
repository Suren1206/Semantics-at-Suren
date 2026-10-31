from dotenv import load_dotenv
load_dotenv()

# ==============================
# MODEL CONFIGURATION
# ==============================

MODEL_CONFIG = {
    "openai_large": {
        "provider": "openai",
        "model_name": "text-embedding-3-large"
    },
    "openai_small": {
        "provider": "openai",
        "model_name": "text-embedding-3-small"
    },
    "hf_minilm": {
        "provider": "huggingface",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "hf_mpnet": {
        "provider": "huggingface",
        "model_name": "sentence-transformers/all-mpnet-base-v2"
    },
    "ollama_nomic": {
        "provider": "ollama",
        "model_name": "nomic-embed-text:latest"
    },
    "ollama_mxbai": {
        "provider": "ollama",
        "model_name": "mxbai-embed-large:335m"
    }
}

# ==============================
# ENVIRONMENT SETUP
# ==============================

import os
import numpy as np
import requests
import faiss
import json

from openai import OpenAI
from sentence_transformers import SentenceTransformer

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

OLLAMA_BASE_URL = "http://localhost:11434"

HF_LOCAL_MODELS = {}

def get_hf_model(model_name):
    if model_name not in HF_LOCAL_MODELS:
        print(f"Loading local HF model: {model_name}")
        HF_LOCAL_MODELS[model_name] = SentenceTransformer(model_name)
    return HF_LOCAL_MODELS[model_name]

# ==============================
# EMBEDDING WRAPPER
# ==============================

def embed_texts(model_key, texts):

    if model_key not in MODEL_CONFIG:
        raise ValueError(f"Model '{model_key}' not found.")

    provider = MODEL_CONFIG[model_key]["provider"]
    model_name = MODEL_CONFIG[model_key]["model_name"]

    if isinstance(texts, str):
        texts = [texts]

    # ---- OpenAI ----
    if provider == "openai":
        response = openai_client.embeddings.create(
            model=model_name,
            input=texts
        )
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)

    # ---- HuggingFace (Local) ----
    elif provider == "huggingface":
        model = get_hf_model(model_name)
        vectors = model.encode(texts)
        return np.array(vectors, dtype=np.float32)

    # ---- Ollama ----
    elif provider == "ollama":
        vectors = []
        for text in texts:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": model_name, "prompt": text}
            )
            if response.status_code != 200:
                raise RuntimeError(f"Ollama error: {response.text}")
            vectors.append(response.json()["embedding"])
        return np.array(vectors, dtype=np.float32)

    else:
        raise ValueError("Unsupported provider.")

# ==============================
# LOAD PREBUILT FAISS DATABASE
# ==============================

BASE_SAVE_DIR = "embedding_dbs"

def load_model_db(model_key):

    model_dir = os.path.join(BASE_SAVE_DIR, model_key)

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Database for {model_key} not found.")

    index = faiss.read_index(os.path.join(model_dir, "index.faiss"))

    with open(os.path.join(model_dir, "texts.txt"), "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines()]

    return index, texts

# ==============================
# SEARCH FUNCTION
# ==============================

def search_query(model_key, query, threshold=0.5, top_k=5):

    index, texts = load_model_db(model_key)

    query_vec = embed_texts(model_key, query)

    # cosine normalization
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
    query_vec = query_vec.astype("float32")

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= threshold:
            results.append((float(score), texts[idx]))

    return results
