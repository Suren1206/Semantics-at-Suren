import os
import numpy as np
import csv
import faiss
from docx import Document
from 01Bembedding_wrapper_dev import embed_texts, MODEL_CONFIG

DOC_PATH = "ML_Question_Bank.csv"
BASE_SAVE_DIR = "embedding_dbs"

os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# ---- Load questions exactly as they are ----


def load_questions(path):
    questions = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                questions.append(row[0].strip())
    return questions

questions = load_questions(DOC_PATH)

print("Total questions loaded:", len(questions))
print("\nFIRST 5 STORED QUESTIONS:")
for q in questions[:5]:
    print(q)

# ---- Build FAISS per model ----
for model_key in MODEL_CONFIG.keys():

    print(f"\nBuilding index for: {model_key}")

    vectors = embed_texts(model_key, questions)

    # Normalize for cosine similarity
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors.astype("float32")

    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    model_dir = os.path.join(BASE_SAVE_DIR, model_key)
    os.makedirs(model_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(model_dir, "index.faiss"))

    with open(os.path.join(model_dir, "texts.txt"), "w", encoding="utf-8") as f:
        for q in questions:
            f.write(q + "\n")

    print(f"Saved index for {model_key} | dim={dim}")
