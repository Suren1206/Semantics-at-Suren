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
# GLOBAL MEAN VECTOR
# ==============================

global_mean_vector = np.mean(doc_vectors, axis=0)

#print("\nGlobal Mean Vector shape:", global_mean_vector.shape)
#print("First 10 values:", global_mean_vector[:10])

# ==============================
# ALIGNMENT TO GLOBAL MEAN (COSINE)
# ==============================

# Normalize document vectors
doc_norm = doc_vectors / np.linalg.norm(doc_vectors, axis=1, keepdims=True)

# Normalize global mean vector
mean_norm = global_mean_vector / np.linalg.norm(global_mean_vector)

# Compute cosine similarity
alignment_scores = np.dot(doc_norm, mean_norm)

#print("\n================ ALIGNMENT TO GLOBAL MEAN ================")
#for i, score in enumerate(alignment_scores):
#    print(f"Doc {i}: {score:.6f}")

##==============================
# BUCKETING ALIGNMENT SCORES
# ==============================

low = alignment_scores[alignment_scores < 0.50]
mid = alignment_scores[(alignment_scores >= 0.50) & (alignment_scores <= 0.74)]
high = alignment_scores[alignment_scores >= 0.75]

print("\n================ ALIGNMENT SUMMARY ================")
print(f"Less than 0.50       : {len(low)}")
print(f"Between 0.50 - 0.74  : {len(mid)}")
print(f"Greater than 0.75    : {len(high)}")

total = len(alignment_scores)

print(f"\nPercentage < 0.50       : {len(low)/total*100:.2f}%")
print(f"Percentage 0.50 - 0.74  : {len(mid)/total*100:.2f}%")
print(f"Percentage >= 0.75      : {len(high)/total*100:.2f}%")