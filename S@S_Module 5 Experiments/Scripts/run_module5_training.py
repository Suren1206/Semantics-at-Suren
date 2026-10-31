# ============================================================
# S@S MODULE 5 — REPRESENTATION LEARNING EXPERIMENT
# Training Phase
#
# E1 = untouched all-MiniLM-L6-v2 baseline
# E2 = Contrastive
# E3 = Triplet
# E4 = In-Batch InfoNCE
# E5 = Hard Negative
# E6 = Angular Margin
#
# IMPORTANT:
# - Only the 5 cases in Intervention_Matrix.csv are used.
# - Every E2-E6 starts from a fresh MiniLM checkpoint.
# - E4 uses the 5 positive pairs as ONE batch.
#   The other 4 positives automatically act as in-batch negatives.
# - Q108 is excluded from Triplet because the matrix says
#   "Triplet Loss not applicable".
# - No synthetic corpus is used.
# ============================================================

from pathlib import Path
import os
import random
import re
import json
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.optim import AdamW
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUTS = ROOT / "Inputs"
OUTPUTS = ROOT / "Outputs"
MODELS = OUTPUTS / "Models"

OUTPUTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

MATRIX_FILE = INPUTS / "Intervention_Matrix.csv"
QUESTIONS_FILE = INPUTS / "Questions_B01_150.csv"
CORPUS_METADATA_FILE = INPUTS / "20260424_B01" / "chunk_id_to_metadata.json"


# ============================================================
# 2. EXPERIMENT CONFIGURATION
# ============================================================

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SEED = 42

EPOCHS = 5
LEARNING_RATE = 2e-5
BATCH_SIZE = 5

# These are experimental controls, not parameters supplied
# by Intervention_Matrix.csv.
CONTRASTIVE_MARGIN = 0.5
INFONCE_TEMPERATURE = 0.07
ANGULAR_MARGIN_DEGREES = 10.0

TOP_K = 5

CASES = ["Q56", "Q103", "Q108", "Q112", "Q148"]


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(SEED)


# ============================================================
# 4. DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("S@S MODULE 5 — REPRESENTATION LEARNING TRAINING")
print("=" * 70)
print(f"Base model       : {BASE_MODEL}")
print(f"Device           : {DEVICE}")
print(f"Seed             : {SEED}")
print(f"Epochs           : {EPOCHS}")
print(f"Learning rate    : {LEARNING_RATE}")
print(f"InfoNCE temp.    : {INFONCE_TEMPERATURE}")
print(f"Angular margin   : {ANGULAR_MARGIN_DEGREES} degrees")
print("=" * 70)


# ============================================================
# 5. LOAD INTERVENTION MATRIX
# ============================================================

matrix = pd.read_csv(MATRIX_FILE)

required_matrix_columns = [
    "Unnamed: 0",
    "Contrastive: what constitutes the positive and negatives?",
    "Triplet: which chunk is the negative?",
    "InfoNCE: what constitutes the candidate set?",
    "Hard Negative: is chunk_4 explicitly designated as the hard negative?",
    "Angular Margin: what relationship receives the angular margin?",
]

missing = [
    c for c in required_matrix_columns
    if c not in matrix.columns
]

if missing:
    raise ValueError(
        f"Intervention_Matrix.csv is missing columns: {missing}"
    )


def extract_qno(value):
    if pd.isna(value):
        return None

    match = re.search(r"Q(\d+)", str(value))

    if not match:
        return None

    return f"Q{match.group(1)}"


matrix["Case"] = matrix["Unnamed: 0"].apply(extract_qno)

matrix = matrix[matrix["Case"].isin(CASES)].copy()

if set(matrix["Case"]) != set(CASES):
    missing_cases = sorted(set(CASES) - set(matrix["Case"]))
    raise ValueError(
        f"Intervention matrix does not contain all required cases: {missing_cases}"
    )

matrix = (
    matrix
    .set_index("Case")
    .loc[CASES]
    .reset_index()
)


# ============================================================
# 6. LOAD QUESTIONS
# ============================================================

questions = pd.read_csv(QUESTIONS_FILE)

if "Qno" not in questions.columns:
    raise ValueError(
        "Questions_B01_150.csv must contain column 'Qno'."
    )

if "Question" not in questions.columns:
    raise ValueError(
        "Questions_B01_150.csv must contain column 'Question'."
    )

if len(questions) != 150:
    raise ValueError(
        f"Expected 150 questions, found {len(questions)}."
    )

questions["Case"] = questions["Qno"].apply(
    lambda x: f"Q{int(x)}"
)

question_lookup = dict(
    zip(
        questions["Case"],
        questions["Question"]
    )
)


for case in CASES:
    if case not in question_lookup:
        raise ValueError(
            f"{case} is missing from Questions_B01_150.csv."
        )


# ============================================================
# 7. LOAD B01 CORPUS
# ============================================================

with open(CORPUS_METADATA_FILE, "r", encoding="utf-8") as f:
    corpus_metadata = json.load(f)


def extract_chunk_text(metadata):
    if isinstance(metadata, str):
        return metadata

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Unexpected metadata type: {type(metadata)}"
        )

    for key in [
        "text",
        "chunk_text",
        "content",
        "document_text",
    ]:
        if key in metadata:
            return str(metadata[key])

    raise ValueError(
        f"Could not find chunk text in metadata: {metadata}"
    )


# Handle either dictionary or list representation.
if isinstance(corpus_metadata, dict):

    corpus_items = []

    for chunk_id, metadata in corpus_metadata.items():
        corpus_items.append(
            (
                str(chunk_id),
                extract_chunk_text(metadata),
            )
        )

else:

    corpus_items = []

    for item in corpus_metadata:

        if not isinstance(item, dict):
            continue

        chunk_id = (
            item.get("chunk_id")
            or item.get("id")
        )

        text = extract_chunk_text(item)

        if chunk_id is not None:
            corpus_items.append(
                (
                    str(chunk_id),
                    text,
                )
            )


if len(corpus_items) != 161:
    raise ValueError(
        f"Expected 161 B01 chunks, found {len(corpus_items)}."
    )


chunk_text = {
    chunk_id: text
    for chunk_id, text in corpus_items
}


# ============================================================
# 8. CHUNK IDENTIFIER NORMALIZATION
# ============================================================

def normalize_chunk_id(value):

    if pd.isna(value):
        return None

    text = str(value).strip()

    match = re.search(
        r"chunk[_\s-]*(\d+)",
        text,
        flags=re.IGNORECASE
    )

    if match:
        return f"chunk_{int(match.group(1))}"

    return text


def get_chunk_from_text(value):

    if pd.isna(value):
        return None

    text = str(value)

    match = re.search(
        r"chunk[_\s-]*(\d+)",
        text,
        flags=re.IGNORECASE
    )

    if match:
        return f"chunk_{int(match.group(1))}"

    return None


# ============================================================
# 9. BUILD EXPERIMENT RELATIONSHIPS
# ============================================================

contrastive_col = (
    "Contrastive: what constitutes the positive and negatives?"
)

triplet_col = (
    "Triplet: which chunk is the negative?"
)

infonce_col = (
    "InfoNCE: what constitutes the candidate set?"
)

hard_negative_col = (
    "Hard Negative: is chunk_4 explicitly designated as the hard negative?"
)

angular_col = (
    "Angular Margin: what relationship receives the angular margin?"
)


def parse_contrastive(value):

    if pd.isna(value):
        return None, []

    text = str(value)

    positive_match = re.search(
        r"Chunk\s*(\d+)\s*is\s*positive",
        text,
        flags=re.IGNORECASE
    )

    if positive_match is None:
        return None, []

    positive = f"chunk_{int(positive_match.group(1))}"

    negatives = [
        f"chunk_{int(x)}"
        for x in re.findall(
            r"Chunk\s*(\d+)\s*is\s*negative",
            text,
            flags=re.IGNORECASE
        )
    ]

    return positive, negatives


def parse_triplet(value):

    if pd.isna(value):
        return None

    text = str(value)

    positive_match = re.search(
        r"Chunk\s*(\d+)\s*=\s*positive",
        text,
        flags=re.IGNORECASE
    )

    negative_match = re.search(
        r"Chunk\s*(\d+)\s*=\s*negative",
        text,
        flags=re.IGNORECASE
    )

    if positive_match is None:
        return None

    positive = f"chunk_{int(positive_match.group(1))}"

    negative = (
        f"chunk_{int(negative_match.group(1))}"
        if negative_match
        else None
    )

    margin_match = re.search(
        r"margin\s*=\s*([0-9.]+)",
        text,
        flags=re.IGNORECASE
    )

    margin = (
        float(margin_match.group(1))
        if margin_match
        else None
    )

    return {
        "positive": positive,
        "negative": negative,
        "margin": margin,
    }


def parse_infonce(value):

    if pd.isna(value):
        return None

    return get_chunk_from_text(value)


def parse_hard_negative(value):

    return get_chunk_from_text(value)


def parse_angular(value):

    if pd.isna(value):
        return None

    text = str(value)

    positive = get_chunk_from_text(text)

    degree_match = re.search(
        r"(\d+(?:\.\d+)?)\s*°",
        text
    )

    degrees = (
        float(degree_match.group(1))
        if degree_match
        else ANGULAR_MARGIN_DEGREES
    )

    return {
        "positive": positive,
        "degrees": degrees,
    }


EXPERIMENTS = {}

for _, row in matrix.iterrows():

    case = row["Case"]

    positive, negatives = parse_contrastive(
        row[contrastive_col]
    )

    triplet = parse_triplet(
        row[triplet_col]
    )

    infonce_positive = parse_infonce(
        row[infonce_col]
    )

    hard_negative = parse_hard_negative(
        row[hard_negative_col]
    )

    angular = parse_angular(
        row[angular_col]
    )

    EXPERIMENTS[case] = {
        "question": question_lookup[case],
        "contrastive_positive": positive,
        "contrastive_negatives": negatives,
        "triplet": triplet,
        "infonce_positive": infonce_positive,
        "hard_negative": hard_negative,
        "angular": angular,
    }


# ============================================================
# 10. VALIDATE MATRIX RELATIONSHIPS
# ============================================================

print("\nValidated intervention cases:")

for case in CASES:

    info = EXPERIMENTS[case]

    print(
        f"{case}: "
        f"Contrastive={info['contrastive_positive']}, "
        f"InfoNCE={info['infonce_positive']}, "
        f"HardNegative={info['hard_negative']}"
    )


# Validate every referenced chunk exists.

for case, info in EXPERIMENTS.items():

    references = []

    if info["contrastive_positive"]:
        references.append(info["contrastive_positive"])

    references.extend(
        info["contrastive_negatives"]
    )

    if info["triplet"]:
        references.append(
            info["triplet"]["positive"]
        )

        if info["triplet"]["negative"]:
            references.append(
                info["triplet"]["negative"]
            )

    if info["infonce_positive"]:
        references.append(
            info["infonce_positive"]
        )

    if info["hard_negative"]:
        references.append(
            info["hard_negative"]
        )

    if info["angular"]:
        references.append(
            info["angular"]["positive"]
        )

    for chunk_id in references:

        if chunk_id not in chunk_text:
            raise ValueError(
                f"{case} references {chunk_id}, "
                f"but that chunk is absent from B01."
            )


# ============================================================
# 11. MODEL LOADER
# ============================================================

def load_fresh_model():

    set_seed(SEED)

    model = SentenceTransformer(
        BASE_MODEL,
        device=DEVICE
    )

    return model


# ============================================================
# 12. TEXT ENCODING FOR TRAINING
# ============================================================

def encode_training_texts(model, texts):

    features = model.tokenize(texts)

    features = {
        key: value.to(DEVICE)
        if torch.is_tensor(value)
        else value
        for key, value in features.items()
    }

    output = model(features)

    embeddings = output["sentence_embedding"]

    return F.normalize(
        embeddings,
        p=2,
        dim=1
    )

# ============================================================
# 13. E2 — CONTRASTIVE
# ============================================================

def train_contrastive():

    print("\n" + "=" * 70)
    print("E2 — CONTRASTIVE")
    print("=" * 70)

    model = load_fresh_model()

    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    training_records = []

    for case in CASES:

        info = EXPERIMENTS[case]

        positive = info["contrastive_positive"]

        if positive:

            training_records.append(
                {
                    "case": case,
                    "question": info["question"],
                    "positive": positive,
                    "negatives": info["contrastive_negatives"],
                }
            )

    for epoch in range(EPOCHS):

        random.shuffle(training_records)

        total_loss = 0.0

        for record in training_records:

            optimizer.zero_grad()

            anchor = encode_training_texts(
                model,
                [record["question"]]
            )[0]

            positive = encode_training_texts(
                model,
                [chunk_text[record["positive"]]]
            )[0]

            positive_similarity = torch.sum(
                anchor * positive
            )

            loss = 1.0 - positive_similarity

            if record["negatives"]:

                for negative_id in record["negatives"]:

                    negative = encode_training_texts(
                        model,
                        [chunk_text[negative_id]]
                    )[0]

                    negative_similarity = torch.sum(
                        anchor * negative
                    )

                    ranking_loss = F.relu(
                        CONTRASTIVE_MARGIN
                        - positive_similarity
                        + negative_similarity
                    )

                    loss = loss + ranking_loss

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Loss: {total_loss:.6f}"
        )

    return model


# ============================================================
# 14. E3 — TRIPLET
# ============================================================

def train_triplet():

    print("\n" + "=" * 70)
    print("E3 — TRIPLET")
    print("=" * 70)

    model = load_fresh_model()

    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    records = []

    for case in CASES:

        triplet = EXPERIMENTS[case]["triplet"]

        if triplet is None:
            continue

        if triplet["negative"] is None:
            continue

        records.append(
            {
                "case": case,
                "question": EXPERIMENTS[case]["question"],
                "positive": triplet["positive"],
                "negative": triplet["negative"],
                "margin": triplet["margin"]
                if triplet["margin"] is not None
                else 0.10,
            }
        )

    for epoch in range(EPOCHS):

        random.shuffle(records)

        total_loss = 0.0

        for record in records:

            optimizer.zero_grad()

            anchor = encode_training_texts(
                model,
                [record["question"]]
            )[0]

            positive = encode_training_texts(
                model,
                [chunk_text[record["positive"]]]
            )[0]

            negative = encode_training_texts(
                model,
                [chunk_text[record["negative"]]]
            )[0]

            positive_similarity = torch.sum(
                anchor * positive
            )

            negative_similarity = torch.sum(
                anchor * negative
            )

            loss = F.relu(
                record["margin"]
                - positive_similarity
                + negative_similarity
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Loss: {total_loss:.6f}"
        )

    return model


# ============================================================
# 15. E4 — IN-BATCH INFONCE
# ============================================================

def train_infonce():

    print("\n" + "=" * 70)
    print("E4 — IN-BATCH INFONCE")
    print("=" * 70)

    model = load_fresh_model()

    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Exactly the five matrix cases.
    #
    # Each row has:
    #       query -> designated positive chunk
    #
    # Because all five are in ONE batch:
    #
    # Q56  -> chunk_2
    # Q103 -> chunk_21
    # Q108 -> chunk_17
    # Q112 -> chunk_11
    # Q148 -> chunk_40
    #
    # For each query, the other four positive chunks
    # automatically become in-batch negatives.
    #

    anchors = [
        EXPERIMENTS[case]["question"]
        for case in CASES
    ]

    positives = [
        chunk_text[
            EXPERIMENTS[case]["infonce_positive"]
        ]
        for case in CASES
    ]

    if len(anchors) != 5 or len(positives) != 5:
        raise ValueError(
            "E4 requires exactly five query-positive pairs."
        )

    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        anchor_embeddings = encode_training_texts(
            model,
            anchors
        )

        positive_embeddings = encode_training_texts(
            model,
            positives
        )

        # [5, 384] @ [384, 5] -> [5, 5]
        logits = (
            anchor_embeddings
            @ positive_embeddings.T
        ) / INFONCE_TEMPERATURE

        # Correct positive is on the diagonal.
        targets = torch.arange(
            len(CASES),
            device=DEVICE
        )

        loss = F.cross_entropy(
            logits,
            targets
        )

        loss.backward()

        optimizer.step()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Loss: {loss.item():.6f}"
        )

    return model


# ============================================================
# 16. E5 — HARD NEGATIVE
# ============================================================

def train_hard_negative():

    print("\n" + "=" * 70)
    print("E5 — HARD NEGATIVE")
    print("=" * 70)

    model = load_fresh_model()

    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    records = []

    for case in CASES:

        info = EXPERIMENTS[case]

        hard_negative = info["hard_negative"]

        if hard_negative is None:
            continue

        positive = info["contrastive_positive"]

        if positive is None:
            continue

        records.append(
            {
                "case": case,
                "question": info["question"],
                "positive": positive,
                "hard_negative": hard_negative,
            }
        )

    for epoch in range(EPOCHS):

        random.shuffle(records)

        total_loss = 0.0

        for record in records:

            optimizer.zero_grad()

            anchor = encode_training_texts(
                model,
                [record["question"]]
            )[0]

            positive = encode_training_texts(
                model,
                [chunk_text[record["positive"]]]
            )[0]

            hard_negative = encode_training_texts(
                model,
                [chunk_text[record["hard_negative"]]]
            )[0]

            positive_similarity = torch.sum(
                anchor * positive
            )

            negative_similarity = torch.sum(
                anchor * hard_negative
            )

            loss = F.relu(
                CONTRASTIVE_MARGIN
                - positive_similarity
                + negative_similarity
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Loss: {total_loss:.6f}"
        )

    return model


# ============================================================
# 17. E6 — ANGULAR MARGIN
# ============================================================

def train_angular_margin():

    print("\n" + "=" * 70)
    print("E6 — ANGULAR MARGIN")
    print("=" * 70)

    model = load_fresh_model()

    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    margin_radians = np.deg2rad(
        ANGULAR_MARGIN_DEGREES
    )

    records = []

    for case in CASES:

        angular = EXPERIMENTS[case]["angular"]

        if angular is None:
            continue

        records.append(
            {
                "case": case,
                "question": EXPERIMENTS[case]["question"],
                "positive": angular["positive"],
                "degrees": angular["degrees"],
            }
        )

    for epoch in range(EPOCHS):

        random.shuffle(records)

        total_loss = 0.0

        for record in records:

            optimizer.zero_grad()

            anchor = encode_training_texts(
                model,
                [record["question"]]
            )[0]

            positive = encode_training_texts(
                model,
                [chunk_text[record["positive"]]]
            )[0]

            cosine = torch.sum(
                anchor * positive
            )

            cosine = torch.clamp(
                cosine,
                -1.0 + 1e-7,
                1.0 - 1e-7
            )

            theta = torch.acos(cosine)

            target = torch.cos(
                theta + margin_radians
            )

            loss = 1.0 - target

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Loss: {total_loss:.6f}"
        )

    return model


# ============================================================
# 18. GENERATE B01 EMBEDDINGS
# ============================================================

def generate_b01_embeddings(model, experiment_name):

    model.eval()

    texts = [
        text
        for _, text in corpus_items
    ]

    with torch.no_grad():

        embeddings = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    if embeddings.shape != (161, 384):
        raise ValueError(
            f"{experiment_name}: expected "
            f"(161, 384), got {embeddings.shape}"
        )

    output_file = (
        OUTPUTS
        / f"{experiment_name}_B01_embeddings.npy"
    )

    np.save(
        output_file,
        embeddings
    )

    print(
        f"Saved embeddings: {output_file}"
    )

    return embeddings


# ============================================================
# 19. RUN 150 B01 QUESTIONS
# ============================================================

def run_retrieval(
    model,
    embeddings,
    experiment_name
):

    model.eval()

    questions_text = questions["Question"].tolist()

    with torch.no_grad():

        query_embeddings = model.encode(
            questions_text,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        )

    query_embeddings = np.asarray(
        query_embeddings,
        dtype=np.float32
    )

    similarity = (
        query_embeddings
        @ embeddings.T
    )

    rows = []

    chunk_ids = [
        chunk_id
        for chunk_id, _ in corpus_items
    ]

    for question_index in range(len(questions)):

        scores = similarity[question_index]

        top_indices = np.argsort(
            -scores
        )[:TOP_K]

        qno = questions.iloc[
            question_index
        ]["Qno"]

        question_text = questions.iloc[
            question_index
        ]["Question"]

        for rank, corpus_index in enumerate(
            top_indices,
            start=1
        ):

            rows.append(
                {
                    "Qno": qno,
                    "Question": question_text,
                    "Rank": rank,
                    "Chunk_ID": chunk_ids[corpus_index],
                    "Cosine_Similarity": float(
                        scores[corpus_index]
                    ),
                }
            )

    results = pd.DataFrame(rows)

    output_file = (
        OUTPUTS
        / f"{experiment_name}_B01_retrieval.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved retrieval: {output_file}"
    )

    return results


# ============================================================
# 20. TRAIN + EMBED + RETRIEVE
# ============================================================

EXPERIMENT_RUNNERS = {
    "E2_Contrastive_MiniLM": train_contrastive,
    "E3_Triplet_MiniLM": train_triplet,
    "E4_InfoNCE_MiniLM": train_infonce,
    "E5_HardNegative_MiniLM": train_hard_negative,
    "E6_AngularMargin_MiniLM": train_angular_margin,
}


for experiment_name, trainer in EXPERIMENT_RUNNERS.items():

    print("\n\n")
    print("#" * 70)
    print(f"STARTING {experiment_name}")
    print("#" * 70)

    set_seed(SEED)

    model = trainer()

    model_output_dir = (
        MODELS / experiment_name
    )

    model.save(
        str(model_output_dir)
    )

    print(
        f"Saved model: {model_output_dir}"
    )

    embeddings = generate_b01_embeddings(
        model,
        experiment_name
    )

    run_retrieval(
        model,
        embeddings,
        experiment_name
    )

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"COMPLETED {experiment_name}"
    )


# ============================================================
# 21. END
# ============================================================

print("\n" + "=" * 70)
print("MODULE 5 TRAINING / RETRIEVAL STAGE COMPLETE")
print("=" * 70)
print("Generated E2, E3, E4, E5 and E6 models.")
print("Generated 161-chunk B01 embeddings for each model.")
print("Generated 150-question Top-5 retrieval results for each model.")
print("No synthetic corpus was processed.")
print("Stopping here — no downstream analysis performed.")
print("=" * 70)