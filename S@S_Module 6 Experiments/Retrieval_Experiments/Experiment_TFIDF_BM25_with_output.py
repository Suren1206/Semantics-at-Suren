"""
===============================================================================
Retrieval Architecture Lab — Phase 1
Experiment: TF-IDF vs BM25
File: Experiment_TFIDF_BM25.py
===============================================================================

EXPERIMENTAL OBJECTIVE
----------------------
Implement two lexical retrieval methods over the SAME 161-chunk corpus:

    1. TF-IDF
    2. BM25

For each query, retrieve Top-5 chunks and expose enough internal evidence to
inspect how lexical retrieval works.

The experiment is intentionally verbose. This is a learning/laboratory script,
not a production retrieval component.

The script is designed to help us observe:

    Query
      ↓
    tokenization
      ↓
    lexical terms
      ↓
    term/document statistics
      ↓
    scoring
      ↓
    ranking
      ↓
    Top-K
      ↓
    original chunk

IMPORTANT EXPERIMENTAL CONTROL
------------------------------
Both TF-IDF and BM25 use the SAME:

    - CSV corpus
    - 161 chunks
    - chunk IDs
    - chunk text
    - tokenizer
    - queries
    - Top-K value

Only the retrieval/scoring method changes.

The CSV header is NOT included in the indexed text. The header is retained for
display only. This keeps Phase 1 focused on the chunk text rather than allowing
metadata/topic labels to influence lexical matching.

The script also builds a small EXPLICIT positional inverted index for inspection.
This diagnostic structure is NOT used to calculate the TF-IDF or BM25 scores.
It exists so that we can directly inspect:

    term -> chunk IDs -> positions

This distinction is important: an index can store positions, while a particular
retrieval scoring method may or may not actually use proximity information.

===============================================================================
"""

# =============================================================================
# BLOCK 1 — IMPORTS
# =============================================================================
#
# PURPOSE
# -------
# Import the libraries required for:
#
#   - reading the CSV
#   - TF-IDF retrieval
#   - BM25 retrieval
#   - numerical inspection
#
# REQUIRED PACKAGES
# -----------------
# Install once, if necessary:
#
#     pip install pandas scikit-learn rank-bm25
#
# =============================================================================

from pathlib import Path
import math
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi


# =============================================================================
# BLOCK 2 — EXPERIMENT PARAMETERS
# =============================================================================
#
# PURPOSE
# -------
# Keep all experiment controls in one place.
#
# This makes the experiment reproducible and makes it easy to see exactly what
# is being held constant.
#
# =============================================================================

# The CSV is expected to be in the SAME DIRECTORY as this Python script.
#
# This is deliberate: you can copy both files into your Retrieval_Experiments
# directory and run the script without hard-coding your home directory.
CSV_PATH = Path(__file__).resolve().parent / "chunk_corpus.csv"

# We retrieve the same number of results from both methods.
TOP_K = 5

# BM25 parameters.
#
# These are the commonly used Okapi BM25 defaults and are kept fixed for this
# first experiment. We are NOT tuning them yet.
BM25_K1 = 1.5
BM25_B = 0.75

# The two queries are frozen exactly as selected for the first experiment.
QUERY_1 = (
    "What is the method of notification to my manager "
    "when I am unable to attend work?"
)

QUERY_2 = (
    "When can self declaration form be used for illness absence?"
)

QUERIES = {
    "Q1": QUERY_1,
    "Q2": QUERY_2,
}


# =============================================================================
# BLOCK 3 — COMMON TOKENIZER
# =============================================================================
#
# PURPOSE
# -------
# Ensure that TF-IDF and BM25 receive the SAME lexical representation.
#
# Example:
#
#     "Employees must notify their direct manager."
#
# becomes approximately:
#
#     ["employees", "must", "notify", "their", "direct", "manager"]
#
# We deliberately keep the preprocessing simple for this first laboratory
# experiment. We do NOT introduce stemming, lemmatization, synonym expansion,
# or stop-word removal.
#
# Why?
# ----
# Because those changes would introduce additional variables before we have
# understood the basic lexical mechanisms.
#
# =============================================================================

TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text):
    """
    Convert text into the common lexical representation used by both methods.

    The tokenizer:
        1. converts text to lowercase
        2. extracts word-like tokens
        3. preserves repeated occurrences

    Repeated occurrences matter because term frequency (TF) depends on them.
    """
    return TOKEN_PATTERN.findall(str(text).lower())


# =============================================================================
# BLOCK 4 — LOAD AND VALIDATE THE COMMON CORPUS
# =============================================================================
#
# PURPOSE
# -------
# Load the CSV created from the original JSON and verify that the expected
# corpus structure is present.
#
# We expect:
#
#     chunk_id | header | text
#
# We also verify:
#
#     - 161 chunks
#     - IDs 0 through 160
#     - no missing text
#
# =============================================================================

def load_corpus():
    """Load the common 161-chunk corpus and perform basic integrity checks."""

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Corpus CSV not found:\n{CSV_PATH}\n\n"
            "Place chunk_corpus.csv in the same directory as this script."
        )

    df = pd.read_csv(CSV_PATH)

    required_columns = {"chunk_id", "header", "text"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing_columns)}"
        )

    # Convert IDs to integers so that ordering and mapping are unambiguous.
    df["chunk_id"] = df["chunk_id"].astype(int)

    # Verify the expected number of chunks.
    if len(df) != 161:
        raise ValueError(
            f"Expected 161 chunks, but found {len(df)}."
        )

    # Verify stable IDs 0 ... 160.
    expected_ids = list(range(161))
    actual_ids = df["chunk_id"].tolist()

    if actual_ids != expected_ids:
        raise ValueError(
            "Chunk IDs are not exactly 0 through 160 in order."
        )

    # Text must be present because it is the actual lexical retrieval field.
    if df["text"].isna().any():
        raise ValueError("One or more chunks has missing text.")

    return df


# =============================================================================
# BLOCK 5 — BUILD THE COMMON TOKEN REPRESENTATION
# =============================================================================
#
# PURPOSE
# -------
# Tokenize every chunk ONCE.
#
# Both retrieval methods will then consume this same tokenized representation.
#
# This is another experimental control:
#
#                 SAME TOKENS
#                    /   \
#                   /     \
#               TF-IDF    BM25
#
# =============================================================================

def build_tokenized_corpus(df):
    """
    Return one token list per chunk.

    The list position is aligned with the DataFrame row:

        tokenized_corpus[0] -> chunk_id 0
        tokenized_corpus[1] -> chunk_id 1
        ...
    """

    return [
        tokenize(text)
        for text in df["text"].tolist()
    ]


# =============================================================================
# BLOCK 6 — BUILD AN EXPLICIT INVERTED INDEX WITH POSITIONS
# =============================================================================
#
# PURPOSE
# -------
# Make the lexical index visible for inspection.
#
# An inverted index conceptually changes the direction of lookup:
#
#     DOCUMENT -> TERMS
#
# into:
#
#     TERM -> DOCUMENTS
#
# For example:
#
#     "manager"
#         -> chunk_0 at positions [...]
#         -> chunk_105 at positions [...]
#
# The positions allow us to inspect where a term occurs inside a chunk.
#
# IMPORTANT
# ---------
# This structure is a DIAGNOSTIC index for the laboratory.
#
# TF-IDF and BM25 scores below are produced by their respective retrieval
# implementations. We are not secretly replacing their scoring mechanisms with
# this dictionary.
#
# =============================================================================

def build_positional_inverted_index(df, tokenized_corpus):
    """
    Build:
        term -> chunk_id -> list of token positions
    """

    positional_index = defaultdict(lambda: defaultdict(list))

    for row_position, tokens in enumerate(tokenized_corpus):

        chunk_id = int(df.iloc[row_position]["chunk_id"])

        for position, term in enumerate(tokens):

            positional_index[term][chunk_id].append(position)

    return positional_index


# =============================================================================
# BLOCK 7 — TERM STATISTICS
# =============================================================================
#
# PURPOSE
# -------
# Calculate the basic lexical statistics we want to inspect:
#
#     TF = Term Frequency
#     DF = Document Frequency
#     IDF = Inverse Document Frequency
#
# TF:
#     How many times does the term occur in THIS chunk?
#
# DF:
#     In how many chunks does the term occur?
#
# IDF:
#     How discriminative is the term across the corpus?
#
# IMPORTANT
# ---------
# The IDF values shown here use the SAME smoothed IDF convention used by
# scikit-learn's TfidfVectorizer:
#
#     idf(t) = log((N + 1) / (DF + 1)) + 1
#
# where N is the number of documents.
#
# =============================================================================

def calculate_document_frequency(tokenized_corpus):
    """Calculate DF for every unique term."""

    df_counter = Counter()

    for tokens in tokenized_corpus:
        unique_terms = set(tokens)

        for term in unique_terms:
            df_counter[term] += 1

    return df_counter


def calculate_sklearn_idf(document_count, document_frequency):
    """Calculate the smoothed IDF convention used by sklearn."""

    return (
        math.log(
            (document_count + 1)
            / (document_frequency + 1)
        )
        + 1
    )


# =============================================================================
# BLOCK 8 — CREATE THE TF-IDF RETRIEVER
# =============================================================================
#
# PURPOSE
# -------
# Let scikit-learn construct the TF-IDF representation.
#
# Important settings:
#
#     lowercase=True
#     token_pattern=our common tokenizer pattern
#     norm="l2"
#
# L2 normalization is used so that the retrieval score can be calculated as
# cosine similarity through the dot product of normalized vectors.
#
# We deliberately do NOT remove stop words.
#
# We deliberately do NOT use semantic expansion.
#
# =============================================================================

def build_tfidf_retriever(df):
    """
    Build the TF-IDF matrix and return the fitted vectorizer.

    Each row in tfidf_matrix corresponds to one corpus chunk.
    Each column corresponds to one vocabulary term.
    """

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    )

    tfidf_matrix = vectorizer.fit_transform(
        df["text"].astype(str)
    )

    return vectorizer, tfidf_matrix


# =============================================================================
# BLOCK 9 — TF-IDF RETRIEVAL
# =============================================================================
#
# PURPOSE
# -------
# Retrieve Top-K chunks for one query.
#
# PROCESS
# -------
#
#     Query
#       ↓
#     TF-IDF query vector
#       ↓
#     cosine similarity against all chunk vectors
#       ↓
#     descending score
#       ↓
#     Top-K
#
# Because the vectors are L2-normalized, the dot product is cosine similarity.
#
# =============================================================================

def retrieve_tfidf(vectorizer, tfidf_matrix, df, query, top_k=5):
    """Return Top-K TF-IDF results."""

    query_vector = vectorizer.transform([query])

    # Sparse matrix multiplication gives cosine similarity because both the
    # query and document vectors are L2-normalized.
    scores = (tfidf_matrix @ query_vector.T).toarray().ravel()

    # Sort from highest score to lowest score.
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, row_index in enumerate(ranked_indices, start=1):

        row_index = int(row_index)

        results.append(
            {
                "rank": rank,
                "chunk_id": int(df.iloc[row_index]["chunk_id"]),
                "score": float(scores[row_index]),
                "header": df.iloc[row_index]["header"],
                "text": df.iloc[row_index]["text"],
            }
        )

    return results


# =============================================================================
# BLOCK 10 — CREATE THE BM25 RETRIEVER
# =============================================================================
#
# PURPOSE
# -------
# Build BM25 over the EXACT SAME tokenized corpus.
#
# BM25 does not represent a document as a dense vector.
#
# Instead, it evaluates lexical term matching while incorporating factors such
# as:
#
#     - term frequency
#     - inverse document frequency
#     - document length normalization
#
# =============================================================================

def build_bm25_retriever(tokenized_corpus):
    """Create an Okapi BM25 retriever using the common token representation."""

    return BM25Okapi(
        tokenized_corpus,
        k1=BM25_K1,
        b=BM25_B,
    )


# =============================================================================
# BLOCK 11 — BM25 RETRIEVAL
# =============================================================================
#
# PURPOSE
# -------
# Retrieve Top-K chunks using BM25.
#
# PROCESS
# -------
#
#     Query
#       ↓
#     tokenize query
#       ↓
#     BM25 lexical scoring against all chunks
#       ↓
#     descending score
#       ↓
#     Top-K
#
# The BM25 score scale is NOT directly comparable to the TF-IDF cosine score
# scale. We compare rankings and scoring behaviour, not raw numerical magnitude.
#
# =============================================================================

def retrieve_bm25(bm25, df, query, top_k=5):
    """Return Top-K BM25 results."""

    query_tokens = tokenize(query)

    scores = np.asarray(
        bm25.get_scores(query_tokens),
        dtype=float,
    )

    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, row_index in enumerate(ranked_indices, start=1):

        row_index = int(row_index)

        results.append(
            {
                "rank": rank,
                "chunk_id": int(df.iloc[row_index]["chunk_id"]),
                "score": float(scores[row_index]),
                "header": df.iloc[row_index]["header"],
                "text": df.iloc[row_index]["text"],
            }
        )

    return results


# =============================================================================
# BLOCK 12 — QUERY TERM DIAGNOSTICS
# =============================================================================
#
# PURPOSE
# -------
# Show the lexical terms that the query actually contributes to retrieval.
#
# For every unique query term we report:
#
#     Query TF
#     Corpus DF
#     IDF
#     Number of chunks containing the term
#
# This is one of the main learning outputs for Phase 1.
#
# =============================================================================

def print_query_term_diagnostics(
    query,
    tokenized_corpus,
    document_frequency,
):
    """Print TF/DF/IDF information for the query terms."""

    query_tokens = tokenize(query)
    query_tf = Counter(query_tokens)

    document_count = len(tokenized_corpus)

    print("\n")
    print("=" * 100)
    print("QUERY TERM DIAGNOSTICS")
    print("=" * 100)

    print(f"\nQuery: {query}")
    print(f"Tokenized query: {query_tokens}")

    print(
        f"\n{'TERM':<25}"
        f"{'QUERY TF':>12}"
        f"{'CORPUS DF':>14}"
        f"{'IDF':>14}"
        f"{'CHUNKS CONTAINING TERM':>26}"
    )

    print("-" * 100)

    for term in sorted(query_tf):

        df_value = document_frequency.get(term, 0)

        idf_value = calculate_sklearn_idf(
            document_count,
            df_value,
        )

        print(
            f"{term:<25}"
            f"{query_tf[term]:>12}"
            f"{df_value:>14}"
            f"{idf_value:>14.6f}"
            f"{df_value:>26}"
        )


# =============================================================================
# BLOCK 13 — INSPECT THE POSITIONAL INVERTED INDEX
# =============================================================================
#
# PURPOSE
# -------
# Show where important query terms occur in the retrieved chunks.
#
# This allows us to inspect:
#
#     term -> chunk -> token positions
#
# Example:
#
#     manager -> chunk_0 -> [4]
#
# The positions are zero-based token positions.
#
# IMPORTANT
# ---------
# This is evidence about the INDEX REPRESENTATION.
#
# It does NOT mean that the TF-IDF or BM25 score automatically includes
# proximity information.
#
# =============================================================================

def print_position_diagnostics(
    query,
    retrieved_results,
    positional_index,
):
    """Print query-term positions inside retrieved chunks."""

    query_terms = sorted(set(tokenize(query)))

    print("\n")
    print("=" * 100)
    print("POSITION / INVERTED-INDEX DIAGNOSTICS")
    print("=" * 100)

    print(f"\nQuery terms: {query_terms}")

    for result in retrieved_results:

        chunk_id = result["chunk_id"]

        print(
            f"\nRank {result['rank']} | "
            f"Chunk {chunk_id}"
        )

        found_any = False

        for term in query_terms:

            positions = positional_index.get(term, {}).get(
                chunk_id,
                [],
            )

            if positions:

                found_any = True

                print(
                    f"  {term:<25} positions={positions}"
                )

        if not found_any:
            print("  No exact query terms found in this chunk.")


# =============================================================================
# BLOCK 14 — PRINT RETRIEVAL RESULTS
# =============================================================================
#
# PURPOSE
# -------
# Produce a common human-readable output for TF-IDF and BM25.
#
# Each result contains:
#
#     Rank
#     Chunk ID
#     Score
#     Header
#     Text
#
# The format is intentionally verbose because the output will later be taken
# into the main Retrieval Architecture thread for interpretation.
#
# =============================================================================

def print_results(method_name, query_name, query, results):
    """Print Top-K retrieval results."""

    print("\n")
    print("=" * 100)
    print(f"{method_name} — {query_name}")
    print("=" * 100)

    print(f"\nQuery: {query}")
    print(f"Top-K: {TOP_K}")

    for result in results:

        print("\n" + "-" * 100)

        print(f"Rank       : {result['rank']}")
        print(f"Chunk ID   : {result['chunk_id']}")
        print(f"Score      : {result['score']:.8f}")
        print(f"Header     : {result['header']}")
        print(f"Text       : {result['text']}")


# =============================================================================
# BLOCK 15 — COMPARE THE TWO RANKINGS
# =============================================================================
#
# PURPOSE
# -------
# Provide a simple structural comparison after both methods have retrieved
# their Top-K results.
#
# We intentionally do NOT interpret the result.
#
# The output simply records:
#
#     - which chunk IDs appear in both Top-K sets
#     - which are unique to TF-IDF
#     - which are unique to BM25
#
# This gives us raw evidence to discuss later.
#
# =============================================================================

def print_ranking_comparison(tfidf_results, bm25_results):
    """Print overlap and rank information without interpreting it."""

    tfidf_ids = [item["chunk_id"] for item in tfidf_results]
    bm25_ids = [item["chunk_id"] for item in bm25_results]

    common_ids = set(tfidf_ids) & set(bm25_ids)

    tfidf_only = set(tfidf_ids) - set(bm25_ids)
    bm25_only = set(bm25_ids) - set(tfidf_ids)

    print("\n")
    print("=" * 100)
    print("TF-IDF vs BM25 — RAW RANKING COMPARISON")
    print("=" * 100)

    print(f"\nTF-IDF Top-K IDs : {tfidf_ids}")
    print(f"BM25 Top-K IDs   : {bm25_ids}")

    print(f"\nCommon Top-K IDs : {sorted(common_ids)}")
    print(f"TF-IDF only      : {sorted(tfidf_only)}")
    print(f"BM25 only        : {sorted(bm25_only)}")

    print("\nRank-by-rank comparison:")

    print(
        f"{'RANK':<8}"
        f"{'TF-IDF CHUNK':<18}"
        f"{'TF-IDF SCORE':<18}"
        f"{'BM25 CHUNK':<18}"
        f"{'BM25 SCORE':<18}"
    )

    print("-" * 80)

    for rank in range(TOP_K):

        tfidf_item = tfidf_results[rank]
        bm25_item = bm25_results[rank]

        print(
            f"{rank + 1:<8}"
            f"{tfidf_item['chunk_id']:<18}"
            f"{tfidf_item['score']:<18.8f}"
            f"{bm25_item['chunk_id']:<18}"
            f"{bm25_item['score']:<18.8f}"
        )


# =============================================================================
# BLOCK 16 — RUN ONE QUERY THROUGH BOTH METHODS
# =============================================================================
#
# PURPOSE
# -------
# Keep the experimental sequence explicit.
#
# For each query:
#
#     1. print query term statistics
#     2. retrieve with TF-IDF
#     3. retrieve with BM25
#     4. inspect positions for TF-IDF results
#     5. inspect positions for BM25 results
#     6. print both rankings
#     7. print raw ranking overlap
#
# No interpretation is performed here.
#
# =============================================================================

def run_query(
    query_name,
    query,
    df,
    tokenized_corpus,
    document_frequency,
    positional_index,
    tfidf_vectorizer,
    tfidf_matrix,
    bm25,
):
    """Run one frozen query through both lexical retrievers."""

    print("\n\n")
    print("#" * 100)
    print(f"EXPERIMENT: {query_name}")
    print("#" * 100)

    print_query_term_diagnostics(
        query=query,
        tokenized_corpus=tokenized_corpus,
        document_frequency=document_frequency,
    )

    # -------------------------------------------------------------------------
    # TF-IDF RETRIEVAL
    # -------------------------------------------------------------------------

    tfidf_results = retrieve_tfidf(
        vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        df=df,
        query=query,
        top_k=TOP_K,
    )

    print_results(
        method_name="TF-IDF",
        query_name=query_name,
        query=query,
        results=tfidf_results,
    )

    print_position_diagnostics(
        query=query,
        retrieved_results=tfidf_results,
        positional_index=positional_index,
    )

    # -------------------------------------------------------------------------
    # BM25 RETRIEVAL
    # -------------------------------------------------------------------------

    bm25_results = retrieve_bm25(
        bm25=bm25,
        df=df,
        query=query,
        top_k=TOP_K,
    )

    print_results(
        method_name="BM25",
        query_name=query_name,
        query=query,
        results=bm25_results,
    )

    print_position_diagnostics(
        query=query,
        retrieved_results=bm25_results,
        positional_index=positional_index,
    )

    # -------------------------------------------------------------------------
    # RAW COMPARISON
    # -------------------------------------------------------------------------

    print_ranking_comparison(
        tfidf_results=tfidf_results,
        bm25_results=bm25_results,
    )


# =============================================================================
# BLOCK 17 — MAIN EXPERIMENT DRIVER
# =============================================================================
#
# PURPOSE
# -------
# Assemble all components and run the two frozen queries.
#
# EXPERIMENTAL FLOW
# -----------------
#
#     chunk_corpus.csv
#           ↓
#     validate corpus
#           ↓
#     common tokenization
#           ↓
#       ┌───────────────┐
#       │               │
#     TF-IDF          BM25
#       │               │
#       └───────┬───────┘
#               ↓
#          same queries
#               ↓
#          Top-5 results
#               ↓
#       diagnostics/raw output
#
# =============================================================================

def main():

    print("=" * 100)
    print("Retrieval Architecture Lab — Phase 1")
    print("TF-IDF vs BM25")
    print("=" * 100)

    print(f"\nCorpus path : {CSV_PATH}")
    print(f"Top-K      : {TOP_K}")
    print(f"BM25 k1    : {BM25_K1}")
    print(f"BM25 b     : {BM25_B}")

    # -------------------------------------------------------------------------
    # STEP 1 — LOAD COMMON CORPUS
    # -------------------------------------------------------------------------

    print("\nLoading corpus...")

    df = load_corpus()

    print(f"Chunks loaded: {len(df)}")
    print(
        f"Chunk ID range: "
        f"{df['chunk_id'].min()} -> {df['chunk_id'].max()}"
    )

    # -------------------------------------------------------------------------
    # STEP 2 — COMMON TOKENIZATION
    # -------------------------------------------------------------------------

    print("\nBuilding common token representation...")

    tokenized_corpus = build_tokenized_corpus(df)

    print("Common tokenization complete.")

    # -------------------------------------------------------------------------
    # STEP 3 — DOCUMENT FREQUENCY
    # -------------------------------------------------------------------------

    print("\nCalculating document frequency statistics...")

    document_frequency = calculate_document_frequency(
        tokenized_corpus
    )

    print(
        f"Unique vocabulary terms: "
        f"{len(document_frequency)}"
    )

    # -------------------------------------------------------------------------
    # STEP 4 — POSITIONAL INVERTED INDEX
    # -------------------------------------------------------------------------

    print("\nBuilding diagnostic positional inverted index...")

    positional_index = build_positional_inverted_index(
        df,
        tokenized_corpus,
    )

    print(
        f"Inverted-index terms: "
        f"{len(positional_index)}"
    )

    # -------------------------------------------------------------------------
    # STEP 5 — TF-IDF INDEX / REPRESENTATION
    # -------------------------------------------------------------------------

    print("\nBuilding TF-IDF representation...")

    tfidf_vectorizer, tfidf_matrix = build_tfidf_retriever(df)

    print(
        f"TF-IDF matrix shape: "
        f"{tfidf_matrix.shape}"
    )

    print(
        f"TF-IDF vocabulary size: "
        f"{len(tfidf_vectorizer.vocabulary_)}"
    )

    # -------------------------------------------------------------------------
    # STEP 6 — BM25 INDEX
    # -------------------------------------------------------------------------

    print("\nBuilding BM25 index...")

    bm25 = build_bm25_retriever(
        tokenized_corpus
    )

    print("BM25 index ready.")

    # -------------------------------------------------------------------------
    # STEP 7 — RUN THE TWO FROZEN QUERIES
    # -------------------------------------------------------------------------

    for query_name, query in QUERIES.items():

        run_query(
            query_name=query_name,
            query=query,
            df=df,
            tokenized_corpus=tokenized_corpus,
            document_frequency=document_frequency,
            positional_index=positional_index,
            tfidf_vectorizer=tfidf_vectorizer,
            tfidf_matrix=tfidf_matrix,
            bm25=bm25,
        )

    # -------------------------------------------------------------------------
    # STEP 8 — COMPLETION
    # -------------------------------------------------------------------------

    print("\n\n")
    print("=" * 100)
    print("PHASE 1 INITIAL TF-IDF / BM25 RUN COMPLETE")
    print("=" * 100)

    print(
        "\nRaw outputs above contain rankings, scores, term statistics, "
        "and positional evidence."
    )

    print(
        "\nNo architectural interpretation is performed by this script."
    )


# =============================================================================
# BLOCK 17 — DUAL OUTPUT: TERMINAL + RAW TEXT FILE
# =============================================================================
#
# PURPOSE
# -------
# Preserve the complete experimental output automatically.
#
# The experiment continues to print normally to the terminal, but everything
# printed by the script is also written to:
#
#     Phase1_TFIDF_BM25_RawOutput.txt
#
# This is useful because terminal output uses fixed-width spacing. Copying that
# output into Word can disturb alignment, especially for score tables.
#
# The TXT file is therefore the PRIMARY raw experimental record.
#
# The file is recreated on every run so that each run represents one clean
# experimental execution.
# =============================================================================

class TeeOutput:
    """Send printed output to both the terminal and a text file."""

    def __init__(self, terminal, file_handle):
        self.terminal = terminal
        self.file_handle = file_handle

    def write(self, text):
        self.terminal.write(text)
        self.file_handle.write(text)

    def flush(self):
        self.terminal.flush()
        self.file_handle.flush()


# =============================================================================
# BLOCK 18 — SCRIPT ENTRY POINT
# =============================================================================
#
# PURPOSE
# -------
# Run the experiment and preserve the exact console output simultaneously.
# =============================================================================

if __name__ == "__main__":

    import sys

    output_path = (
        Path(__file__).resolve().parent
        / "Phase1_TFIDF_BM25_RawOutput.txt"
    )

    print(f"Raw output will be saved to:\n{output_path}\n")

    with output_path.open("w", encoding="utf-8") as output_file:

        original_stdout = sys.stdout

        sys.stdout = TeeOutput(
            terminal=original_stdout,
            file_handle=output_file,
        )

        try:
            main()
        finally:
            sys.stdout = original_stdout

    print(
        "\nComplete raw experimental output saved to:\n"
        f"{output_path}"
    )
