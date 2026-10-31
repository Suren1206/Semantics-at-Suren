"""
================================================================================
Semantics @ Suren — Module 4
GENERAL-CENTRIC LOCAL + CROSS-GROUP COSINE ANALYSIS
================================================================================

For every chunk belonging to 5.General:

    1. Print the reference General chunk.
    2. Find its top 5 nearest chunks WITHIN 5.General.
    3. Find its top 5 nearest chunks OUTSIDE 5.General.
    4. Repeat for all General chunks.

The complete output is written to a text file so that no terminal-output
limit causes information to be lost.

No mean-centering.
No PCA.
No whitening.
No dimensionality reduction.
Original embeddings are not modified.
================================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# The script is stored in:
#
# S@S_Module 4  Experiments/
#     Codes_Module_4/
#
# Therefore PROJECT_DIR is automatically identified as the parent directory.

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent

CORPUS_FILE = PROJECT_DIR / "Synthetic_Corpus.csv"

EMBEDDING_FILE = (
    PROJECT_DIR
    / "Output_Files"
    / "synthetic_raw_embeddings.npy"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "Output_Files"
    / "synthetic_top5_GeneralvsRest_results.txt"
)


# =============================================================================
# SETTINGS
# =============================================================================

TARGET_GROUP = "5.General"
TOP_K = 5


# =============================================================================
# OUTPUT BUFFER
# =============================================================================

output = []


def write(line=""):
    """
    Add a line to the output buffer.
    """
    output.append(str(line))


def separator(character="=", length=100):
    write(character * length)


# =============================================================================
# RECORD DISPLAY
# =============================================================================

def write_record(index, label):
    """
    Write complete corpus information for a vector.
    """

    separator("-")

    write(label)

    separator("-")

    write(
        f"Embedding index : {index}"
    )

    # CSV physical row:
    # Row 1 = header
    # Embedding index 0 = CSV row 2

    write(
        f"CSV row         : {index + 2}"
    )

    write(
        f"Column A        : {groups.iloc[index]}"
    )

    write(
        f"Column B        : {metadata_b.iloc[index]}"
    )

    write(
        f"Column C        : {metadata_c.iloc[index]}"
    )

    write(
        f"Column D        : {content.iloc[index]}"
    )


# =============================================================================
# FIND NEIGHBOURS
# =============================================================================

def find_neighbours(reference_index, candidate_indices):
    """
    Calculate cosine similarity between one reference vector and all
    candidate vectors.

    Returns the top TOP_K candidates in descending cosine order.
    """

    reference_vector = normalized_embeddings[
        reference_index
    ]

    candidate_vectors = normalized_embeddings[
        candidate_indices
    ]

    cosine_scores = candidate_vectors @ reference_vector

    ranking = np.argsort(cosine_scores)[::-1]

    results = []

    for position in ranking[:TOP_K]:

        neighbour_index = candidate_indices[position]

        score = cosine_scores[position]

        results.append(
            (
                int(neighbour_index),
                float(score)
            )
        )

    return results


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():

    global groups
    global metadata_b
    global metadata_c
    global content
    global normalized_embeddings


    # -------------------------------------------------------------------------
    # HEADER
    # -------------------------------------------------------------------------

    separator()

    write(
        "Semantics @ Suren — Module 4"
    )

    write(
        "GENERAL-CENTRIC LOCAL + CROSS-GROUP COSINE ANALYSIS"
    )

    separator()

    write()

    write(
        f"Corpus     : {CORPUS_FILE}"
    )

    write(
        f"Embeddings : {EMBEDDING_FILE}"
    )

    write(
        f"Target     : {TARGET_GROUP}"
    )


    # -------------------------------------------------------------------------
    # CHECK FILES
    # -------------------------------------------------------------------------

    if not CORPUS_FILE.exists():

        raise FileNotFoundError(
            f"\nCorpus file not found:\n{CORPUS_FILE}"
        )

    if not EMBEDDING_FILE.exists():

        raise FileNotFoundError(
            f"\nEmbedding file not found:\n{EMBEDDING_FILE}"
        )


    # -------------------------------------------------------------------------
    # LOAD CORPUS
    # -------------------------------------------------------------------------

    write()
    separator()
    write("LOADING CORPUS")
    separator()

    df = pd.read_csv(
        CORPUS_FILE,
        header=0
    )

    write()

    write(
        f"Corpus shape : {df.shape}"
    )


    if df.shape[1] < 4:

        raise ValueError(
            "CSV must contain at least four columns."
        )


    # -------------------------------------------------------------------------
    # LOAD EMBEDDINGS
    # -------------------------------------------------------------------------

    write()
    separator()
    write("LOADING EMBEDDINGS")
    separator()

    embeddings = np.load(
        EMBEDDING_FILE
    )

    write()

    write(
        f"Embedding shape : {embeddings.shape}"
    )


    if embeddings.ndim != 2:

        raise ValueError(
            f"Expected a 2-D embedding matrix. "
            f"Found: {embeddings.shape}"
        )


    number_of_vectors, dimensions = embeddings.shape


    # -------------------------------------------------------------------------
    # VERIFY ALIGNMENT
    # -------------------------------------------------------------------------

    if len(df) != number_of_vectors:

        raise ValueError(
            "\nMismatch between corpus rows and embeddings.\n"
            f"Corpus rows : {len(df)}\n"
            f"Embeddings  : {number_of_vectors}\n\n"
            "Every corpus data row must correspond to exactly one embedding."
        )


    write()
    write(
        "CSV rows and embedding rows are aligned."
    )


    # -------------------------------------------------------------------------
    # EXTRACT COLUMNS
    # -------------------------------------------------------------------------

    # Column A = group
    # Column B = metadata
    # Column C = metadata
    # Column D = actual content

    groups = (
        df.iloc[:, 0]
        .astype(str)
        .str.strip()
    )

    metadata_b = df.iloc[:, 1]

    metadata_c = df.iloc[:, 2]

    content = (
        df.iloc[:, 3]
        .astype(str)
    )


    # -------------------------------------------------------------------------
    # IDENTIFY GENERAL AND OUTSIDE-GENERAL VECTORS
    # -------------------------------------------------------------------------

    general_indices = np.where(
        groups.values == TARGET_GROUP
    )[0]

    outside_indices = np.where(
        groups.values != TARGET_GROUP
    )[0]


    write()
    separator()
    write("COMPARISON POPULATION")
    separator()

    write()

    write(
        f"General chunks       : {len(general_indices)}"
    )

    write(
        f"Outside-General      : {len(outside_indices)}"
    )

    write(
        f"Embedding dimensions : {dimensions}"
    )


    if len(general_indices) == 0:

        raise ValueError(
            f"No chunks found with Column A = '{TARGET_GROUP}'."
        )


    if len(outside_indices) == 0:

        raise ValueError(
            "No vectors exist outside the General category."
        )


    # -------------------------------------------------------------------------
    # NORMALIZE FOR COSINE CALCULATION
    # -------------------------------------------------------------------------

    # This creates a separate normalized copy.
    # The original embeddings remain unchanged.

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )


    if np.any(norms == 0):

        zero_indices = np.where(
            norms.flatten() == 0
        )[0]

        raise ValueError(
            f"Zero-norm embeddings found at indices: "
            f"{zero_indices.tolist()}"
        )


    normalized_embeddings = (
        embeddings / norms
    )


    # -------------------------------------------------------------------------
    # PROCESS EACH GENERAL CHUNK
    # -------------------------------------------------------------------------

    for reference_number, reference_index in enumerate(
        general_indices,
        start=1
    ):

        write()
        write()
        write(
            "#" * 100
        )

        write(
            f"GENERAL REFERENCE {reference_number} "
            f"OF {len(general_indices)}"
        )

        write(
            "#" * 100
        )


        # =====================================================================
        # REFERENCE CHUNK
        # =====================================================================

        write()

        write_record(
            reference_index,
            "REFERENCE GENERAL CHUNK"
        )


        # =====================================================================
        # WITHIN-GENERAL
        # =====================================================================

        same_group_candidates = general_indices[
            general_indices != reference_index
        ]


        within_results = find_neighbours(
            reference_index,
            same_group_candidates
        )


        write()
        separator()

        write(
            "TOP 5 WITHIN-GENERAL CHUNKS"
        )

        separator()

        write()

        write(
            f"{'Rank':<8}"
            f"{'Embedding Idx':<16}"
            f"{'Cosine':<15}"
            f"{'CSV Row':<10}"
            f"{'Group':<35}"
        )

        write("-" * 100)


        for rank, (index, score) in enumerate(
            within_results,
            start=1
        ):

            write(
                f"{rank:<8}"
                f"{index:<16}"
                f"{score:<15.8f}"
                f"{index + 2:<10}"
                f"{groups.iloc[index]:<35}"
            )


        # ---------------------------------------------------------------------
        # Full within-General records
        # ---------------------------------------------------------------------

        write()

        for rank, (index, score) in enumerate(
            within_results,
            start=1
        ):

            write()

            write(
                f"WITHIN-GENERAL RANK {rank}"
            )

            write(
                f"Cosine similarity : {score:.8f}"
            )

            write_record(
                index,
                "MATCHING GENERAL CHUNK"
            )


        # =====================================================================
        # OUTSIDE-GENERAL
        # =====================================================================

        outside_results = find_neighbours(
            reference_index,
            outside_indices
        )


        write()
        separator()

        write(
            "TOP 5 OUTSIDE-GENERAL CHUNKS"
        )

        separator()

        write()

        write(
            f"{'Rank':<8}"
            f"{'Embedding Idx':<16}"
            f"{'Cosine':<15}"
            f"{'CSV Row':<10}"
            f"{'Group':<35}"
        )

        write("-" * 100)


        for rank, (index, score) in enumerate(
            outside_results,
            start=1
        ):

            write(
                f"{rank:<8}"
                f"{index:<16}"
                f"{score:<15.8f}"
                f"{index + 2:<10}"
                f"{groups.iloc[index]:<35}"
            )


        # ---------------------------------------------------------------------
        # Full outside-General records
        # ---------------------------------------------------------------------

        write()

        for rank, (index, score) in enumerate(
            outside_results,
            start=1
        ):

            write()

            write(
                f"OUTSIDE-GENERAL RANK {rank}"
            )

            write(
                f"Cosine similarity : {score:.8f}"
            )

            write_record(
                index,
                "MATCHING OUTSIDE-GENERAL CHUNK"
            )


    # -------------------------------------------------------------------------
    # FOOTER
    # -------------------------------------------------------------------------

    write()
    write()
    separator()

    write(
        "EXPERIMENT COMPLETE"
    )

    separator()

    write()

    write(
        f"Analyzed all {len(general_indices)} "
        f"chunks belonging to {TARGET_GROUP}."
    )

    write()

    write(
        "For each General chunk:"
    )

    write(
        "  • Reference chunk was displayed."
    )

    write(
        "  • Top 5 within-General neighbours were calculated."
    )

    write(
        "  • Top 5 outside-General neighbours were calculated."
    )

    write()

    write(
        "Column A = grouping metadata"
    )

    write(
        "Column B = metadata"
    )

    write(
        "Column C = metadata"
    )

    write(
        "Column D = embedded content"
    )

    write()

    write(
        "No mean-centering was applied."
    )

    write(
        "No PCA was applied."
    )

    write(
        "No whitening was applied."
    )

    write(
        "No dimensionality reduction was applied."
    )

    write(
        "Original embeddings were not modified."
    )


    # -------------------------------------------------------------------------
    # SAVE COMPLETE OUTPUT
    # -------------------------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(output)
        )

        f.write("\n")


    # -------------------------------------------------------------------------
    # TERMINAL MESSAGE
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("EXPERIMENT COMPLETE")
    print("=" * 100)

    print()
    print("The complete analysis has been saved to:")

    print()
    print(OUTPUT_FILE)

    print()
    print("Open the text file to view the complete results.")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()