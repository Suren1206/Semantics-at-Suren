from pathlib import Path
import sys

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_DIR = Path(
    "/home/surendran/LangChain/Semantics@Suren/"
    "S@S_Module 1 Experiments "
)

sys.path.insert(0, str(PROJECT_DIR / "raGROW_S@S"))

from rag_engine.retriever import Retriever


# ---------------------------------------------------------
# Experiment
# ---------------------------------------------------------
QUERY = "When can self-certification be used for illness absence?"
VERSION_ID = "20260424_B01"


def main():

    print("=" * 80)
    print("mxbai-embed-large — Original Query")
    print("=" * 80)

    print(f"\nQuery: {QUERY}")
    print(f"Version: {VERSION_ID}")
    print("Model: mxbai-embed-large:335m")
    print("Top-K: 5")

    # Use the existing Retriever implementation unchanged
    retriever = Retriever(version_id=VERSION_ID)

    result = retriever.retrieve(
        query=QUERY,
        top_k=5,
        score_threshold=0.0,
        return_metadata=True
    )

    print("\n" + "-" * 80)
    print("TOP-5 RESULTS")
    print("-" * 80)

    for item in result["results"]:

        print(f"\nRank      : {item['rank']}")
        print(f"Chunk ID   : {item['chunk_id']}")
        print(f"Chunk Index: {item['chunk_index']}")
        print(f"Cosine     : {item['score']:.6f}")
        print(f"Header     : {item['metadata'].get('header', '')}")
        print(f"Text       : {item['text']}")

    print("\n" + "=" * 80)
    print("END")
    print("=" * 80)


if __name__ == "__main__":
    main()