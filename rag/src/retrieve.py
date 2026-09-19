import csv
from pathlib import Path

import numpy as np

from sentence_transformers import SentenceTransformer


# ------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDINGS_FOLDER = (
    PROJECT_ROOT
    / "results"
    / "embeddings"
)


# ------------------------------------------------
# 2. BASELINE SETTINGS
# ------------------------------------------------

MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

MODEL_LABEL = "minilm_l6"

DEFAULT_CONFIG = "medium"

DEFAULT_TOP_K = 5


# ------------------------------------------------
# 3. LOAD STORED VECTORS
# ------------------------------------------------

def load_vector_store(
    config_name=DEFAULT_CONFIG
):

    vector_file = (
        EMBEDDINGS_FOLDER
        / f"{MODEL_LABEL}_{config_name}.npy"
    )


    metadata_file = (
        EMBEDDINGS_FOLDER
        / (
            f"{MODEL_LABEL}_"
            f"{config_name}_metadata.csv"
        )
    )


    if not vector_file.exists():

        raise FileNotFoundError(
            f"Vector file not found: "
            f"{vector_file}"
        )


    if not metadata_file.exists():

        raise FileNotFoundError(
            f"Metadata file not found: "
            f"{metadata_file}"
        )


    embeddings = np.load(
        vector_file
    )


    metadata = []


    with metadata_file.open(
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(
            file
        )


        for row in reader:

            metadata.append(
                row
            )


    if len(embeddings) != len(metadata):

        raise ValueError(
            "Vector count and metadata "
            "count do not match."
        )


    return embeddings, metadata


# ------------------------------------------------
# 4. LOAD EMBEDDING MODEL
# ------------------------------------------------

def load_embedding_model():

    model = SentenceTransformer(
        MODEL_NAME
    )

    return model


# ------------------------------------------------
# 5. EMBED USER QUESTION
# ------------------------------------------------

def embed_query(
    question,
    model
):

    query_vector = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )


    return query_vector


# ------------------------------------------------
# 6. RETRIEVE TOP-K CHUNKS
# ------------------------------------------------

def retrieve(
    question,
    model,
    embeddings,
    metadata,
    top_k=DEFAULT_TOP_K
):

    query_vector = embed_query(
        question,
        model
    )


    # --------------------------------------------
    # Because both stored vectors and query vector
    # are normalized, dot product gives cosine
    # similarity.
    # --------------------------------------------

    scores = embeddings @ query_vector


    ranked_indexes = np.argsort(
        scores
    )[::-1]


    top_indexes = ranked_indexes[
        :top_k
    ]


    results = []


    for rank, index in enumerate(
        top_indexes,
        start=1
    ):

        item = metadata[index].copy()


        item["rank"] = rank

        item["score"] = float(
            scores[index]
        )


        results.append(
            item
        )


    return (
        results,
        query_vector
    )


# ------------------------------------------------
# 7. DISPLAY RESULTS
# ------------------------------------------------

def display_results(
    question,
    results,
    query_vector
):

    print("\n" + "=" * 70)

    print("SEMANTIC RETRIEVAL")

    print("=" * 70)


    print(
        f"\nQuestion:\n{question}"
    )


    print(
        f"\nQuestion vector shape: "
        f"{query_vector.shape}"
    )


    print(
        "\nTOP RETRIEVED CHUNKS"
    )

    print("-" * 70)


    for result in results:

        print(
            f"\nRank: "
            f"{result['rank']}"
        )

        print(
            f"Similarity: "
            f"{result['score']:.4f}"
        )

        print(
            f"Document: "
            f"{result['document_name']}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print(
            f"Version: "
            f"{result['version']}"
        )

        print(
            f"Status: "
            f"{result['status']}"
        )

        print(
            f"Text:\n"
            f"{result['embedding_text']}"
        )

        print("-" * 70)


# ------------------------------------------------
# 8. INTERACTIVE TEST
# ------------------------------------------------

if __name__ == "__main__":

    print("\nLoading retrieval system...")


    model = load_embedding_model()


    (
        embeddings,
        metadata

    ) = load_vector_store(
        config_name=DEFAULT_CONFIG
    )


    print(
        f"Loaded {len(metadata)} "
        f"chunks."
    )


    print(
        f"Vector matrix shape: "
        f"{embeddings.shape}"
    )


    while True:

        print("\n" + "=" * 70)

        question = input(
            "\nAsk a question "
            "(or type 'quit'): "
        ).strip()


        if question.lower() in [
            "quit",
            "exit"
        ]:

            print(
                "\nRetrieval test closed."
            )

            break


        if not question:

            print(
                "Please enter a question."
            )

            continue


        (
            results,
            query_vector

        ) = retrieve(
            question,
            model,
            embeddings,
            metadata
        )


        display_results(
            question,
            results,
            query_vector
        )