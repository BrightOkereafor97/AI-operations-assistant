import re

from retrieve import retrieve


# ------------------------------------------------
# 1. DETECT MULTI-PART QUESTIONS
# ------------------------------------------------

def is_multi_part_question(question):

    cleaned = (
        question
        .strip()
        .lower()
    )


    # We currently look for a conjunction
    # connecting two question requirements.
    if ", and " in cleaned:

        return True


    return False


# ------------------------------------------------
# 2. SPLIT QUESTION INTO SUB-QUESTIONS
# ------------------------------------------------

def split_question(question):

    # Example:
    #
    # "How many annual leave days are provided,
    #  and how many remote-work days are allowed?"
    #
    # becomes:
    #
    # 1. How many annual leave days are provided?
    # 2. How many remote-work days are allowed?


    parts = re.split(
        r",\s+and\s+",
        question,
        maxsplit=1,
        flags=re.IGNORECASE
    )


    cleaned_parts = []


    for part in parts:

        part = part.strip()


        if not part:

            continue


        # Make each piece look like
        # a complete question.
        if not part.endswith("?"):

            part = (
                part.rstrip(".")
                + "?"
            )


        cleaned_parts.append(
            part
        )


    return cleaned_parts


# ------------------------------------------------
# 3. RETRIEVE FOR EACH SUB-QUESTION
# ------------------------------------------------

def retrieve_sub_questions(
    question,
    model,
    embeddings,
    metadata,
    top_k_per_part=5
):

    sub_questions = split_question(
        question
    )


    all_sub_results = []


    for sub_question in sub_questions:

        results, _ = retrieve(
            sub_question,
            model,
            embeddings,
            metadata,
            top_k=top_k_per_part
        )


        all_sub_results.append({

            "sub_question":
                sub_question,

            "results":
                results
        })


    return all_sub_results


# ------------------------------------------------
# 4. MERGE RESULTS WITH SOURCE DIVERSITY
# ------------------------------------------------

def merge_diverse_results(
    sub_results,
    final_top_k=5
):

    merged = []

    used_chunk_ids = set()


    # ------------------------------------------------
    # FIRST PASS
    #
    # Take the best result from each sub-question.
    #
    # This ensures each part gets representation.
    # ------------------------------------------------

    for item in sub_results:

        if not item["results"]:

            continue


        best_result = (
            item["results"][0]
            .copy()
        )


        chunk_id = (
            best_result[
                "chunk_id"
            ]
        )


        if chunk_id not in used_chunk_ids:

            best_result[
                "matched_sub_question"
            ] = item[
                "sub_question"
            ]


            merged.append(
                best_result
            )


            used_chunk_ids.add(
                chunk_id
            )


    # ------------------------------------------------
    # SECOND PASS
    #
    # Add remaining strong results.
    # ------------------------------------------------

    remaining_results = []


    for item in sub_results:

        for result in item[
            "results"
        ]:

            chunk_id = (
                result[
                    "chunk_id"
                ]
            )


            if chunk_id in used_chunk_ids:

                continue


            candidate = (
                result.copy()
            )


            candidate[
                "matched_sub_question"
            ] = item[
                "sub_question"
            ]


            remaining_results.append(
                candidate
            )


    # Highest similarity first
    remaining_results.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    for result in remaining_results:

        if len(merged) >= final_top_k:

            break


        chunk_id = (
            result["chunk_id"]
        )


        if chunk_id in used_chunk_ids:

            continue


        merged.append(
            result
        )


        used_chunk_ids.add(
            chunk_id
        )


    # ------------------------------------------------
    # REASSIGN FINAL RANKS
    # ------------------------------------------------

    for rank, result in enumerate(
        merged,
        start=1
    ):

        result["rank"] = rank


    return merged


# ------------------------------------------------
# 5. COMPLETE MULTI-PART RETRIEVAL
# ------------------------------------------------

def retrieve_multi_part(
    question,
    model,
    embeddings,
    metadata,
    final_top_k=5
):

    sub_results = (
        retrieve_sub_questions(
            question,
            model,
            embeddings,
            metadata
        )
    )


    merged_results = (
        merge_diverse_results(
            sub_results,
            final_top_k=final_top_k
        )
    )


    return (
        merged_results,
        sub_results
    )


# ------------------------------------------------
# 6. TEST SCRIPT
# ------------------------------------------------

if __name__ == "__main__":

    from retrieve import (
        load_embedding_model,
        load_vector_store
    )


    print(
        "\nLoading embedding model..."
    )


    model = (
        load_embedding_model()
    )


    embeddings, metadata = (
        load_vector_store(
            config_name="medium"
        )
    )


    question = (
        "How many annual leave days are provided, "
        "and how many remote-work days are "
        "currently allowed each week?"
    )


    print(
        "\nORIGINAL QUESTION:"
    )

    print(
        question
    )


    print(
        "\nMULTI-PART:",
        is_multi_part_question(
            question
        )
    )


    print(
        "\nSUB-QUESTIONS:"
    )


    for part in split_question(
        question
    ):

        print(
            "-",
            part
        )


    results, sub_results = (
        retrieve_multi_part(
            question,
            model,
            embeddings,
            metadata,
            final_top_k=5
        )
    )


    print(
        "\nFINAL MERGED RESULTS"
    )

    print(
        "=" * 70
    )


    for result in results:

        print(
            f"\nRank: "
            f"{result['rank']}"
        )

        print(
            "Document:",
            result[
                "document_name"
            ]
        )

        print(
            "Chunk:",
            result[
                "chunk_id"
            ]
        )

        print(
            "Similarity:",
            f"{result['score']:.4f}"
        )

        print(
            "Matched part:",
            result[
                "matched_sub_question"
            ]
        )

        print(
            "-" * 70
        )