from tools import search_company_knowledge


# ================================================================
# STAGE 4 — RAG TOOL BASELINE AUDIT
#
# Goal:
# Inspect what the current Project 2 RAG tool returns BEFORE
# modifying it for Stage 4.
# ================================================================


TEST_QUESTIONS = [

    {
        "id": "RAG-001",
        "type": "answerable",
        "question":
            "How many annual leave days do full-time employees receive?"
    },

    {
        "id": "RAG-002",
        "type": "answerable",
        "question":
            "What does the company policy say about remote work?"
    },

    {
        "id": "RAG-003",
        "type": "answerable",
        "question":
            "What is the company refund policy?"
    },

    {
        "id": "RAG-004",
        "type": "answerable",
        "question":
            "What are the rules for employee travel expenses?"
    },

    {
        "id": "RAG-005",
        "type": "answerable",
        "question":
            "Tell me the company's escalation policy."
    },

    {
        "id": "RAG-006",
        "type": "answerable",
        "question":
            "What does our policy say about sick leave?"
    },

    {
        "id": "RAG-007",
        "type": "unsupported",
        "question":
            "What is the company policy for employees bringing pets to work?"
    },

    {
        "id": "RAG-008",
        "type": "unsupported",
        "question":
            "What colour should employees paint their home office?"
    },
]


# ================================================================
# DISPLAY HELPER
# ================================================================

def print_field(
    name,
    value
):

    print(
        f"{name}: {value}"
    )


# ================================================================
# AUDIT RESULT
# ================================================================

def audit_result(
    test,
    result
):

    print(
        "\n"
        + "=" * 72
    )

    print(
        f"{test['id']} — {test['type']}"
    )

    print(
        f"Question: {test['question']}"
    )

    print(
        "-" * 72
    )


    # ------------------------------------------------------------
    # Make sure the tool returned a dictionary.
    # ------------------------------------------------------------

    if not isinstance(
        result,
        dict
    ):

        print(
            "ERROR: RAG tool did not return a dictionary."
        )

        print(
            "Raw result:",
            result
        )

        return


    # ------------------------------------------------------------
    # CURRENT FIELDS
    # ------------------------------------------------------------

    print_field(
        "Answer",
        result.get(
            "answer"
        )
    )

    print_field(
        "Sources",
        result.get(
            "sources"
        )
    )

    print_field(
        "Retrieval mode",
        result.get(
            "retrieval_mode"
        )
    )

    print_field(
        "Conflict",
        result.get(
            "conflict"
        )
    )


    # ------------------------------------------------------------
    # STAGE 4 REQUIRED / DESIRED EVIDENCE FIELDS
    #
    # These may currently be missing.
    #
    # That is exactly what this baseline test is trying to prove.
    # ------------------------------------------------------------

    print(
        "\nSTAGE 4 EVIDENCE CHECK"
    )

    print(
        "-" * 72
    )


    desired_fields = [

        "chunks",

        "evidence",

        "retrieval_scores",

        "abstained",
    ]


    for field in desired_fields:

        if field in result:

            print(
                f"{field}: PRESENT ✅"
            )

            print(
                f"  value: {result[field]}"
            )

        else:

            print(
                f"{field}: MISSING ❌"
            )


# ================================================================
# MAIN
# ================================================================

def main():

    print(
        "\nSTAGE 4 — CURRENT RAG TOOL BASELINE"
    )

    print(
        "=" * 72
    )

    print(
        "Purpose:"
    )

    print(
        "Inspect the current Project 2 RAG tool before changing it."
    )

    print(
        "\nQuestions:",
        len(
            TEST_QUESTIONS
        )
    )


    for test in TEST_QUESTIONS:

        try:

            result = (
                search_company_knowledge(
                    test[
                        "question"
                    ]
                )
            )


            audit_result(
                test,
                result
            )


        except Exception as error:

            print(
                "\n"
                + "=" * 72
            )

            print(
                f"{test['id']} FAILED"
            )

            print(
                f"Question: {test['question']}"
            )

            print(
                f"Error: {type(error).__name__}: {error}"
            )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "BASELINE AUDIT COMPLETE"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":

    main()