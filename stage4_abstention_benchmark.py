import csv

from pathlib import Path

from tools import (
    search_company_knowledge
)


# ================================================================
# STAGE 4 — ABSTENTION BENCHMARK
#
# PURPOSE:
#
# Test whether the current RAG system knows when
# company documents contain enough evidence to answer.
#
# IMPORTANT:
#
# We are NOT changing the threshold in this experiment.
# We are measuring the CURRENT behaviour first.
# ================================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

RESULTS_DIR.mkdir(
    exist_ok=True
)


DETAIL_FILE = (
    RESULTS_DIR
    / "stage4_abstention_baseline.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "stage4_abstention_summary.csv"
)


# ================================================================
# FIXED TEST SET
#
# expected_abstained:
#
# False = company documents should contain an answer
# True  = system should say there is not enough evidence
# ================================================================

TEST_QUESTIONS = [

    # ------------------------------------------------------------
    # ANSWERABLE QUESTIONS
    # ------------------------------------------------------------

    {
        "id": "S4-001",
        "category": "answerable",
        "question":
            "How many annual leave days do full-time employees receive?",
        "expected_abstained": False
    },

    {
        "id": "S4-002",
        "category": "answerable",
        "question":
            "What does the company policy say about remote work?",
        "expected_abstained": False
    },

    {
        "id": "S4-003",
        "category": "answerable",
        "question":
            "What is the company refund policy?",
        "expected_abstained": False
    },

    {
        "id": "S4-004",
        "category": "answerable",
        "question":
            "What are the rules for employee travel expenses?",
        "expected_abstained": False
    },

    {
        "id": "S4-005",
        "category": "answerable",
        "question":
            "Tell me the company's escalation policy.",
        "expected_abstained": False
    },

    {
        "id": "S4-006",
        "category": "answerable",
        "question":
            "What does our policy say about sick leave?",
        "expected_abstained": False
    },

    {
        "id": "S4-007",
        "category": "answerable",
        "question":
            "What is the process for requesting annual leave?",
        "expected_abstained": False
    },

    {
        "id": "S4-008",
        "category": "answerable",
        "question":
            "According to company policy, how should customer refunds be handled?",
        "expected_abstained": False
    },


    # ------------------------------------------------------------
    # UNSUPPORTED QUESTIONS
    #
    # These topics should NOT exist in our fictional company
    # knowledge base.
    # ------------------------------------------------------------

    {
        "id": "S4-009",
        "category": "unsupported",
        "question":
            "What is the company policy for employees bringing pets to work?",
        "expected_abstained": True
    },

    {
        "id": "S4-010",
        "category": "unsupported",
        "question":
            "What colour should employees paint their home office?",
        "expected_abstained": True
    },

    {
        "id": "S4-011",
        "category": "unsupported",
        "question":
            "Does the company provide free gym memberships?",
        "expected_abstained": True
    },

    {
        "id": "S4-012",
        "category": "unsupported",
        "question":
            "Can employees bring bicycles into the office?",
        "expected_abstained": True
    },

    {
        "id": "S4-013",
        "category": "unsupported",
        "question":
            "What is the dress code for company parties?",
        "expected_abstained": True
    },

    {
        "id": "S4-014",
        "category": "unsupported",
        "question":
            "Does the company reimburse employees for childcare expenses?",
        "expected_abstained": True
    },

    {
        "id": "S4-015",
        "category": "unsupported",
        "question":
            "What is the company policy on employee parking spaces?",
        "expected_abstained": True
    },

    {
        "id": "S4-016",
        "category": "unsupported",
        "question":
            "Does the company give employees free lunch every Friday?",
        "expected_abstained": True
    },
]


# ================================================================
# HELPER — FIRST RETRIEVAL DETAIL
# ================================================================

def get_first_retrieval_detail(
    result
):

    details = result.get(
        "retrieval_details",
        []
    )

    if not details:

        return {}

    return details[0]


# ================================================================
# HELPER — TOP EVIDENCE SCORE
# ================================================================

def get_top_evidence_score(
    result
):

    scores = result.get(
        "retrieval_scores",
        []
    )

    numeric_scores = []

    for score in scores:

        try:

            numeric_scores.append(
                float(score)
            )

        except (
            TypeError,
            ValueError
        ):

            continue

    if not numeric_scores:

        return None

    return max(
        numeric_scores
    )


# ================================================================
# RUN ONE TEST
# ================================================================

def run_test(
    test
):

    result = (
        search_company_knowledge(
            test[
                "question"
            ]
        )
    )

    actual_abstained = bool(
        result.get(
            "abstained",
            False
        )
    )

    expected_abstained = (
        test[
            "expected_abstained"
        ]
    )

    correct = (
        actual_abstained
        ==
        expected_abstained
    )

    top_score = (
        get_top_evidence_score(
            result
        )
    )

    detail = (
        get_first_retrieval_detail(
            result
        )
    )

    sources = result.get(
        "sources",
        []
    )

    chunks = result.get(
        "chunks",
        []
    )

    evidence = result.get(
        "evidence",
        []
    )


    row = {

        "id":
            test[
                "id"
            ],

        "category":
            test[
                "category"
            ],

        "question":
            test[
                "question"
            ],

        "expected_abstained":
            expected_abstained,

        "actual_abstained":
            actual_abstained,

        "abstention_correct":
            correct,

        "answer":
            result.get(
                "answer",
                ""
            ),

        "source":
            (
                sources[0]
                if sources
                else ""
            ),

        "chunk":
            (
                chunks[0]
                if chunks
                else ""
            ),

        "focused_evidence":
            (
                evidence[0]
                if evidence
                else ""
            ),

        "top_evidence_score":
            top_score,

        "chunk_similarity":
            detail.get(
                "chunk_similarity"
            ),

        "sentence_similarity":
            detail.get(
                "sentence_similarity"
            ),

        "adjusted_evidence_score":
            detail.get(
                "adjusted_evidence_score"
            ),

        "retrieval_mode":
            result.get(
                "retrieval_mode",
                ""
            ),

        "conflict":
            result.get(
                "conflict"
            )
    }

    return row


# ================================================================
# DISPLAY ONE TEST
# ================================================================

def display_test(
    row
):

    print(
        "\n"
        + "=" * 78
    )

    print(
        f"{row['id']} — "
        f"{row['category'].upper()}"
    )

    print(
        f"Question: "
        f"{row['question']}"
    )

    print(
        "-" * 78
    )

    print(
        "Expected abstention:",
        row[
            "expected_abstained"
        ]
    )

    print(
        "Actual abstention:",
        row[
            "actual_abstained"
        ]
    )

    print(
        "Decision correct:",
        (
            "✅"
            if row[
                "abstention_correct"
            ]
            else "❌"
        )
    )

    print(
        "\nAnswer:",
        row[
            "answer"
        ]
    )

    print(
        "\nSource:",
        row[
            "source"
        ]
    )

    print(
        "Chunk:",
        row[
            "chunk"
        ]
    )

    print(
        "\nFocused evidence:",
        row[
            "focused_evidence"
        ]
    )

    print(
        "\nChunk similarity:",
        row[
            "chunk_similarity"
        ]
    )

    print(
        "Sentence similarity:",
        row[
            "sentence_similarity"
        ]
    )

    print(
        "Adjusted evidence score:",
        row[
            "adjusted_evidence_score"
        ]
    )


# ================================================================
# SAVE DETAILED RESULTS
# ================================================================

def save_results(
    rows
):

    fieldnames = [

        "id",
        "category",
        "question",

        "expected_abstained",
        "actual_abstained",
        "abstention_correct",

        "answer",

        "source",
        "chunk",
        "focused_evidence",

        "top_evidence_score",

        "chunk_similarity",
        "sentence_similarity",
        "adjusted_evidence_score",

        "retrieval_mode",
        "conflict"
    ]


    with DETAIL_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ================================================================
# CALCULATE SUMMARY
# ================================================================

def build_summary(
    rows
):

    total = len(
        rows
    )

    correct = sum(
        1
        for row in rows
        if row[
            "abstention_correct"
        ]
    )

    answerable_rows = [
        row
        for row in rows
        if row[
            "category"
        ]
        == "answerable"
    ]

    unsupported_rows = [
        row
        for row in rows
        if row[
            "category"
        ]
        == "unsupported"
    ]


    # ------------------------------------------------------------
    # FALSE ABSTENTION
    #
    # The system refused an answerable question.
    # ------------------------------------------------------------

    false_abstentions = [
        row
        for row in answerable_rows
        if row[
            "actual_abstained"
        ]
    ]


    # ------------------------------------------------------------
    # FALSE ANSWER
    #
    # The system answered a question that should
    # have been unsupported.
    # ------------------------------------------------------------

    false_answers = [
        row
        for row in unsupported_rows
        if not row[
            "actual_abstained"
        ]
    ]


    answerable_scores = [
        row[
            "top_evidence_score"
        ]
        for row in answerable_rows
        if row[
            "top_evidence_score"
        ]
        is not None
    ]


    unsupported_scores = [
        row[
            "top_evidence_score"
        ]
        for row in unsupported_rows
        if row[
            "top_evidence_score"
        ]
        is not None
    ]


    lowest_answerable = (
        min(
            answerable_scores
        )
        if answerable_scores
        else None
    )


    highest_unsupported = (
        max(
            unsupported_scores
        )
        if unsupported_scores
        else None
    )


    accuracy = (
        (
            correct
            / total
        )
        * 100
        if total
        else 0
    )


    return {

        "requests":
            total,

        "abstention_decision_accuracy":
            accuracy,

        "false_abstentions":
            len(
                false_abstentions
            ),

        "false_answers":
            len(
                false_answers
            ),

        "lowest_answerable_score":
            lowest_answerable,

        "highest_unsupported_score":
            highest_unsupported
    }


# ================================================================
# SAVE SUMMARY
# ================================================================

def save_summary(
    summary
):

    with SUMMARY_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "metric",
                "value"
            ]
        )

        for key, value in (
            summary.items()
        ):

            writer.writerow(
                [
                    key,
                    value
                ]
            )


# ================================================================
# MAIN
# ================================================================

def main():

    print(
        "\nSTAGE 4 — ABSTENTION BENCHMARK"
    )

    print(
        "=" * 78
    )

    print(
        "This experiment measures current RAG behaviour."
    )

    print(
        "No retrieval threshold is being changed yet."
    )

    print(
        "\nTotal requests:",
        len(
            TEST_QUESTIONS
        )
    )


    rows = []


    for test in TEST_QUESTIONS:

        try:

            row = (
                run_test(
                    test
                )
            )

            rows.append(
                row
            )

            display_test(
                row
            )


        except Exception as error:

            print(
                "\n"
                + "=" * 78
            )

            print(
                f"{test['id']} FAILED"
            )

            print(
                "Question:",
                test[
                    "question"
                ]
            )

            print(
                "Error:",
                f"{type(error).__name__}: "
                f"{error}"
            )


    if not rows:

        print(
            "\nNo tests completed."
        )

        return


    save_results(
        rows
    )


    summary = (
        build_summary(
            rows
        )
    )


    save_summary(
        summary
    )


    print(
        "\n"
        + "=" * 78
    )

    print(
        "STAGE 4 — ABSTENTION BASELINE SUMMARY"
    )

    print(
        "=" * 78
    )

    print(
        "Requests:",
        summary[
            "requests"
        ]
    )

    print(
        "Abstention Decision Accuracy:",
        f"{summary['abstention_decision_accuracy']:.2f}%"
    )

    print(
        "False Abstentions:",
        summary[
            "false_abstentions"
        ]
    )

    print(
        "False Answers:",
        summary[
            "false_answers"
        ]
    )

    print(
        "Lowest Answerable Evidence Score:",
        summary[
            "lowest_answerable_score"
        ]
    )

    print(
        "Highest Unsupported Evidence Score:",
        summary[
            "highest_unsupported_score"
        ]
    )


    print(
        "\nDetailed results:"
    )

    print(
        DETAIL_FILE
    )


    print(
        "\nSummary:"
    )

    print(
        SUMMARY_FILE
    )


if __name__ == "__main__":

    main()