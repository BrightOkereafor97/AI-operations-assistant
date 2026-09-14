import csv
import re

from pathlib import Path


# ================================================================
# STAGE 4 — EVIDENCE VALIDATOR EXPERIMENT
#
# PURPOSE:
#
# Compare different abstention strategies WITHOUT changing
# the live RAG pipeline.
#
# This experiment uses the saved Stage 4 baseline CSV.
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

INPUT_FILE = (
    RESULTS_DIR
    / "stage4_abstention_baseline.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "stage4_evidence_validator_comparison.csv"
)


# ================================================================
# GENERIC WORDS
#
# These words describe the type of request rather than
# the actual subject being asked about.
#
# Example:
#
# "What is the company policy on employee parking spaces?"
#
# Subject terms:
#     parking, spaces
#
# Generic terms:
#     company, policy, employee
# ================================================================


GENERIC_WORDS = {

    "a",
    "an",
    "and",
    "are",
    "about",
    "according",
    "be",
    "can",
    "company",
    "companies",
    "does",
    "do",
    "for",
    "from",
    "give",
    "gives",
    "handled",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "our",
    "please",
    "policy",
    "policies",
    "process",
    "provide",
    "provides",
    "reimburse",
    "reimburses",
    "rule",
    "rules",
    "say",
    "says",
    "should",
    "show",
    "tell",
    "the",
    "their",
    "these",
    "this",
    "to",
    "us",
    "what",
    "when",
    "who",
    "with",

    # Business-generic actor terms
    "employee",
    "employees",
    "staff",

    # Generic request/action terms
    "bring",
    "bringing",
    "request",
    "requesting"
}


# ================================================================
# BASIC TOKEN NORMALISATION
# ================================================================

def normalise_word(
    word
):

    word = (
        word
        .lower()
        .strip()
    )

    # ------------------------------------------------------------
    # Very small general-purpose stemming.
    #
    # This is not meant to replace a full NLP lemmatiser.
    #
    # It simply helps match:
    #
    # refunds  -> refund
    # expenses -> expense
    # working  -> work
    # requests -> request
    # ------------------------------------------------------------

    if (
        word.endswith("ies")
        and len(word) > 4
    ):

        word = (
            word[:-3]
            + "y"
        )

    elif (
        word.endswith("ing")
        and len(word) > 5
    ):

        word = word[:-3]

    elif (
        word.endswith("ed")
        and len(word) > 4
    ):

        word = word[:-2]

    elif (
        word.endswith("es")
        and len(word) > 4
    ):

        word = word[:-2]

    elif (
        word.endswith("s")
        and len(word) > 3
    ):

        word = word[:-1]

    return word


# ================================================================
# TOKENISE TEXT
# ================================================================

def tokenise(
    text
):

    text = (
        str(text)
        .lower()
        .replace("-", " ")
    )

    words = re.findall(
        r"[a-z]+",
        text
    )

    return [
        normalise_word(
            word
        )
        for word in words
    ]


# ================================================================
# GET IMPORTANT QUESTION TERMS
# ================================================================

def extract_topic_terms(
    question
):

    terms = []

    for word in tokenise(
        question
    ):

        if word in GENERIC_WORDS:

            continue

        if len(word) < 3:

            continue

        if word not in terms:

            terms.append(
                word
            )

    return terms


# ================================================================
# TOPIC COVERAGE
#
# Example:
#
# Question topics:
#     childcare, expense
#
# Evidence:
#     business expenses including travel...
#
# Only "expense" matches.
#
# Coverage = 1 / 2 = 0.50
# ================================================================

def calculate_topic_coverage(
    question,
    evidence
):

    topic_terms = (
        extract_topic_terms(
            question
        )
    )

    evidence_terms = set(
        tokenise(
            evidence
        )
    )

    if not topic_terms:

        return (
            0.0,
            [],
            []
        )

    matched_terms = [
        term
        for term in topic_terms
        if term in evidence_terms
    ]

    missing_terms = [
        term
        for term in topic_terms
        if term not in evidence_terms
    ]

    coverage = (
        len(
            matched_terms
        )
        /
        len(
            topic_terms
        )
    )

    return (
        coverage,
        matched_terms,
        missing_terms
    )


# ================================================================
# BOOLEAN READER
# ================================================================

def read_bool(
    value
):

    return (
        str(value)
        .strip()
        .lower()
        ==
        "true"
    )


# ================================================================
# SCORE READER
# ================================================================

def read_score(
    value
):

    value = (
        str(value)
        .strip()
    )

    if not value:

        return None

    try:

        return float(
            value
        )

    except ValueError:

        return None


# ================================================================
# STRATEGY 1 — CURRENT SYSTEM
# ================================================================

def current_strategy(
    row
):

    return read_bool(
        row[
            "actual_abstained"
        ]
    )


# ================================================================
# STRATEGY 2 — THRESHOLD 0.50
# ================================================================

def threshold_050_strategy(
    row
):

    score = read_score(
        row[
            "adjusted_evidence_score"
        ]
    )

    if score is None:

        return True

    return (
        score
        < 0.50
    )


# ================================================================
# STRATEGY 3 — THRESHOLD 0.60
# ================================================================

def threshold_060_strategy(
    row
):

    score = read_score(
        row[
            "adjusted_evidence_score"
        ]
    )

    if score is None:

        return True

    return (
        score
        < 0.60
    )


# ================================================================
# STRATEGY 4 — HYBRID VALIDATOR
#
# Rule:
#
# 1. No evidence
#       -> abstain
#
# 2. Evidence score >= 0.60
#       -> accept
#
# 3. Evidence score < 0.50
#       -> abstain
#
# 4. Evidence between 0.50 and 0.60
#       -> require strong topic grounding
#
# This handles the "grey zone".
# ================================================================

def hybrid_strategy(
    row
):

    score = read_score(
        row[
            "adjusted_evidence_score"
        ]
    )

    if score is None:

        return True

    question = row[
        "question"
    ]

    evidence = row[
        "focused_evidence"
    ]

    (
        coverage,
        matched_terms,
        missing_terms
    ) = calculate_topic_coverage(
        question,
        evidence
    )


    # ------------------------------------------------------------
    # Clearly strong semantic evidence
    # ------------------------------------------------------------

    if score >= 0.60:

        return False


    # ------------------------------------------------------------
    # Clearly weak semantic evidence
    # ------------------------------------------------------------

    if score < 0.50:

        return True


    # ------------------------------------------------------------
    # GREY ZONE: 0.50 <= score < 0.60
    #
    # Semantic similarity alone is not trusted.
    #
    # We require at least 80% of the important topic terms
    # to be grounded in the actual evidence.
    # ------------------------------------------------------------

    if coverage >= 0.80:

        return False

    return True


# ================================================================
# EVALUATE ONE STRATEGY
# ================================================================

def evaluate_strategy(
    rows,
    strategy_name,
    strategy_function
):

    total = len(
        rows
    )

    correct = 0

    false_abstentions = 0

    false_answers = 0

    predictions = []


    for row in rows:

        expected = read_bool(
            row[
                "expected_abstained"
            ]
        )

        predicted = (
            strategy_function(
                row
            )
        )

        if predicted == expected:

            correct += 1

        elif (
            predicted
            and not expected
        ):

            false_abstentions += 1

        elif (
            not predicted
            and expected
        ):

            false_answers += 1


        predictions.append(
            predicted
        )


    accuracy = (
        correct
        / total
        * 100
    )


    return {

        "strategy":
            strategy_name,

        "accuracy":
            accuracy,

        "false_abstentions":
            false_abstentions,

        "false_answers":
            false_answers,

        "predictions":
            predictions
    }


# ================================================================
# LOAD BASELINE RESULTS
# ================================================================

def load_results():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Baseline file not found: "
            f"{INPUT_FILE}"
        )


    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        return list(
            csv.DictReader(
                file
            )
        )


# ================================================================
# DISPLAY QUESTION ANALYSIS
# ================================================================

def display_question_analysis(
    rows
):

    print(
        "\n"
        + "=" * 78
    )

    print(
        "TOPIC-GROUNDING ANALYSIS"
    )

    print(
        "=" * 78
    )


    for row in rows:

        question = row[
            "question"
        ]

        evidence = row[
            "focused_evidence"
        ]

        score = read_score(
            row[
                "adjusted_evidence_score"
            ]
        )

        (
            coverage,
            matched,
            missing
        ) = calculate_topic_coverage(
            question,
            evidence
        )


        print(
            f"\n{row['id']} — "
            f"{row['category'].upper()}"
        )

        print(
            "Score:",
            score
        )

        print(
            "Topic terms:",
            extract_topic_terms(
                question
            )
        )

        print(
            "Matched:",
            matched
        )

        print(
            "Missing:",
            missing
        )

        print(
            "Coverage:",
            f"{coverage:.2f}"
        )


# ================================================================
# SAVE COMPARISON
# ================================================================

def save_comparison(
    results
):

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "strategy",
                "accuracy",
                "false_abstentions",
                "false_answers"
            ]
        )


        for result in results:

            writer.writerow(
                [
                    result[
                        "strategy"
                    ],

                    result[
                        "accuracy"
                    ],

                    result[
                        "false_abstentions"
                    ],

                    result[
                        "false_answers"
                    ]
                ]
            )


# ================================================================
# MAIN
# ================================================================

def main():

    print(
        "\nSTAGE 4 — EVIDENCE VALIDATOR EXPERIMENT"
    )

    print(
        "=" * 78
    )

    print(
        "Using saved baseline results."
    )

    print(
        "The live RAG pipeline is NOT being changed."
    )


    rows = load_results()


    display_question_analysis(
        rows
    )


    strategies = [

        (
            "Current behaviour",
            current_strategy
        ),

        (
            "Threshold 0.50",
            threshold_050_strategy
        ),

        (
            "Threshold 0.60",
            threshold_060_strategy
        ),

        (
            "Hybrid semantic + topic grounding",
            hybrid_strategy
        )
    ]


    results = []


    for (
        name,
        function
    ) in strategies:

        result = (
            evaluate_strategy(
                rows,
                name,
                function
            )
        )

        results.append(
            result
        )


    print(
        "\n"
        + "=" * 78
    )

    print(
        "STRATEGY COMPARISON"
    )

    print(
        "=" * 78
    )


    for result in results:

        print(
            f"\n{result['strategy']}"
        )

        print(
            "Accuracy:",
            f"{result['accuracy']:.2f}%"
        )

        print(
            "False Abstentions:",
            result[
                "false_abstentions"
            ]
        )

        print(
            "False Answers:",
            result[
                "false_answers"
            ]
        )


    save_comparison(
        results
    )


    print(
        "\nComparison saved to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":

    main()