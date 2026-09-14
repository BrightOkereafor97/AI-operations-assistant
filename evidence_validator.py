import re


# ================================================================
# STAGE 4 — RAG EVIDENCE VALIDATOR
#
# This does NOT perform retrieval or generation.
#
# Its only job is to decide whether retrieved evidence
# is sufficiently grounded to safely answer the question.
# ================================================================


LOW_EVIDENCE_THRESHOLD = 0.50

HIGH_EVIDENCE_THRESHOLD = 0.60

GREY_ZONE_TOPIC_COVERAGE = 0.80


# ================================================================
# GENERIC LANGUAGE
#
# These words normally describe the request rather than
# the specific business topic being asked about.
# ================================================================

GENERIC_WORDS = {
    "a",
    "about",
    "according",
    "an",
    "and",
    "are",
    "be",
    "can",
    "company",
    "do",
    "does",
    "for",
    "from",
    "give",
    "gives",
    "handled",
    "how",
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

    # Generic business actors
    "employee",
    "employees",
    "staff",

    # Generic actions
    "bring",
    "bringing",
    "request",
    "requesting"
}


# ================================================================
# TOKENISE
# ================================================================

def raw_tokens(
    text
):

    return re.findall(
        r"[a-z]+",
        str(text).lower()
    )


# ================================================================
# SIMPLE NORMALISATION
# ================================================================

def normalise_word(
    word
):

    word = (
        word
        .lower()
        .strip()
    )

    # expenses -> expense
    # rules -> rule
    # refunds -> refund

    if (
        word.endswith("s")
        and
        not word.endswith("ss")
        and
        len(word) > 3
    ):

        word = word[:-1]

    return word


# ================================================================
# EXTRACT TOPIC TERMS
# ================================================================

def extract_topic_terms(
    question
):

    terms = []

    for raw_word in raw_tokens(
        question
    ):

        # Remove generic words BEFORE normalisation.
        # This avoids forms such as:
        #
        # does -> doe
        # process -> proces
        #

        if raw_word in GENERIC_WORDS:

            continue

        word = normalise_word(
            raw_word
        )

        if len(word) < 3:

            continue

        if word not in terms:

            terms.append(
                word
            )

    return terms


# ================================================================
# EVIDENCE TERMS
# ================================================================

def extract_evidence_terms(
    evidence
):

    return {
        normalise_word(
            word
        )
        for word in raw_tokens(
            evidence
        )
    }


# ================================================================
# TOPIC COVERAGE
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

    evidence_terms = (
        extract_evidence_terms(
            evidence
        )
    )

    if not topic_terms:

        return (
            0.0,
            [],
            []
        )

    matched = [
        term
        for term in topic_terms
        if term in evidence_terms
    ]

    missing = [
        term
        for term in topic_terms
        if term not in evidence_terms
    ]

    coverage = (
        len(matched)
        /
        len(topic_terms)
    )

    return (
        coverage,
        matched,
        missing
    )


# ================================================================
# VALIDATE EVIDENCE
# ================================================================

def validate_evidence(
    question,
    evidence,
    adjusted_evidence_score
):

    """
    Return a structured evidence-validation decision.

    Rules:

    1. Missing evidence -> reject.

    2. Score below 0.50 -> reject.

    3. Score >= 0.60 -> accept.

    4. Score between 0.50 and 0.60:
       require strong topic grounding.
    """

    if not evidence:

        return {
            "supported": False,
            "reason": "NO_EVIDENCE",
            "score": adjusted_evidence_score,
            "topic_coverage": 0.0,
            "matched_terms": [],
            "missing_terms": []
        }


    try:

        score = float(
            adjusted_evidence_score
        )

    except (
        TypeError,
        ValueError
    ):

        return {
            "supported": False,
            "reason": "INVALID_EVIDENCE_SCORE",
            "score": adjusted_evidence_score,
            "topic_coverage": 0.0,
            "matched_terms": [],
            "missing_terms": []
        }


    (
        coverage,
        matched,
        missing
    ) = calculate_topic_coverage(
        question,
        evidence
    )


    # ------------------------------------------------------------
    # CLEARLY WEAK
    # ------------------------------------------------------------

    if score < LOW_EVIDENCE_THRESHOLD:

        return {
            "supported": False,
            "reason": "LOW_EVIDENCE_SCORE",
            "score": score,
            "topic_coverage": coverage,
            "matched_terms": matched,
            "missing_terms": missing
        }


    # ------------------------------------------------------------
    # CLEARLY STRONG
    # ------------------------------------------------------------

    if score >= HIGH_EVIDENCE_THRESHOLD:

        return {
            "supported": True,
            "reason": "STRONG_SEMANTIC_EVIDENCE",
            "score": score,
            "topic_coverage": coverage,
            "matched_terms": matched,
            "missing_terms": missing
        }


    # ------------------------------------------------------------
    # GREY ZONE
    #
    # Semantic similarity alone is not enough.
    # Require strong lexical/topic grounding too.
    # ------------------------------------------------------------

    if coverage >= GREY_ZONE_TOPIC_COVERAGE:

        return {
            "supported": True,
            "reason": "GREY_ZONE_TOPIC_GROUNDED",
            "score": score,
            "topic_coverage": coverage,
            "matched_terms": matched,
            "missing_terms": missing
        }


    return {
        "supported": False,
        "reason": "GREY_ZONE_NOT_GROUNDED",
        "score": score,
        "topic_coverage": coverage,
        "matched_terms": matched,
        "missing_terms": missing
    }