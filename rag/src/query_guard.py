import re


# ------------------------------------------------
# LIGHTWEIGHT AMBIGUITY GUARD
#
# This is intentionally conservative.
#
# We only block questions whose target is clearly
# underspecified.
# ------------------------------------------------


AMBIGUOUS_PATTERNS = [

    r"what is (the )?policy",

    r"how long does (it|this|that) take",

    r"what does (the )?company allow",

    r"what is the current limit",

    r"how many days do (i|we) get"
]


def normalize_question(question):

    cleaned = (
        question
        .strip()
        .lower()
    )

    cleaned = re.sub(
        r"[?.!]+$",
        "",
        cleaned
    )

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned
    )

    return cleaned


def is_ambiguous_question(question):

    cleaned = normalize_question(
        question
    )


    for pattern in AMBIGUOUS_PATTERNS:

        if re.fullmatch(
            pattern,
            cleaned
        ):

            return True


    return False


def clarification_message():

    return (
        "Your question is ambiguous. "
        "Please specify the policy, process, "
        "or topic you are asking about."
    )


# ------------------------------------------------
# QUICK TEST
# ------------------------------------------------

if __name__ == "__main__":

    tests = [

        "What is the policy?",

        "How long does it take?",

        "What does the company allow?",

        "What is the current limit?",

        "How many days do I get?",

        "What is the CEO's home address?",

        "What colour is the CEO's car?",

        "How long do customers have to request a refund?"
    ]


    for question in tests:

        print(
            question,
            "->",
            is_ambiguous_question(
                question
            )
        )