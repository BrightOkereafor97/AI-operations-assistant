import re


# ================================================================
# PROJECT 3 — STAGE 9
# REUSABLE ADVERSARIAL SAFETY GUARDS
#
# These guards sit BEFORE risky execution or generation.
# They are intentionally simple and inspectable.
# ================================================================


UNSUPPORTED_WRITE_VERBS = {
    "approve",
    "approved",
    "reject",
    "rejected",
    "delete",
    "remove",
    "update",
    "modify",
    "change",
    "edit",
    "close",
    "reopen",
    "send",
    "email",
}


PROMPT_INJECTION_PATTERNS = [
    r"\bignore\s+(?:all\s+)?previous\s+instructions?\b",
    r"\bignore\s+(?:all\s+)?prior\s+instructions?\b",
    r"\boverride\s+(?:the\s+)?instructions?\b",
    r"\breveal\s+(?:the\s+)?system\s+prompt\b",
    r"\breveal\s+(?:the\s+)?developer\s+message\b",
    r"\bshow\s+(?:the\s+)?hidden\s+instructions?\b",
    r"\bdo\s+not\s+follow\s+(?:the\s+)?instructions?\b",
    r"\bcall\s+(?:the\s+)?tool\b",
    r"\bexecute\s+(?:the\s+)?tool\b",
]


def detect_unsupported_write_intent(
    user_request
):
    """
    Block mutation requests for capabilities that are not available.

    Stage 8 supports create_escalation().
    Other mutation verbs remain unsupported.
    """

    text = str(
        user_request
    ).lower()

    tokens = set(
        re.findall(
            r"[a-zA-Z]+",
            text
        )
    )

    matched = sorted(
        token
        for token in UNSUPPORTED_WRITE_VERBS
        if token in tokens
    )

    if matched:

        return {
            "blocked": True,
            "reason": "UNSUPPORTED_WRITE_INTENT",
            "matched_verbs": matched,
            "message": (
                "This request asks for a state-changing action "
                "that is not available in the current tool set."
            ),
        }

    return {
        "blocked": False,
        "reason": None,
        "matched_verbs": [],
        "message": None,
    }


def detect_prompt_injection(
    text
):
    """
    Detect obvious prompt-injection instructions inside retrieved content.

    This is a defensive screening layer, not a complete security solution.
    """

    content = str(
        text
    )

    matched_patterns = []

    for pattern in PROMPT_INJECTION_PATTERNS:

        if re.search(
            pattern,
            content,
            re.IGNORECASE
        ):

            matched_patterns.append(
                pattern
            )

    return {
        "detected": bool(
            matched_patterns
        ),
        "matched_patterns": matched_patterns,
    }


def screen_retrieved_evidence(
    evidence_items
):
    """
    Reject retrieved evidence if any item appears to contain
    prompt-injection instructions.
    """

    unsafe_items = []

    for index, item in enumerate(
        evidence_items
    ):

        result = detect_prompt_injection(
            item
        )

        if result[
            "detected"
        ]:

            unsafe_items.append(
                {
                    "index": index,
                    "text": str(item),
                    "matched_patterns": (
                        result[
                            "matched_patterns"
                        ]
                    ),
                }
            )

    if unsafe_items:

        return {
            "safe": False,
            "reason": "PROMPT_INJECTION_DETECTED",
            "unsafe_items": unsafe_items,
        }

    return {
        "safe": True,
        "reason": None,
        "unsafe_items": [],
    }
