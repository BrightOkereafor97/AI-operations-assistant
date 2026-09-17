import re

import stage5_multi_tool_v4_final as v4

# Re-export the Stage 5 V4 implementation.
#
# This keeps the frozen V4 baseline untouched while V4.1 changes
# only the two intent-detection behaviours being tested.
from stage5_multi_tool_v4_final import *


# ================================================================
# STAGE 5 — V4.1 INTENT FIX
#
# Controlled changes from frozen V4:
#
# 1. Recognize:
#       remote work
#       remote working
#       remote-working
#
#    as company-knowledge intent.
#
# 2. Recognize:
#       "Calculate the expense total."
#
#    as calculation intent even when the amounts are missing.
#    Stage 7 can then ask for the missing amounts instead of
#    incorrectly asking for an expense-claim ID.
#
# Everything else continues to use the frozen V4 implementation.
# ================================================================


V4_1_INTENT_FIX = True


# ================================================================
# CALCULATION INTENT
# ================================================================

def find_calculation_position(user_request):
    match = re.search(
        r"\badd\b|\bcalculate\b|\btotal\b|\bsum\b|\bplus\b",
        user_request,
        re.IGNORECASE,
    )

    if not match:
        return None

    cleaned = TICKET_ID_PATTERN.sub(
        "",
        user_request,
    )

    cleaned = CLAIM_ID_PATTERN.sub(
        "",
        cleaned,
    )

    numbers = NUMBER_PATTERN.findall(
        cleaned
    )

    has_claim = (
        find_claim_position(
            user_request
        )
        is not None
    )

    # ------------------------------------------------------------
    # Normal calculation request:
    #
    # "Add 32000 and 8500."
    # ------------------------------------------------------------

    if len(numbers) >= 2:
        return match.start()

    # ------------------------------------------------------------
    # Dependent calculation request:
    #
    # "Get the amount on EXP-010 and add 7200."
    #
    # One numeric value comes from the user request and the other
    # will come from the previous expense-claim tool result.
    # ------------------------------------------------------------

    if (
        has_claim
        and
        len(numbers) >= 1
    ):
        return match.start()

    # ------------------------------------------------------------
    # V4.1 FIX
    #
    # This request clearly asks for a calculation:
    #
    #     "Calculate the expense total."
    #
    # There are no amounts yet, so execution should not guess.
    #
    # We still include calculate_expense in the plan.
    # Stage 7 will then see an empty argument list and ask:
    #
    #     "Which amounts should I calculate?"
    #
    # Keep this deliberately narrow so a generic use of the word
    # "total" does not automatically create a calculator call.
    # ------------------------------------------------------------

    if re.search(
        r"^\s*calculate\b",
        user_request,
        re.IGNORECASE,
    ):
        return match.start()

    return None


# ================================================================
# KNOWLEDGE INTENT
# ================================================================

def find_knowledge_position(user_request):
    lower_request = (
        user_request.lower()
    )

    phrases = [
        "policy",
        "process",
        "procedure",
        "escalation",
        "response target",
        "service level",
        "annual leave",
        "sick leave",

        # --------------------------------------------------------
        # V4.1 FIX
        #
        # Handle common variations of remote-work wording.
        # --------------------------------------------------------

        "remote work",
        "remote working",
        "remote-working",

        "company says",
        "company say",
        "what does the company",
    ]

    positions = [
        lower_request.find(
            phrase
        )
        for phrase in phrases
        if lower_request.find(
            phrase
        )
        >= 0
    ]

    if not positions:
        return None

    return min(
        positions
    )


# ================================================================
# STRUCTURAL PLAN
#
# Rebuilt here so the plan uses the V4.1 intent detectors above
# instead of the frozen V4 versions.
# ================================================================

def build_structural_plan(user_request):
    detected = []

    candidates = [
        (
            find_ticket_position(
                user_request
            ),
            "get_ticket",
        ),
        (
            find_claim_position(
                user_request
            ),
            "get_expense_claim",
        ),
        (
            find_calculation_position(
                user_request
            ),
            "calculate_expense",
        ),
        (
            find_knowledge_position(
                user_request
            ),
            "search_company_knowledge",
        ),
    ]

    for (
        position,
        tool_name,
    ) in candidates:

        if position is not None:
            detected.append(
                (
                    position,
                    tool_name,
                )
            )

    detected.sort(
        key=lambda item: item[0]
    )

    plan = []

    for (
        _,
        tool_name,
    ) in detected:

        if tool_name not in plan:
            plan.append(
                tool_name
            )

    return plan


# ================================================================
# PLAN VALIDATION
#
# Same V4 validation rule.
#
# The only difference is that structural_plan now uses the
# V4.1 intent detectors above.
# ================================================================

def validate_tool_plan(
    user_request,
    model_tools,
):
    structural_plan = (
        build_structural_plan(
            user_request
        )
    )

    if structural_plan:

        reason = (
            "MODEL_PLAN_ACCEPTED"
            if model_tools
            ==
            structural_plan
            else
            "STRUCTURAL_OVERRIDE"
        )

        return (
            structural_plan,
            structural_plan,
            reason,
        )

    if model_tools:

        return (
            model_tools,
            structural_plan,
            "MODEL_FALLBACK",
        )

    return (
        [],
        structural_plan,
        "NO_VALID_PLAN",
    )


# ================================================================
# PATCH THE FROZEN V4 MODULE
#
# Some V4 functions such as run_test() execute inside the imported
# V4 module and therefore look up their helper functions from the
# V4 module's own namespace.
#
# Point those specific helpers to the V4.1 versions so both:
#
#     Stage 7 imports
#
# and
#
#     direct Stage 5 regression runs
#
# use the same V4.1 logic.
# ================================================================

v4.find_calculation_position = (
    find_calculation_position
)

v4.find_knowledge_position = (
    find_knowledge_position
)

v4.build_structural_plan = (
    build_structural_plan
)

v4.validate_tool_plan = (
    validate_tool_plan
)


# ================================================================
# DIRECT EXECUTION
#
# Running:
#
#     python stage5_multi_tool.py
#
# still runs the original Stage 5 V4 benchmark, but through the
# V4.1 intent detectors above.
# ================================================================

if __name__ == "__main__":
    v4.main()