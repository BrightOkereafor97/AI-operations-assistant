import csv
from pathlib import Path

from tools import (
    get_ticket,
    calculate_expense,
    search_company_knowledge
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)


EXPENSE_FILE = (
    PROJECT_ROOT
    / "data"
    / "expenses.csv"
)


def heading(title):

    print(
        "\n" + "=" * 70
    )

    print(title)

    print(
        "=" * 70
    )


# ================================================================
# TOOL 1 TESTS
# ================================================================

heading(
    "TOOL 1 — GET TICKET"
)


print(
    "\nVALID TEST:"
)


ticket = get_ticket(
    "TKT-005"
)


print(ticket)


print(
    "\nERROR TEST:"
)


try:

    get_ticket(
        "TKT-999"
    )

except Exception as error:

    print(
        type(error).__name__,
        ":",
        error
    )


# ================================================================
# TOOL 2 TESTS
# ================================================================

heading(
    "TOOL 2 — CALCULATE EXPENSE"
)


with EXPENSE_FILE.open(
    "r",
    encoding="utf-8"
) as file:

    claims = list(
        csv.DictReader(file)
    )


sample_claims = claims[:3]


print(
    "\nClaims:"
)


for claim in sample_claims:

    print(
        claim[
            "claim_id"
        ],
        claim[
            "amount"
        ]
    )


total = calculate_expense(
    sample_claims
)


print(
    "\nCalculated total:",
    total
)


print(
    "\nERROR TEST:"
)


try:

    calculate_expense(
        [
            5000,
            {
                "amount":
                    "not-a-number"
            }
        ]
    )

except Exception as error:

    print(
        type(error).__name__,
        ":",
        error
    )


# ================================================================
# TOOL 3 TESTS
# ================================================================

heading(
    "TOOL 3 — SEARCH COMPANY KNOWLEDGE"
)


print(
    "\nVALID TEST:"
)


knowledge_result = (
    search_company_knowledge(
        "How many annual leave "
        "days do employees receive?"
    )
)


print(
    "Answer:",
    knowledge_result[
        "answer"
    ]
)


print(
    "Sources:",
    knowledge_result[
        "sources"
    ]
)


print(
    "\nMISSING-INFORMATION TEST:"
)


missing_result = (
    search_company_knowledge(
        "What is the CEO's "
        "home address?"
    )
)


print(
    "Answer:",
    missing_result[
        "answer"
    ]
)


print(
    "\nINVALID INPUT TEST:"
)


try:

    search_company_knowledge(
        ""
    )

except Exception as error:

    print(
        type(error).__name__,
        ":",
        error
    )


# ================================================================
# COMPLETE
# ================================================================

heading(
    "ALL DIRECT TOOL TESTS COMPLETE"
)