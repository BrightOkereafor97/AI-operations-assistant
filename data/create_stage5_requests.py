import csv

from pathlib import Path


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

OUTPUT_FILE = (
    DATA_DIR
    / "stage5_requests.csv"
)


# ================================================================
# STAGE 5 FIXED MULTI-TOOL TEST SET
#
# expected_tools uses | as the separator.
#
# Example:
# get_ticket|search_company_knowledge
#
# Some tasks are independent.
# Some tasks are dependent, meaning the result from one tool is
# needed before the next tool can be used correctly.
# ================================================================

TEST_CASES = [

    # ------------------------------------------------------------
    # TICKET + KNOWLEDGE
    # ------------------------------------------------------------

    {
        "request_id": "S5-001",

        "category":
            "ticket_knowledge_independent",

        "user_request":
            (
                "Check ticket TKT-005 and tell me "
                "what the company escalation policy says."
            ),

        "expected_tools":
            "get_ticket|search_company_knowledge",

        "dependency":
            "independent",

        "expected_first_argument":
            "TKT-005",

        "notes":
            (
                "Ticket lookup and escalation-policy lookup "
                "can be performed independently."
            )
    },


    {
        "request_id": "S5-002",

        "category":
            "ticket_knowledge_dependent",

        "user_request":
            (
                "Check TKT-003 and tell me what response "
                "target applies to its priority."
            ),

        "expected_tools":
            "get_ticket|search_company_knowledge",

        "dependency":
            "dependent",

        "expected_first_argument":
            "TKT-003",

        "notes":
            (
                "Ticket must be read first. Its priority is "
                "then used to search company knowledge."
            )
    },


    {
        "request_id": "S5-003",

        "category":
            "ticket_knowledge_independent",

        "user_request":
            (
                "Who owns TKT-017 and what does the "
                "company escalation process say?"
            ),

        "expected_tools":
            "get_ticket|search_company_knowledge",

        "dependency":
            "independent",

        "expected_first_argument":
            "TKT-017",

        "notes":
            (
                "Return ticket owner plus relevant "
                "company escalation information."
            )
    },


    # ------------------------------------------------------------
    # EXPENSE CLAIM + KNOWLEDGE
    # ------------------------------------------------------------

    {
        "request_id": "S5-004",

        "category":
            "claim_knowledge",

        "user_request":
            (
                "Check expense claim EXP-010 and tell me "
                "what the company policy says about "
                "employee travel expenses."
            ),

        "expected_tools":
            "get_expense_claim|search_company_knowledge",

        "dependency":
            "independent",

        "expected_first_argument":
            "EXP-010",

        "notes":
            (
                "Expense-claim lookup plus company "
                "travel-expense policy."
            )
    },


    {
        "request_id": "S5-005",

        "category":
            "claim_knowledge",

        "user_request":
            (
                "What is the status of EXP-004 and "
                "what does the Employee Expense Policy say?"
            ),

        "expected_tools":
            "get_expense_claim|search_company_knowledge",

        "dependency":
            "independent",

        "expected_first_argument":
            "EXP-004",

        "notes":
            (
                "Claim record and policy information "
                "must both appear in the response."
            )
    },


    # ------------------------------------------------------------
    # CALCULATION + KNOWLEDGE
    # ------------------------------------------------------------

    {
        "request_id": "S5-006",

        "category":
            "calculation_knowledge",

        "user_request":
            (
                "Add 32000 hotel and 8500 transport, "
                "then tell me what our travel-expense "
                "policy says."
            ),

        "expected_tools":
            "calculate_expense|search_company_knowledge",

        "dependency":
            "independent",

        "expected_first_argument":
            "32000|8500",

        "notes":
            (
                "Arithmetic must be performed by Python, "
                "while policy information comes from RAG."
            )
    },


    {
        "request_id": "S5-007",

        "category":
            "calculation_knowledge",

        "user_request":
            (
                "Calculate 18500 plus 7200 plus 45000 "
                "and tell me what the company says about "
                "employee expenses."
            ),

        "expected_tools":
            "calculate_expense|search_company_knowledge",

        "dependency":
            "independent",

        "expected_first_argument":
            "18500|7200|45000",

        "notes":
            (
                "Tests deterministic arithmetic plus "
                "company-knowledge retrieval."
            )
    },


    # ------------------------------------------------------------
    # TICKET + EXPENSE CLAIM
    # ------------------------------------------------------------

    {
        "request_id": "S5-008",

        "category":
            "ticket_claim",

        "user_request":
            (
                "Check TKT-010 and EXP-010. Tell me "
                "the ticket status and the expense-claim status."
            ),

        "expected_tools":
            "get_ticket|get_expense_claim",

        "dependency":
            "independent",

        "expected_first_argument":
            "TKT-010",

        "notes":
            (
                "Requires two different structured-record tools."
            )
    },


    {
        "request_id": "S5-009",

        "category":
            "ticket_claim",

        "user_request":
            (
                "Who owns TKT-020 and who submitted EXP-002?"
            ),

        "expected_tools":
            "get_ticket|get_expense_claim",

        "dependency":
            "independent",

        "expected_first_argument":
            "TKT-020",

        "notes":
            (
                "Two record lookups; answer must combine "
                "owner and employee information."
            )
    },


    # ------------------------------------------------------------
    # CLAIM → CALCULATION DEPENDENCY
    # ------------------------------------------------------------

    {
        "request_id": "S5-010",

        "category":
            "claim_calculation_dependent",

        "user_request":
            (
                "Get the amount on EXP-010 and add "
                "7200 to it."
            ),

        "expected_tools":
            "get_expense_claim|calculate_expense",

        "dependency":
            "dependent",

        "expected_first_argument":
            "EXP-010",

        "notes":
            (
                "The calculator cannot receive its complete "
                "arguments until EXP-010 has been retrieved."
            )
    },


    # ------------------------------------------------------------
    # THREE-TOOL REQUEST
    # ------------------------------------------------------------

    {
        "request_id": "S5-011",

        "category":
            "three_tool",

        "user_request":
            (
                "Check TKT-005, add 32000 and 8500, "
                "and tell me what our escalation policy says."
            ),

        "expected_tools":
            (
                "get_ticket|calculate_expense|"
                "search_company_knowledge"
            ),

        "dependency":
            "independent",

        "expected_first_argument":
            "TKT-005",

        "notes":
            (
                "Tests whether three required capabilities "
                "can be used for one user request."
            )
    },


    # ------------------------------------------------------------
    # MULTI-PART KNOWLEDGE + RECORD CONTROL
    # ------------------------------------------------------------

    {
        "request_id": "S5-012",

        "category":
            "claim_knowledge_multi_part",

        "user_request":
            (
                "Who submitted EXP-007, and how many annual "
                "leave days do full-time employees receive?"
            ),

        "expected_tools":
            "get_expense_claim|search_company_knowledge",

        "dependency":
            "independent",

        "expected_first_argument":
            "EXP-007",

        "notes":
            (
                "Two unrelated questions in one user message. "
                "Both must be completed."
            )
    },
]


# ================================================================
# CREATE CSV
# ================================================================

def main():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    fieldnames = [
        "request_id",
        "category",
        "user_request",
        "expected_tools",
        "dependency",
        "expected_first_argument",
        "notes"
    ]


    with OUTPUT_FILE.open(
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
            TEST_CASES
        )


    print(
        "\nStage 5 requests created successfully."
    )

    print(
        f"Requests: {len(TEST_CASES)}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":

    main()