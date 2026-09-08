import csv
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = DATA_DIR / "stage3_requests.csv"


REQUESTS = [

    # ============================================================
    # KNOWLEDGE / POLICY — 8
    # ============================================================

    {
        "request_id": "S3-001",
        "category": "knowledge",
        "user_request": "How many annual leave days do full-time employees receive?",
        "expected_tool": "search_company_knowledge",
        "expected_argument": "How many annual leave days do full-time employees receive?",
    },

    {
        "request_id": "S3-002",
        "category": "knowledge",
        "user_request": "What does the company policy say about remote work?",
        "expected_tool": "search_company_knowledge",
        "expected_argument": "What does the company policy say about remote work?",
    },

    {
        "request_id": "S3-003",
        "category": "knowledge",
        "user_request": "What is the company refund policy?",
        "expected_tool": "search_company_knowledge",
        "expected_argument": "What is the company refund policy?",
    },

    {
        "request_id": "S3-004",
        "category": "knowledge",
        "user_request": "What are the rules for employee travel expenses?",
        "expected_tool": "search_company_knowledge",
        "expected_argument": "What are the rules for employee travel expenses?",
    },

    {
        "request_id": "S3-005",
        "category": "knowledge",
        "user_request": "Tell me the company's escalation policy.",
        "expected_tool": "search_company_knowledge",
        "expected_argument": "Tell me the company's escalation policy.",
    },

    {
        "request_id": "S3-006",
        "category": "knowledge",
        "user_request": "What does our policy say about sick leave?",
        "expected_tool": "search_company_knowledge",
        "expected_argument": "What does our policy say about sick leave?",
    },

    {
        "request_id": "S3-007",
        "category": "knowledge",
        "user_request": "What is the process for requesting annual leave?",
        "expected_tool": "search_company_knowledge",
        "expected_argument": "What is the process for requesting annual leave?",
    },

    {
        "request_id": "S3-008",
        "category": "knowledge",
        "user_request": "According to company policy, how should customer refunds be handled?",
        "expected_tool": "search_company_knowledge",
        "expected_argument": "According to company policy, how should customer refunds be handled?",
    },


    # ============================================================
    # TICKET LOOKUP — 6
    # ============================================================

    {
        "request_id": "S3-009",
        "category": "ticket",
        "user_request": "What is happening with ticket TKT-005?",
        "expected_tool": "get_ticket",
        "expected_argument": "TKT-005",
    },

    {
        "request_id": "S3-010",
        "category": "ticket",
        "user_request": "Who owns ticket 10?",
        "expected_tool": "get_ticket",
        "expected_argument": "TKT-010",
    },

    {
        "request_id": "S3-011",
        "category": "ticket",
        "user_request": "Check support case #3.",
        "expected_tool": "get_ticket",
        "expected_argument": "TKT-003",
    },

    {
        "request_id": "S3-012",
        "category": "ticket",
        "user_request": "Show me ticket number 12.",
        "expected_tool": "get_ticket",
        "expected_argument": "TKT-012",
    },

    {
        "request_id": "S3-013",
        "category": "ticket",
        "user_request": "What is the status of TKT-017?",
        "expected_tool": "get_ticket",
        "expected_argument": "TKT-017",
    },

    {
        "request_id": "S3-014",
        "category": "ticket",
        "user_request": "Find ticket 20 and tell me who owns it.",
        "expected_tool": "get_ticket",
        "expected_argument": "TKT-020",
    },


    # ============================================================
    # EXPENSE CALCULATION — 5
    # expected_argument uses comma-separated amounts
    # ============================================================

    {
        "request_id": "S3-015",
        "category": "calculation",
        "user_request": "Add 32000 hotel and 8500 transport.",
        "expected_tool": "calculate_expense",
        "expected_argument": "32000,8500",
    },

    {
        "request_id": "S3-016",
        "category": "calculation",
        "user_request": "Calculate the total of 18500, 7200 and 45000.",
        "expected_tool": "calculate_expense",
        "expected_argument": "18500,7200,45000",
    },

    {
        "request_id": "S3-017",
        "category": "calculation",
        "user_request": "What is 12000 hotel plus 3500 taxi?",
        "expected_tool": "calculate_expense",
        "expected_argument": "12000,3500",
    },

    {
        "request_id": "S3-018",
        "category": "calculation",
        "user_request": "Please total these expenses: 5000, 9000 and 2500.",
        "expected_tool": "calculate_expense",
        "expected_argument": "5000,9000,2500",
    },

    {
        "request_id": "S3-019",
        "category": "calculation",
        "user_request": "Add the expense amounts 15000 and 4200.",
        "expected_tool": "calculate_expense",
        "expected_argument": "15000,4200",
    },


    # ============================================================
    # EXPENSE CLAIM LOOKUP — 5
    # ============================================================

    {
        "request_id": "S3-020",
        "category": "expense_claim",
        "user_request": "What happened to expense claim EXP-010?",
        "expected_tool": "get_expense_claim",
        "expected_argument": "EXP-010",
    },

    {
        "request_id": "S3-021",
        "category": "expense_claim",
        "user_request": "Check claim number 4 for me.",
        "expected_tool": "get_expense_claim",
        "expected_argument": "EXP-004",
    },

    {
        "request_id": "S3-022",
        "category": "expense_claim",
        "user_request": "What is the status of EXP-014?",
        "expected_tool": "get_expense_claim",
        "expected_argument": "EXP-014",
    },

    {
        "request_id": "S3-023",
        "category": "expense_claim",
        "user_request": "Who submitted expense claim 7?",
        "expected_tool": "get_expense_claim",
        "expected_argument": "EXP-007",
    },

    {
        "request_id": "S3-024",
        "category": "expense_claim",
        "user_request": "Show me the details for claim EXP-002.",
        "expected_tool": "get_expense_claim",
        "expected_argument": "EXP-002",
    },


    # ============================================================
    # NO TOOL / UNSUPPORTED — 6
    # ============================================================

    {
        "request_id": "S3-025",
        "category": "no_tool",
        "user_request": "Hello, how are you?",
        "expected_tool": "NO_TOOL",
        "expected_argument": "",
    },

    {
        "request_id": "S3-026",
        "category": "no_tool",
        "user_request": "Write a poem about customer service.",
        "expected_tool": "NO_TOOL",
        "expected_argument": "",
    },

    {
        "request_id": "S3-027",
        "category": "no_tool",
        "user_request": "What is the capital of France?",
        "expected_tool": "NO_TOOL",
        "expected_argument": "",
    },

    {
        "request_id": "S3-028",
        "category": "no_tool",
        "user_request": "Send an email to the finance manager.",
        "expected_tool": "NO_TOOL",
        "expected_argument": "",
    },

    {
        "request_id": "S3-029",
        "category": "no_tool",
        "user_request": "Delete ticket TKT-005.",
        "expected_tool": "NO_TOOL",
        "expected_argument": "",
    },

    {
        "request_id": "S3-030",
        "category": "no_tool",
        "user_request": "Approve my expense claim.",
        "expected_tool": "NO_TOOL",
        "expected_argument": "",
    },
]


def main():

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=REQUESTS[0].keys()
        )

        writer.writeheader()
        writer.writerows(REQUESTS)

    print(
        f"Created {len(REQUESTS)} Stage 3 requests."
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()