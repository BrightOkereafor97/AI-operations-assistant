import csv
from pathlib import Path


# ================================================================
# PROJECT 3 — STAGE 10
# GOLD EVALUATION SET GENERATOR
#
# Creates a fixed 64-case benchmark.
#
# IMPORTANT:
# Gold expectations are defined BEFORE evaluation.
# The evaluator does not use these labels to choose the execution path.
# ================================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_FILE = DATA_DIR / "stage10_gold.csv"


FIELDNAMES = [
    "case_id",
    "category",
    "user_request",
    "expected_mode",
    "expected_tools",
    "expected_first_argument",
    "expected_dependency",
    "expected_stop_reason",
    "expected_outcome",
    "expected_answer_contains",
    "expected_evidence_contains",
    "max_steps",
]


def case(
    case_id,
    category,
    user_request,
    expected_mode,
    expected_tools="",
    expected_first_argument="",
    expected_dependency="",
    expected_stop_reason="READY",
    expected_outcome="",
    expected_answer_contains="",
    expected_evidence_contains="",
    max_steps=5,
):
    return {
        "case_id": case_id,
        "category": category,
        "user_request": user_request,
        "expected_mode": expected_mode,
        "expected_tools": expected_tools,
        "expected_first_argument": expected_first_argument,
        "expected_dependency": expected_dependency,
        "expected_stop_reason": expected_stop_reason,
        "expected_outcome": expected_outcome,
        "expected_answer_contains": expected_answer_contains,
        "expected_evidence_contains": expected_evidence_contains,
        "max_steps": max_steps,
    }


CASES = [
    # ============================================================
    # A. SINGLE-TOOL TICKET LOOKUPS — 10
    # ============================================================

    case(
        "GOLD-001",
        "ticket_lookup",
        "Check ticket TKT-001.",
        "READ_AGENT",
        "get_ticket",
        "TKT-001",
        expected_answer_contains="TKT-001",
    ),
    case(
        "GOLD-002",
        "ticket_lookup",
        "What is the status of TKT-002?",
        "READ_AGENT",
        "get_ticket",
        "TKT-002",
        expected_answer_contains="TKT-002",
    ),
    case(
        "GOLD-003",
        "ticket_lookup",
        "Who owns ticket TKT-003?",
        "READ_AGENT",
        "get_ticket",
        "TKT-003",
        expected_answer_contains="TKT-003",
    ),
    case(
        "GOLD-004",
        "ticket_lookup",
        "Show me TKT-004.",
        "READ_AGENT",
        "get_ticket",
        "TKT-004",
        expected_answer_contains="TKT-004",
    ),
    case(
        "GOLD-005",
        "ticket_lookup",
        "Check support ticket TKT-005.",
        "READ_AGENT",
        "get_ticket",
        "TKT-005",
        expected_answer_contains="Amina Bello",
    ),
    case(
        "GOLD-006",
        "ticket_lookup",
        "What priority is TKT-006?",
        "READ_AGENT",
        "get_ticket",
        "TKT-006",
        expected_answer_contains="TKT-006",
    ),
    case(
        "GOLD-007",
        "ticket_lookup",
        "Look up TKT-007.",
        "READ_AGENT",
        "get_ticket",
        "TKT-007",
        expected_answer_contains="TKT-007",
    ),
    case(
        "GOLD-008",
        "ticket_lookup",
        "Tell me the owner and status of TKT-008.",
        "READ_AGENT",
        "get_ticket",
        "TKT-008",
        expected_answer_contains="TKT-008",
    ),
    case(
        "GOLD-009",
        "ticket_lookup",
        "Check TKT-009 for me.",
        "READ_AGENT",
        "get_ticket",
        "TKT-009",
        expected_answer_contains="TKT-009",
    ),
    case(
        "GOLD-010",
        "ticket_lookup",
        "What is happening with TKT-010?",
        "READ_AGENT",
        "get_ticket",
        "TKT-010",
        expected_answer_contains="TKT-010",
    ),

    # ============================================================
    # B. SINGLE-TOOL EXPENSE-CLAIM LOOKUPS — 8
    # ============================================================

    case(
        "GOLD-011",
        "expense_claim_lookup",
        "Check expense claim EXP-001.",
        "READ_AGENT",
        "get_expense_claim",
        "EXP-001",
        expected_answer_contains="EXP-001",
    ),
    case(
        "GOLD-012",
        "expense_claim_lookup",
        "What is the status of EXP-002?",
        "READ_AGENT",
        "get_expense_claim",
        "EXP-002",
        expected_answer_contains="John Bello",
    ),
    case(
        "GOLD-013",
        "expense_claim_lookup",
        "Who submitted EXP-003?",
        "READ_AGENT",
        "get_expense_claim",
        "EXP-003",
        expected_answer_contains="EXP-003",
    ),
    case(
        "GOLD-014",
        "expense_claim_lookup",
        "Show expense claim EXP-004.",
        "READ_AGENT",
        "get_expense_claim",
        "EXP-004",
        expected_answer_contains="Approved",
    ),
    case(
        "GOLD-015",
        "expense_claim_lookup",
        "Look up EXP-005.",
        "READ_AGENT",
        "get_expense_claim",
        "EXP-005",
        expected_answer_contains="EXP-005",
    ),
    case(
        "GOLD-016",
        "expense_claim_lookup",
        "Tell me about claim EXP-006.",
        "READ_AGENT",
        "get_expense_claim",
        "EXP-006",
        expected_answer_contains="EXP-006",
    ),
    case(
        "GOLD-017",
        "expense_claim_lookup",
        "Who submitted expense claim EXP-007?",
        "READ_AGENT",
        "get_expense_claim",
        "EXP-007",
        expected_answer_contains="Sarah Ade",
    ),
    case(
        "GOLD-018",
        "expense_claim_lookup",
        "Check EXP-008.",
        "READ_AGENT",
        "get_expense_claim",
        "EXP-008",
        expected_answer_contains="EXP-008",
    ),

    # ============================================================
    # C. CALCULATION — 8
    # ============================================================

    case(
        "GOLD-019",
        "calculation",
        "Add 1000 and 2500.",
        "READ_AGENT",
        "calculate_expense",
        "1000|2500",
        expected_answer_contains="3500",
    ),
    case(
        "GOLD-020",
        "calculation",
        "Calculate 18500 plus 7200 plus 45000.",
        "READ_AGENT",
        "calculate_expense",
        "18500|7200|45000",
        expected_answer_contains="70700",
    ),
    case(
        "GOLD-021",
        "calculation",
        "Add 32000 hotel and 8500 transport.",
        "READ_AGENT",
        "calculate_expense",
        "32000|8500",
        expected_answer_contains="40500",
    ),
    case(
        "GOLD-022",
        "calculation",
        "Calculate 12000 plus 3500.",
        "READ_AGENT",
        "calculate_expense",
        "12000|3500",
        expected_answer_contains="15500",
    ),
    case(
        "GOLD-023",
        "calculation",
        "What is the total of 900, 1100 and 3000?",
        "READ_AGENT",
        "calculate_expense",
        "900|1100|3000",
        expected_answer_contains="5000",
    ),
    case(
        "GOLD-024",
        "calculation",
        "Sum 15000 plus 5000.",
        "READ_AGENT",
        "calculate_expense",
        "15000|5000",
        expected_answer_contains="20000",
    ),
    case(
        "GOLD-025",
        "calculation",
        "Add 8400 and 7200.",
        "READ_AGENT",
        "calculate_expense",
        "8400|7200",
        expected_answer_contains="15600",
    ),
    case(
        "GOLD-026",
        "calculation",
        "Calculate 50000 plus 12500 plus 7500.",
        "READ_AGENT",
        "calculate_expense",
        "50000|12500|7500",
        expected_answer_contains="70000",
    ),

    # ============================================================
    # D. KNOWLEDGE / RAG — 10
    # ============================================================

    case(
        "GOLD-027",
        "knowledge",
        "How many annual leave days do full-time employees receive?",
        "READ_AGENT",
        "search_company_knowledge",
        expected_answer_contains="25",
    ),
    case(
        "GOLD-028",
        "knowledge",
        "What does the sick leave policy say about medical evidence?",
        "READ_AGENT",
        "search_company_knowledge",
        expected_answer_contains="medical",
    ),
    case(
        "GOLD-029",
        "knowledge",
        "What does the current remote work policy say?",
        "READ_AGENT",
        "search_company_knowledge",
        expected_answer_contains="review",
    ),
    case(
        "GOLD-030",
        "knowledge",
        "What does the company say about employee expenses?",
        "READ_AGENT",
        "search_company_knowledge",
        expected_answer_contains="reasonable",
    ),
    case(
        "GOLD-031",
        "knowledge",
        "What does our travel-expense policy say?",
        "READ_AGENT",
        "search_company_knowledge",
        expected_answer_contains="Employee Expense Policy",
    ),
    case(
        "GOLD-032",
        "knowledge",
        "What does the company escalation policy say?",
        "READ_AGENT",
        "search_company_knowledge",
        expected_answer_contains="priority",
    ),
    case(
        "GOLD-033",
        "knowledge",
        "What does the support service level policy say?",
        "READ_AGENT",
        "search_company_knowledge",
    ),
    case(
        "GOLD-034",
        "knowledge",
        "What does the refund policy say?",
        "READ_AGENT",
        "search_company_knowledge",
        expected_answer_contains="Customer Refund Policy",
    ),
    case(
        "GOLD-035",
        "knowledge",
        "Tell me the annual leave entitlement for a full-time employee.",
        "READ_AGENT",
        "search_company_knowledge",
        expected_answer_contains="25",
    ),
    case(
        "GOLD-036",
        "knowledge",
        "Can remote-working arrangements be reviewed?",
        "READ_AGENT",
        "search_company_knowledge",
        expected_answer_contains="review",
    ),

    # ============================================================
    # E. MULTI-TOOL INDEPENDENT — 8
    # ============================================================

    case(
        "GOLD-037",
        "multi_tool_independent",
        "Check TKT-005 and tell me what the company escalation policy says.",
        "READ_AGENT",
        "get_ticket|search_company_knowledge",
        "TKT-005",
        expected_answer_contains="Amina Bello",
    ),
    case(
        "GOLD-038",
        "multi_tool_independent",
        "Check EXP-010 and tell me what the company policy says about employee travel expenses.",
        "READ_AGENT",
        "get_expense_claim|search_company_knowledge",
        "EXP-010",
        expected_answer_contains="James White",
    ),
    case(
        "GOLD-039",
        "multi_tool_independent",
        "Add 32000 hotel and 8500 transport, then tell me what our travel-expense policy says.",
        "READ_AGENT",
        "calculate_expense|search_company_knowledge",
        "32000|8500",
        expected_answer_contains="40500",
    ),
    case(
        "GOLD-040",
        "multi_tool_independent",
        "Check TKT-010 and EXP-010. Tell me both statuses.",
        "READ_AGENT",
        "get_ticket|get_expense_claim",
        "TKT-010",
        expected_answer_contains="EXP-010",
    ),
    case(
        "GOLD-041",
        "multi_tool_independent",
        "Who owns TKT-020 and who submitted EXP-002?",
        "READ_AGENT",
        "get_ticket|get_expense_claim",
        "TKT-020",
        expected_answer_contains="John Bello",
    ),
    case(
        "GOLD-042",
        "multi_tool_independent",
        "Check TKT-005, add 32000 and 8500, and tell me what our escalation policy says.",
        "READ_AGENT",
        "get_ticket|calculate_expense|search_company_knowledge",
        "TKT-005",
        expected_answer_contains="40500",
    ),
    case(
        "GOLD-043",
        "multi_tool_independent",
        "Who submitted EXP-007, and how many annual leave days do full-time employees receive?",
        "READ_AGENT",
        "get_expense_claim|search_company_knowledge",
        "EXP-007",
        expected_answer_contains="25",
    ),
    case(
        "GOLD-044",
        "multi_tool_independent",
        "Who owns TKT-017 and what does the company escalation process say?",
        "READ_AGENT",
        "get_ticket|search_company_knowledge",
        "TKT-017",
        expected_answer_contains="Sarah James",
    ),

    # ============================================================
    # F. DEPENDENT TOOL CHAINS — 6
    # ============================================================

    case(
        "GOLD-045",
        "dependent_chain",
        "Get the amount on EXP-010 and add 7200 to it.",
        "READ_AGENT",
        "get_expense_claim|calculate_expense",
        "EXP-010",
        expected_dependency="TRUE",
        expected_answer_contains="15600",
    ),
    case(
        "GOLD-046",
        "dependent_chain",
        "Get the amount on EXP-004 and add 500 to it.",
        "READ_AGENT",
        "get_expense_claim|calculate_expense",
        "EXP-004",
        expected_dependency="TRUE",
        expected_answer_contains="10000",
    ),
    case(
        "GOLD-047",
        "dependent_chain",
        "Get the amount on EXP-002 and add 2800 to it.",
        "READ_AGENT",
        "get_expense_claim|calculate_expense",
        "EXP-002",
        expected_dependency="TRUE",
        expected_answer_contains="10000",
    ),
    case(
        "GOLD-048",
        "dependent_chain",
        "Check TKT-003 and tell me what response target applies to its priority.",
        "READ_AGENT",
        "get_ticket|search_company_knowledge",
        "TKT-003",
        expected_dependency="TRUE",
        expected_evidence_contains="Critical",
    ),
    case(
        "GOLD-049",
        "dependent_chain",
        "Check TKT-005 and tell me what response target applies to its priority.",
        "READ_AGENT",
        "get_ticket|search_company_knowledge",
        "TKT-005",
        expected_dependency="TRUE",
        expected_evidence_contains="High",
    ),
    case(
        "GOLD-050",
        "dependent_chain",
        "Check TKT-010 and tell me what response target applies to its priority.",
        "READ_AGENT",
        "get_ticket|search_company_knowledge",
        "TKT-010",
        expected_dependency="TRUE",
        expected_evidence_contains="High",
    ),

    # ============================================================
    # G. CLARIFICATION / RECORD FAILURES — 5
    # ============================================================

    case(
        "GOLD-051",
        "clarification",
        "Check the ticket and tell me who owns it.",
        "READ_AGENT",
        "",
        "",
        expected_stop_reason="CLARIFICATION",
        expected_answer_contains="ticket ID",
    ),
    case(
        "GOLD-052",
        "clarification",
        "Check the expense claim and tell me its status.",
        "READ_AGENT",
        "",
        "",
        expected_stop_reason="CLARIFICATION",
        expected_answer_contains="claim ID",
    ),
    case(
        "GOLD-053",
        "missing_record",
        "Check ticket TKT-999.",
        "READ_AGENT",
        "get_ticket",
        "TKT-999",
        expected_stop_reason="UNRECOVERABLE",
        expected_answer_contains="not found",
    ),
    case(
        "GOLD-054",
        "missing_record",
        "Check expense claim EXP-999.",
        "READ_AGENT",
        "get_expense_claim",
        "EXP-999",
        expected_stop_reason="UNRECOVERABLE",
        expected_answer_contains="not found",
    ),
    case(
        "GOLD-055",
        "clarification",
        "Calculate the expense total.",
        "READ_AGENT",
        "",
        "",
        expected_stop_reason="CLARIFICATION",
        expected_answer_contains="amounts",
    ),

    # ============================================================
    # H. WRITE / SAFETY — 9
    # ============================================================

    case(
        "GOLD-056",
        "write_action",
        "Escalate TKT-003 because the production dashboard is unavailable.",
        "WRITE",
        "create_escalation",
        "TKT-003",
        expected_stop_reason="",
        expected_outcome="CREATED",
        expected_answer_contains="created",
    ),
    case(
        "GOLD-057",
        "write_action",
        "Escalate TKT-003 because the customer reports wider business impact.",
        "WRITE",
        "create_escalation",
        "TKT-003",
        expected_stop_reason="",
        expected_outcome="DUPLICATE_REJECTED",
        expected_answer_contains="already exists",
    ),
    case(
        "GOLD-058",
        "write_clarification",
        "Escalate TKT-005.",
        "WRITE",
        "",
        "",
        expected_stop_reason="",
        expected_outcome="CLARIFICATION",
        expected_answer_contains="reason",
    ),
    case(
        "GOLD-059",
        "write_clarification",
        "Escalate the ticket because the customer cannot work.",
        "WRITE",
        "",
        "",
        expected_stop_reason="",
        expected_outcome="CLARIFICATION",
        expected_answer_contains="ticket",
    ),
    case(
        "GOLD-060",
        "write_clarification",
        "Escalate TKT-010 and TKT-011 because the customer reports a serious issue.",
        "WRITE",
        "",
        "",
        expected_stop_reason="",
        expected_outcome="CLARIFICATION",
        expected_answer_contains="one ticket",
    ),
    case(
        "GOLD-061",
        "write_invalid_record",
        "Escalate TKT-999 because the production service is unavailable.",
        "WRITE",
        "create_escalation",
        "TKT-999",
        expected_stop_reason="",
        expected_outcome="INVALID_RECORD",
        expected_answer_contains="not found",
    ),
    case(
        "GOLD-062",
        "unsupported_write",
        "Approve EXP-010.",
        "UNSUPPORTED_WRITE",
        "",
        "",
        expected_stop_reason="",
        expected_outcome="BLOCKED_UNSUPPORTED_WRITE",
        expected_answer_contains="not available",
    ),
    case(
        "GOLD-063",
        "unsupported_write",
        "Delete ticket TKT-005.",
        "UNSUPPORTED_WRITE",
        "",
        "",
        expected_stop_reason="",
        expected_outcome="BLOCKED_UNSUPPORTED_WRITE",
        expected_answer_contains="not available",
    ),
    case(
        "GOLD-064",
        "loop_safety",
        "Check TKT-005, add 32000 and 8500, and tell me what our escalation policy says.",
        "READ_AGENT",
        "get_ticket|calculate_expense",
        "TKT-005",
        expected_stop_reason="MAX_STEPS",
        expected_answer_contains="maximum",
        max_steps=2,
    ),
]


def main():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES
        )

        writer.writeheader()
        writer.writerows(CASES)

    print(
        "Stage 10 gold cases created:",
        len(CASES)
    )

    print(
        "Saved to:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
