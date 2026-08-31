import csv
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent

OUTPUT_FILE = DATA_DIR / "stage2_requests.csv"


REQUESTS = [
    {
        "request_id": "S2-001",
        "user_request": "What is happening with ticket TKT-005?",
        "expected_tool_requested": 1,
        "expected_tool_name": "get_ticket",
        "expected_ticket_id": "TKT-005",
    },
    {
        "request_id": "S2-002",
        "user_request": "Check support case #3 for me.",
        "expected_tool_requested": 1,
        "expected_tool_name": "get_ticket",
        "expected_ticket_id": "TKT-003",
    },
    {
        "request_id": "S2-003",
        "user_request": "Can you pull up TKT-019?",
        "expected_tool_requested": 1,
        "expected_tool_name": "get_ticket",
        "expected_ticket_id": "TKT-019",
    },
    {
        "request_id": "S2-004",
        "user_request": "Who owns ticket 10?",
        "expected_tool_requested": 1,
        "expected_tool_name": "get_ticket",
        "expected_ticket_id": "TKT-010",
    },
    {
        "request_id": "S2-005",
        "user_request": "What is the current status of support ticket TKT-014?",
        "expected_tool_requested": 1,
        "expected_tool_name": "get_ticket",
        "expected_ticket_id": "TKT-014",
    },
    {
        "request_id": "S2-006",
        "user_request": "Show me the details for case 7.",
        "expected_tool_requested": 1,
        "expected_tool_name": "get_ticket",
        "expected_ticket_id": "TKT-007",
    },
    {
        "request_id": "S2-007",
        "user_request": "Look up ticket number 12.",
        "expected_tool_requested": 1,
        "expected_tool_name": "get_ticket",
        "expected_ticket_id": "TKT-012",
    },
    {
        "request_id": "S2-008",
        "user_request": "I need the priority and owner for TKT-017.",
        "expected_tool_requested": 1,
        "expected_tool_name": "get_ticket",
        "expected_ticket_id": "TKT-017",
    },
    {
        "request_id": "S2-009",
        "user_request": "Please check what happened on support case 2.",
        "expected_tool_requested": 1,
        "expected_tool_name": "get_ticket",
        "expected_ticket_id": "TKT-002",
    },
    {
        "request_id": "S2-010",
        "user_request": "Find ticket 20 and tell me its status.",
        "expected_tool_requested": 1,
        "expected_tool_name": "get_ticket",
        "expected_ticket_id": "TKT-020",
    },

    # Requests where get_ticket should NOT be called

    {
        "request_id": "S2-011",
        "user_request": "What is our annual leave policy?",
        "expected_tool_requested": 0,
        "expected_tool_name": "",
        "expected_ticket_id": "",
    },
    {
        "request_id": "S2-012",
        "user_request": "Add 32000 hotel and 8500 transport.",
        "expected_tool_requested": 0,
        "expected_tool_name": "",
        "expected_ticket_id": "",
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
        f"Created {len(REQUESTS)} Stage 2 requests."
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()