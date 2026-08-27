import csv
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent


# ================================================================
# 1. FICTIONAL SUPPORT TICKETS
# ================================================================

TICKETS = [
    {
        "ticket_id": "TKT-001",
        "customer": "Northstar Foods",
        "category": "Access",
        "priority": "High",
        "status": "Open",
        "owner": "Amina Bello",
        "summary": "User cannot sign in after a password reset."
    },
    {
        "ticket_id": "TKT-002",
        "customer": "BluePeak Logistics",
        "category": "Billing",
        "priority": "Medium",
        "status": "In Progress",
        "owner": "David Cole",
        "summary": "Customer reports an incorrect invoice amount."
    },
    {
        "ticket_id": "TKT-003",
        "customer": "GreenField Retail",
        "category": "Technical",
        "priority": "Critical",
        "status": "Open",
        "owner": "Sarah James",
        "summary": "Production dashboard is unavailable."
    },
    {
        "ticket_id": "TKT-004",
        "customer": "Horizon Media",
        "category": "Account",
        "priority": "Low",
        "status": "Resolved",
        "owner": "Michael Ade",
        "summary": "Customer requested an account email update."
    },
    {
        "ticket_id": "TKT-005",
        "customer": "Silverline Health",
        "category": "Technical",
        "priority": "High",
        "status": "In Progress",
        "owner": "Amina Bello",
        "summary": "API integration returns authentication errors."
    },
    {
        "ticket_id": "TKT-006",
        "customer": "Nova Learning",
        "category": "Billing",
        "priority": "Medium",
        "status": "Open",
        "owner": "David Cole",
        "summary": "Payment appears twice on the customer's statement."
    },
    {
        "ticket_id": "TKT-007",
        "customer": "Summit Energy",
        "category": "Access",
        "priority": "High",
        "status": "Resolved",
        "owner": "Sarah James",
        "summary": "Administrator was locked out of the portal."
    },
    {
        "ticket_id": "TKT-008",
        "customer": "OakBridge Consulting",
        "category": "Feature",
        "priority": "Low",
        "status": "Open",
        "owner": "Michael Ade",
        "summary": "Customer requested bulk export capability."
    },
    {
        "ticket_id": "TKT-009",
        "customer": "MetroWorks",
        "category": "Technical",
        "priority": "Critical",
        "status": "In Progress",
        "owner": "Amina Bello",
        "summary": "Customer data synchronization has stopped."
    },
    {
        "ticket_id": "TKT-010",
        "customer": "BrightPath Finance",
        "category": "Security",
        "priority": "High",
        "status": "Open",
        "owner": "Sarah James",
        "summary": "Suspicious login activity was reported."
    },
    {
        "ticket_id": "TKT-011",
        "customer": "Cedar Homes",
        "category": "Account",
        "priority": "Medium",
        "status": "Resolved",
        "owner": "David Cole",
        "summary": "User profile information required correction."
    },
    {
        "ticket_id": "TKT-012",
        "customer": "Atlas Manufacturing",
        "category": "Technical",
        "priority": "High",
        "status": "Open",
        "owner": "Michael Ade",
        "summary": "Automated report generation is failing."
    },
    {
        "ticket_id": "TKT-013",
        "customer": "ClearWater Services",
        "category": "Billing",
        "priority": "Low",
        "status": "Resolved",
        "owner": "David Cole",
        "summary": "Customer requested a copy of an old invoice."
    },
    {
        "ticket_id": "TKT-014",
        "customer": "Vertex Mobility",
        "category": "Technical",
        "priority": "Medium",
        "status": "In Progress",
        "owner": "Amina Bello",
        "summary": "Mobile application notifications are delayed."
    },
    {
        "ticket_id": "TKT-015",
        "customer": "Pioneer Legal",
        "category": "Access",
        "priority": "Medium",
        "status": "Open",
        "owner": "Sarah James",
        "summary": "New employee cannot access shared workspace."
    },
    {
        "ticket_id": "TKT-016",
        "customer": "Urban Harvest",
        "category": "Feature",
        "priority": "Low",
        "status": "Open",
        "owner": "Michael Ade",
        "summary": "Customer requested additional dashboard filters."
    },
    {
        "ticket_id": "TKT-017",
        "customer": "Apex Insurance",
        "category": "Security",
        "priority": "Critical",
        "status": "In Progress",
        "owner": "Sarah James",
        "summary": "Possible unauthorized account access detected."
    },
    {
        "ticket_id": "TKT-018",
        "customer": "GoldenGate Travel",
        "category": "Billing",
        "priority": "Medium",
        "status": "Resolved",
        "owner": "David Cole",
        "summary": "Customer asked about a failed subscription renewal."
    },
    {
        "ticket_id": "TKT-019",
        "customer": "PrimeEdge Systems",
        "category": "Technical",
        "priority": "High",
        "status": "Open",
        "owner": "Amina Bello",
        "summary": "Webhook events are not reaching the customer server."
    },
    {
        "ticket_id": "TKT-020",
        "customer": "LakeView Hospitality",
        "category": "Account",
        "priority": "Low",
        "status": "Resolved",
        "owner": "Michael Ade",
        "summary": "Customer requested closure of an unused user account."
    },
]


# ================================================================
# 2. FICTIONAL EXPENSE CLAIMS
# ================================================================

EXPENSES = [
    ["EXP-001", "Ada Martin", "Travel", 18500, "2026-08-01", "Approved"],
    ["EXP-002", "John Bello", "Meals", 7200, "2026-08-02", "Approved"],
    ["EXP-003", "Mary Cole", "Hotel", 45000, "2026-08-03", "Pending"],
    ["EXP-004", "Peter James", "Transport", 9500, "2026-08-04", "Approved"],
    ["EXP-005", "Grace Obi", "Training", 30000, "2026-08-05", "Pending"],
    ["EXP-006", "David King", "Meals", 6500, "2026-08-06", "Approved"],
    ["EXP-007", "Sarah Ade", "Software", 22000, "2026-08-07", "Approved"],
    ["EXP-008", "Michael Hart", "Travel", 17500, "2026-08-08", "Rejected"],
    ["EXP-009", "Tola Green", "Hotel", 52000, "2026-08-09", "Pending"],
    ["EXP-010", "James White", "Transport", 8400, "2026-08-10", "Approved"],
    ["EXP-011", "Helen Stone", "Meals", 5800, "2026-08-11", "Approved"],
    ["EXP-012", "Daniel Ford", "Equipment", 68000, "2026-08-12", "Pending"],
    ["EXP-013", "Lucy Brown", "Travel", 21000, "2026-08-13", "Approved"],
    ["EXP-014", "Mark Reed", "Software", 15500, "2026-08-14", "Approved"],
    ["EXP-015", "Ruth Adams", "Transport", 7600, "2026-08-15", "Approved"],
]


# ================================================================
# 3. WRITE FILES
# ================================================================

def create_tickets():

    path = DATA_DIR / "tickets.csv"

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=TICKETS[0].keys()
        )

        writer.writeheader()
        writer.writerows(TICKETS)

    print(
        f"Created {len(TICKETS)} tickets"
    )


def create_expenses():

    path = DATA_DIR / "expenses.csv"

    headers = [
        "claim_id",
        "employee",
        "category",
        "amount",
        "date",
        "status"
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(headers)
        writer.writerows(EXPENSES)

    print(
        f"Created {len(EXPENSES)} expense claims"
    )


if __name__ == "__main__":

    create_tickets()
    create_expenses()

    print(
        "\nStage 1 fictional datasets created successfully."
    )