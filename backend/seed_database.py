import csv
import json

from datetime import (
    date,
    datetime,
    timezone,
)

from decimal import (
    Decimal,
)

from pathlib import (
    Path,
)

from sqlalchemy import (
    func,
    select,
)

from backend.database import (
    Base,
    SessionLocal,
    engine,
)

from backend.models import (
    Escalation,
    ExpenseClaim,
    Ticket,
)


# ================================================================
# 1. PROJECT PATHS
# ================================================================

BACKEND_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

PROJECT_ROOT = (
    BACKEND_DIR
    .parent
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

TICKETS_FILE = (
    DATA_DIR
    / "tickets.csv"
)

EXPENSES_FILE = (
    DATA_DIR
    / "expenses.csv"
)

ESCALATIONS_FILE = (
    DATA_DIR
    / "escalations.json"
)


# ================================================================
# 2. CREATE TABLES
# ================================================================

def create_tables():

    Base.metadata.create_all(
        bind=engine
    )


# ================================================================
# 3. SEED TICKETS
# ================================================================

def seed_tickets(
    db
):

    existing_count = db.scalar(
        select(
            func.count(
                Ticket.ticket_id
            )
        )
    )

    if existing_count:

        print(
            f"Tickets already seeded: "
            f"{existing_count}"
        )

        return


    with TICKETS_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )


        for row in reader:

            ticket = Ticket(
                ticket_id=(
                    row[
                        "ticket_id"
                    ]
                    .strip()
                    .upper()
                ),

                customer=(
                    row[
                        "customer"
                    ]
                    .strip()
                ),

                category=(
                    row[
                        "category"
                    ]
                    .strip()
                ),

                priority=(
                    row[
                        "priority"
                    ]
                    .strip()
                ),

                status=(
                    row[
                        "status"
                    ]
                    .strip()
                ),

                owner=(
                    row[
                        "owner"
                    ]
                    .strip()
                ),

                summary=(
                    row[
                        "summary"
                    ]
                    .strip()
                ),
            )


            db.add(
                ticket
            )


    db.commit()

    print(
        "Tickets seeded successfully."
    )


# ================================================================
# 4. SEED EXPENSE CLAIMS
# ================================================================

def seed_expense_claims(
    db
):

    existing_count = db.scalar(
        select(
            func.count(
                ExpenseClaim.claim_id
            )
        )
    )

    if existing_count:

        print(
            f"Expense claims already seeded: "
            f"{existing_count}"
        )

        return


    with EXPENSES_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )


        for row in reader:

            claim = ExpenseClaim(
                claim_id=(
                    row[
                        "claim_id"
                    ]
                    .strip()
                    .upper()
                ),

                employee=(
                    row[
                        "employee"
                    ]
                    .strip()
                ),

                category=(
                    row[
                        "category"
                    ]
                    .strip()
                ),

                amount=Decimal(
                    row[
                        "amount"
                    ]
                    .strip()
                ),

                date=date.fromisoformat(
                    row[
                        "date"
                    ]
                    .strip()
                ),

                status=(
                    row[
                        "status"
                    ]
                    .strip()
                ),
            )


            db.add(
                claim
            )


    db.commit()

    print(
        "Expense claims seeded successfully."
    )


# ================================================================
# 5. SEED EXISTING ESCALATIONS
#
# Optional.
#
# If escalations.json is empty or missing, the database simply
# starts with no escalation records.
# ================================================================

def seed_escalations(
    db
):

    existing_count = db.scalar(
        select(
            func.count(
                Escalation.escalation_id
            )
        )
    )

    if existing_count:

        print(
            f"Escalations already seeded: "
            f"{existing_count}"
        )

        return


    if not ESCALATIONS_FILE.exists():

        print(
            "No escalations.json file found. "
            "Skipping escalation import."
        )

        return


    raw_text = (
        ESCALATIONS_FILE
        .read_text(
            encoding="utf-8"
        )
        .strip()
    )


    if not raw_text:

        print(
            "Escalations file is empty."
        )

        return


    records = json.loads(
        raw_text
    )


    for record in records:

        created_at_text = (
            record.get(
                "created_at"
            )
        )


        if created_at_text:

            created_at = (
                datetime.fromisoformat(
                    created_at_text
                )
            )

        else:

            created_at = (
                datetime.now(
                    timezone.utc
                )
            )


        escalation = Escalation(
            escalation_id=(
                record[
                    "escalation_id"
                ]
            ),

            ticket_id=(
                record[
                    "ticket_id"
                ]
            ),

            customer=(
                record[
                    "customer"
                ]
            ),

            ticket_priority=(
                record[
                    "ticket_priority"
                ]
            ),

            ticket_owner=(
                record[
                    "ticket_owner"
                ]
            ),

            reason=(
                record[
                    "reason"
                ]
            ),

            status=(
                record.get(
                    "status",
                    "Open",
                )
            ),

            created_at=(
                created_at
            ),
        )


        db.add(
            escalation
        )


    db.commit()

    print(
        "Existing escalations seeded successfully."
    )


# ================================================================
# 6. DATABASE SUMMARY
# ================================================================

def display_summary(
    db
):

    ticket_count = db.scalar(
        select(
            func.count(
                Ticket.ticket_id
            )
        )
    )


    expense_count = db.scalar(
        select(
            func.count(
                ExpenseClaim.claim_id
            )
        )
    )


    escalation_count = db.scalar(
        select(
            func.count(
                Escalation.escalation_id
            )
        )
    )


    print(
        "\n"
        +
        "=" * 60
    )

    print(
        "DATABASE SEED SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        "Tickets:",
        ticket_count,
    )

    print(
        "Expense claims:",
        expense_count,
    )

    print(
        "Escalations:",
        escalation_count,
    )


# ================================================================
# 7. MAIN
# ================================================================

def main():

    print(
        "\nCreating database tables..."
    )


    create_tables()


    db = SessionLocal()


    try:

        seed_tickets(
            db
        )

        seed_expense_claims(
            db
        )

        seed_escalations(
            db
        )

        display_summary(
            db
        )


    except Exception:

        db.rollback()

        raise


    finally:

        db.close()


if __name__ == "__main__":

    main()