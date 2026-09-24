from datetime import (
    date,
    datetime,
)

from decimal import (
    Decimal,
)

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    Numeric,
    String,
    Text,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
)

from backend.database import (
    Base,
)


# ================================================================
# 1. SUPPORT TICKET
# ================================================================

class Ticket(
    Base
):

    __tablename__ = "tickets"


    ticket_id: Mapped[str] = mapped_column(
        String(20),
        primary_key=True,
    )


    customer: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )


    category: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )


    priority: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )


    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )


    owner: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )


    summary: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )


# ================================================================
# 2. EXPENSE CLAIM
# ================================================================

class ExpenseClaim(
    Base
):

    __tablename__ = "expense_claims"


    claim_id: Mapped[str] = mapped_column(
        String(20),
        primary_key=True,
    )


    employee: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )


    category: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )


    amount: Mapped[Decimal] = mapped_column(
        Numeric(
            precision=12,
            scale=2,
        ),
        nullable=False,
    )


    date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )


    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )


# ================================================================
# 3. ESCALATION
#
# Unlike tickets and expense claims, escalation is a write action.
#
# The UNIQUE constraint on ticket_id means one ticket cannot have
# multiple active escalation records in this MVP.
#
# This gives us database-level duplicate protection in addition to
# application-level validation.
# ================================================================

class Escalation(
    Base
):

    __tablename__ = "escalations"


    escalation_id: Mapped[str] = mapped_column(
        String(20),
        primary_key=True,
    )


    ticket_id: Mapped[str] = mapped_column(
        ForeignKey(
            "tickets.ticket_id"
        ),
        nullable=False,
        unique=True,
        index=True,
    )


    customer: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )


    ticket_priority: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )


    ticket_owner: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )


    reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )


    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="Open",
    )


    created_at: Mapped[datetime] = mapped_column(
        DateTime(
            timezone=True
        ),
        nullable=False,
    )