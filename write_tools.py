import json
import re

from datetime import (
    datetime,
    timezone,
)

from pathlib import (
    Path,
)

from sqlalchemy import (
    select,
)

from sqlalchemy.exc import (
    IntegrityError,
)

from backend.database import (
    SessionLocal,
)

from backend.models import (
    Escalation,
)

from tools import (
    get_ticket,
)


# ================================================================
# PROJECT 3 — CONTROLLED WRITE TOOL
#
# Database-backed escalation storage.
#
# Safety controls:
#
# - ticket ID is required
# - reason is required
# - ticket must exist
# - duplicate escalation is rejected
# - database UNIQUE constraint provides a second duplicate guard
# - every attempted write is audited
#
# NOTE:
# Escalation business records now live in SQLAlchemy/database.
# Audit records remain in JSONL temporarily so Stage 8 audit
# behaviour stays unchanged during this migration.
# ================================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

AUDIT_FILE = (
    DATA_DIR
    / "escalation_audit.jsonl"
)

TICKET_ID_PATTERN = re.compile(
    r"^TKT-\d+$",
    re.IGNORECASE,
)


# ================================================================
# TIME
# ================================================================

def utc_timestamp():

    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


# ================================================================
# AUDIT STORAGE
#
# We intentionally keep audit storage separate for this phase.
#
# Business escalation records:
#     SQLAlchemy database
#
# Audit trail:
#     escalation_audit.jsonl
# ================================================================

def ensure_audit_storage():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def append_audit_record(
    record,
):

    ensure_audit_storage()

    with AUDIT_FILE.open(
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            +
            "\n"
        )


# ================================================================
# AUDIT HELPER
# ================================================================

def audit_attempt(
    ticket_id,
    reason,
    success,
    escalation_id=None,
    error=None,
):

    record = {

        "timestamp":
            utc_timestamp(),

        "action":
            "create_escalation",

        "ticket_id":
            ticket_id,

        "reason":
            reason,

        "success":
            bool(
                success
            ),

        "escalation_id":
            escalation_id,

        "error":
            error,
    }


    append_audit_record(
        record
    )


    return record


# ================================================================
# DATABASE SERIALIZATION
#
# Convert an SQLAlchemy Escalation object into the same dictionary
# shape that the old JSON implementation returned.
# ================================================================

def escalation_to_dict(
    escalation,
):

    created_at = (
        escalation.created_at
    )


    if isinstance(
        created_at,
        datetime,
    ):

        created_at = (
            created_at.isoformat()
        )


    return {

        "escalation_id":
            escalation.escalation_id,

        "ticket_id":
            escalation.ticket_id,

        "customer":
            escalation.customer,

        "ticket_priority":
            escalation.ticket_priority,

        "ticket_owner":
            escalation.ticket_owner,

        "reason":
            escalation.reason,

        "status":
            escalation.status,

        "created_at":
            created_at,
    }


# ================================================================
# READ HELPERS
# ================================================================

def list_escalations():

    db = SessionLocal()


    try:

        statement = (
            select(
                Escalation
            )
            .order_by(
                Escalation.escalation_id
            )
        )


        records = (
            db.scalars(
                statement
            )
            .all()
        )


        return [

            escalation_to_dict(
                record
            )

            for record
            in records
        ]


    finally:

        db.close()


def escalation_exists(
    ticket_id,
):

    normalized_ticket = (
        str(
            ticket_id
        )
        .strip()
        .upper()
    )


    if not normalized_ticket:

        return False


    db = SessionLocal()


    try:

        statement = (
            select(
                Escalation
            )
            .where(
                Escalation.ticket_id
                ==
                normalized_ticket
            )
        )


        record = (
            db.scalars(
                statement
            )
            .first()
        )


        return (
            record
            is not None
        )


    finally:

        db.close()


# ================================================================
# NEXT ESCALATION ID
#
# We preserve the original ESC-001, ESC-002... format.
#
# We calculate the highest existing numeric suffix and add one.
# This is safer than simply using len(records) + 1.
# ================================================================

def next_escalation_id(
    db,
):

    existing_ids = (
        db.scalars(
            select(
                Escalation.escalation_id
            )
        )
        .all()
    )


    highest_number = 0


    for escalation_id in existing_ids:

        if not escalation_id:

            continue


        match = re.fullmatch(
            r"ESC-(\d+)",
            str(
                escalation_id
            )
            .strip()
            .upper(),
        )


        if match:

            number = int(
                match.group(
                    1
                )
            )


            highest_number = max(
                highest_number,
                number,
            )


    return (
        f"ESC-{highest_number + 1:03d}"
    )


# ================================================================
# CREATE ESCALATION
# ================================================================

def create_escalation(
    ticket_id,
    reason,
):

    # ------------------------------------------------------------
    # Normalize arguments
    # ------------------------------------------------------------

    normalized_ticket = (
        str(
            ticket_id
        )
        .strip()
        .upper()
        if ticket_id is not None
        else ""
    )


    normalized_reason = (
        str(
            reason
        )
        .strip()
        if reason is not None
        else ""
    )


    # ------------------------------------------------------------
    # Validate ticket identifier
    # ------------------------------------------------------------

    if not normalized_ticket:

        error_text = (
            "ticket_id is required."
        )


        audit_attempt(
            normalized_ticket,
            normalized_reason,
            False,
            error=error_text,
        )


        raise ValueError(
            error_text
        )


    if not TICKET_ID_PATTERN.fullmatch(
        normalized_ticket
    ):

        error_text = (
            "ticket_id must use the TKT-### format."
        )


        audit_attempt(
            normalized_ticket,
            normalized_reason,
            False,
            error=error_text,
        )


        raise ValueError(
            error_text
        )


    # ------------------------------------------------------------
    # Validate reason
    # ------------------------------------------------------------

    if not normalized_reason:

        error_text = (
            "Escalation reason is required."
        )


        audit_attempt(
            normalized_ticket,
            normalized_reason,
            False,
            error=error_text,
        )


        raise ValueError(
            error_text
        )


    # ------------------------------------------------------------
    # Validate that ticket exists
    #
    # get_ticket() still reads the existing ticket source for now.
    #
    # In Phase 2B-2 it will move to SQLAlchemy without changing
    # this interface.
    # ------------------------------------------------------------

    try:

        ticket = get_ticket(
            normalized_ticket
        )


    except Exception as error:

        error_text = (
            f"{type(error).__name__}: "
            f"{error}"
        )


        audit_attempt(
            normalized_ticket,
            normalized_reason,
            False,
            error=error_text,
        )


        raise


    # ------------------------------------------------------------
    # Open database session
    # ------------------------------------------------------------

    db = SessionLocal()


    try:

        # --------------------------------------------------------
        # Application-level duplicate check
        # --------------------------------------------------------

        duplicate_statement = (
            select(
                Escalation
            )
            .where(
                Escalation.ticket_id
                ==
                normalized_ticket
            )
        )


        duplicate = (
            db.scalars(
                duplicate_statement
            )
            .first()
        )


        if duplicate is not None:

            error_text = (
                "An escalation already exists for "
                f"{normalized_ticket}: "
                f"{duplicate.escalation_id}"
            )


            audit_attempt(
                normalized_ticket,
                normalized_reason,
                False,
                escalation_id=(
                    duplicate.escalation_id
                ),
                error=error_text,
            )


            raise ValueError(
                error_text
            )


        # --------------------------------------------------------
        # Create next escalation identifier
        # --------------------------------------------------------

        escalation_id = (
            next_escalation_id(
                db
            )
        )


        created_at = (
            datetime.now(
                timezone.utc
            )
        )


        # --------------------------------------------------------
        # Build database record
        # --------------------------------------------------------

        escalation = Escalation(

            escalation_id=
                escalation_id,

            ticket_id=
                normalized_ticket,

            customer=
                ticket.get(
                    "customer"
                ),

            ticket_priority=
                ticket.get(
                    "priority"
                ),

            ticket_owner=
                ticket.get(
                    "owner"
                ),

            reason=
                normalized_reason,

            status=
                "Open",

            created_at=
                created_at,
        )


        db.add(
            escalation
        )


        # --------------------------------------------------------
        # Commit transaction
        # --------------------------------------------------------

        db.commit()


        # Refresh ensures SQLAlchemy reloads the saved record.
        db.refresh(
            escalation
        )


        escalation_dict = (
            escalation_to_dict(
                escalation
            )
        )


    except IntegrityError as error:

        db.rollback()


        # --------------------------------------------------------
        # Database-level duplicate protection
        #
        # Even if two requests race past the application check,
        # the UNIQUE ticket_id constraint protects the database.
        # --------------------------------------------------------

        existing = (
            db.scalars(
                select(
                    Escalation
                )
                .where(
                    Escalation.ticket_id
                    ==
                    normalized_ticket
                )
            )
            .first()
        )


        existing_id = (
            existing.escalation_id
            if existing is not None
            else None
        )


        error_text = (
            "An escalation already exists for "
            f"{normalized_ticket}"
        )


        if existing_id:

            error_text += (
                f": {existing_id}"
            )


        audit_attempt(
            normalized_ticket,
            normalized_reason,
            False,
            escalation_id=existing_id,
            error=error_text,
        )


        raise ValueError(
            error_text
        ) from error


    except Exception:

        db.rollback()

        raise


    finally:

        db.close()


    # ------------------------------------------------------------
    # Audit successful action
    # ------------------------------------------------------------

    audit_attempt(
        normalized_ticket,
        normalized_reason,
        True,
        escalation_id=escalation_id,
        error=None,
    )


    # ------------------------------------------------------------
    # Preserve original tool response shape
    # ------------------------------------------------------------

    return {

        "success":
            True,

        "message":
            (
                f"Escalation {escalation_id} "
                f"created for {normalized_ticket}."
            ),

        "escalation":
            escalation_dict,
    }