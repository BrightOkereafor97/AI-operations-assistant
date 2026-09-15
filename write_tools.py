import json
import re

from datetime import datetime, timezone
from pathlib import Path

from tools import get_ticket


# ================================================================
# PROJECT 3 — STAGE 8
# CONTROLLED WRITE TOOL
#
# This module introduces the project's first write capability:
#
#     create_escalation(ticket_id, reason)
#
# All data is fictional and local.
#
# Safety controls:
# - ticket ID is required
# - reason is required
# - ticket must exist
# - duplicate escalation is rejected
# - every attempted write is audited
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

ESCALATION_FILE = (
    DATA_DIR
    / "escalations.json"
)

AUDIT_FILE = (
    DATA_DIR
    / "escalation_audit.jsonl"
)

TICKET_ID_PATTERN = re.compile(
    r"^TKT-\d+$",
    re.IGNORECASE
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
# STORAGE HELPERS
# ================================================================

def ensure_storage():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not ESCALATION_FILE.exists():

        ESCALATION_FILE.write_text(
            "[]",
            encoding="utf-8"
        )


def load_escalations():

    ensure_storage()

    raw_text = (
        ESCALATION_FILE
        .read_text(
            encoding="utf-8"
        )
        .strip()
    )

    if not raw_text:

        return []

    data = json.loads(
        raw_text
    )

    if not isinstance(
        data,
        list
    ):

        raise ValueError(
            "Escalation storage must contain a JSON list."
        )

    return data


def save_escalations(
    escalations
):

    ensure_storage()

    ESCALATION_FILE.write_text(
        json.dumps(
            escalations,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


def append_audit_record(
    record
):

    ensure_storage()

    with AUDIT_FILE.open(
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
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
    error=None
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
# READ HELPERS
# ================================================================

def list_escalations():

    return load_escalations()


def escalation_exists(
    ticket_id
):

    normalized_ticket = (
        str(
            ticket_id
        )
        .strip()
        .upper()
    )

    return any(

        str(
            record.get(
                "ticket_id",
                ""
            )
        )
        .strip()
        .upper()
        ==
        normalized_ticket

        for record
        in load_escalations()
    )


# ================================================================
# CREATE ESCALATION
# ================================================================

def create_escalation(
    ticket_id,
    reason
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
            error=error_text
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
            error=error_text
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
            error=error_text
        )

        raise ValueError(
            error_text
        )


    # ------------------------------------------------------------
    # Validate that ticket exists
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
            error=error_text
        )

        raise


    # ------------------------------------------------------------
    # Prevent duplicate action
    # ------------------------------------------------------------

    escalations = (
        load_escalations()
    )


    duplicate = next(
        (
            record

            for record
            in escalations

            if (
                str(
                    record.get(
                        "ticket_id",
                        ""
                    )
                )
                .strip()
                .upper()
                ==
                normalized_ticket
            )
        ),
        None
    )


    if duplicate is not None:

        error_text = (
            "An escalation already exists for "
            f"{normalized_ticket}: "
            f"{duplicate.get('escalation_id')}"
        )

        audit_attempt(
            normalized_ticket,
            normalized_reason,
            False,
            escalation_id=(
                duplicate.get(
                    "escalation_id"
                )
            ),
            error=error_text
        )

        raise ValueError(
            error_text
        )


    # ------------------------------------------------------------
    # Create fictional local escalation record
    # ------------------------------------------------------------

    escalation_id = (
        f"ESC-{len(escalations) + 1:03d}"
    )


    created_at = (
        utc_timestamp()
    )


    escalation = {

        "escalation_id":
            escalation_id,

        "ticket_id":
            normalized_ticket,

        "customer":
            ticket.get(
                "customer"
            ),

        "ticket_priority":
            ticket.get(
                "priority"
            ),

        "ticket_owner":
            ticket.get(
                "owner"
            ),

        "reason":
            normalized_reason,

        "status":
            "Open",

        "created_at":
            created_at,
    }


    escalations.append(
        escalation
    )


    save_escalations(
        escalations
    )


    # ------------------------------------------------------------
    # Audit successful action
    # ------------------------------------------------------------

    audit_attempt(
        normalized_ticket,
        normalized_reason,
        True,
        escalation_id=escalation_id,
        error=None
    )


    return {

        "success":
            True,

        "message":
            (
                f"Escalation {escalation_id} "
                f"created for {normalized_ticket}."
            ),

        "escalation":
            escalation,
    }
