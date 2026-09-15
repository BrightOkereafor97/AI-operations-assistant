import json
import re

from pathlib import Path

from state import (
    create_agent_state,
    add_message,
    increment_step,
    record_tool_call,
    record_tool_result,
    record_error,
    set_final_answer,
    validate_agent_state,
)

from write_tools import (
    create_escalation,
    list_escalations,
    ESCALATION_FILE,
    AUDIT_FILE,
)


# ================================================================
# PROJECT 3 — STAGE 8
# CONTROLLED WRITE/ACTION TOOL TEST
#
# Goal:
# Introduce the first state-changing tool safely.
#
# Controls demonstrated:
# - required ticket
# - required reason
# - ticket existence check
# - duplicate prevention
# - clarification before unsafe/incomplete writes
# - audit timestamp
# - audit success/failure
# - fictional local storage only
# ================================================================


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "stage8_write_action_results.json"
)


TICKET_ID_PATTERN = re.compile(
    r"\bTKT-\d+\b",
    re.IGNORECASE
)


# ================================================================
# FIXED STAGE 8 TEST SET
# ================================================================

TEST_REQUESTS = [

    {
        "test_id":
            "WRITE-001",

        "description":
            "Valid escalation creation",

        "request":
            (
                "Escalate TKT-003 because "
                "the production dashboard is unavailable."
            ),

        "expected_outcome":
            "CREATED",
    },


    {
        "test_id":
            "WRITE-002",

        "description":
            "Duplicate escalation is rejected",

        "request":
            (
                "Escalate TKT-003 because "
                "the customer reports wider business impact."
            ),

        "expected_outcome":
            "DUPLICATE_REJECTED",
    },


    {
        "test_id":
            "WRITE-003",

        "description":
            "Missing reason requires clarification",

        "request":
            "Escalate TKT-005.",

        "expected_outcome":
            "CLARIFICATION",
    },


    {
        "test_id":
            "WRITE-004",

        "description":
            "Missing ticket ID requires clarification",

        "request":
            (
                "Escalate the ticket because "
                "the customer cannot work."
            ),

        "expected_outcome":
            "CLARIFICATION",
    },


    {
        "test_id":
            "WRITE-005",

        "description":
            "Nonexistent ticket is rejected",

        "request":
            (
                "Escalate TKT-999 because "
                "the production service is unavailable."
            ),

        "expected_outcome":
            "INVALID_RECORD",
    },


    {
        "test_id":
            "WRITE-006",

        "description":
            "Second valid escalation creation",

        "request":
            (
                "Escalate TKT-017 because "
                "possible unauthorized access requires immediate review."
            ),

        "expected_outcome":
            "CREATED",
    },


    {
        "test_id":
            "WRITE-007",

        "description":
            "Ambiguous multiple tickets require clarification",

        "request":
            (
                "Escalate TKT-010 and TKT-011 because "
                "the customer reports a serious issue."
            ),

        "expected_outcome":
            "CLARIFICATION",
    },
]


# ================================================================
# TEST STORAGE RESET
#
# This reset exists ONLY so repeated benchmark runs are reproducible.
# The create_escalation tool itself never clears prior data.
# ================================================================

def reset_stage8_test_storage():

    ESCALATION_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    ESCALATION_FILE.write_text(
        "[]",
        encoding="utf-8"
    )


    if AUDIT_FILE.exists():

        AUDIT_FILE.unlink()


# ================================================================
# EXTRACT TICKET IDS
# ================================================================

def extract_ticket_ids(
    user_request
):

    matches = (
        TICKET_ID_PATTERN.findall(
            user_request
        )
    )


    unique_ids = []


    for match in matches:

        normalized = (
            match.upper()
        )

        if normalized not in unique_ids:

            unique_ids.append(
                normalized
            )


    return unique_ids


# ================================================================
# EXTRACT REASON
# ================================================================

def extract_reason(
    user_request
):

    patterns = [

        r"\bbecause\s+(.+)$",

        r"\breason\s*(?:is|:)\s*(.+)$",
    ]


    for pattern in patterns:

        match = re.search(
            pattern,
            user_request,
            re.IGNORECASE
        )


        if match:

            reason = (
                match.group(1)
                .strip()
                .rstrip(".")
                .strip()
            )


            if reason:

                return reason


    return None


# ================================================================
# VALIDATE REQUEST BEFORE WRITE
# ================================================================

def validate_write_request(
    user_request
):

    ticket_ids = (
        extract_ticket_ids(
            user_request
        )
    )


    if len(
        ticket_ids
    ) == 0:

        return {

            "ready":
                False,

            "outcome":
                "CLARIFICATION",

            "message":
                (
                    "Which ticket should I escalate? "
                    "Please provide one ticket ID, "
                    "for example TKT-003."
                ),
        }


    if len(
        ticket_ids
    ) > 1:

        return {

            "ready":
                False,

            "outcome":
                "CLARIFICATION",

            "message":
                (
                    "I found more than one ticket ID. "
                    "Please specify exactly one ticket "
                    "to escalate."
                ),
        }


    reason = (
        extract_reason(
            user_request
        )
    )


    if not reason:

        return {

            "ready":
                False,

            "outcome":
                "CLARIFICATION",

            "message":
                (
                    "What is the reason for escalating "
                    f"{ticket_ids[0]}?"
                ),
        }


    return {

        "ready":
            True,

        "ticket_id":
            ticket_ids[0],

        "reason":
            reason,
    }


# ================================================================
# CLASSIFY WRITE FAILURE
# ================================================================

def classify_write_error(
    error_text
):

    lower_error = (
        error_text.lower()
    )


    if (
        "already exists"
        in lower_error
    ):

        return (
            "DUPLICATE_REJECTED"
        )


    if (
        "was not found"
        in lower_error
    ):

        return (
            "INVALID_RECORD"
        )


    return (
        "WRITE_FAILED"
    )


# ================================================================
# RUN ONE WRITE REQUEST
# ================================================================

def run_write_request(
    test
):

    user_request = (
        test[
            "request"
        ]
    )


    print(
        "\n"
        +
        "=" * 76
    )


    print(
        test[
            "test_id"
        ],
        "-",
        test[
            "description"
        ]
    )


    print(
        "User:",
        user_request
    )


    state = create_agent_state(
        user_request
    )


    state[
        "status"
    ] = "RUNNING"


    state[
        "outcome"
    ] = None


    add_message(
        state,
        "user",
        user_request
    )


    validation = (
        validate_write_request(
            user_request
        )
    )


    # ------------------------------------------------------------
    # Clarification happens BEFORE any write tool call.
    # ------------------------------------------------------------

    if not validation[
        "ready"
    ]:

        outcome = validation[
            "outcome"
        ]


        final_answer = validation[
            "message"
        ]


        state[
            "status"
        ] = (
            "WAITING_FOR_CLARIFICATION"
        )


        state[
            "outcome"
        ] = outcome


        set_final_answer(
            state,
            final_answer
        )


        add_message(
            state,
            "assistant",
            final_answer
        )


        state_validation = (
            validate_agent_state(
                state
            )
        )


        passed = (
            outcome
            ==
            test[
                "expected_outcome"
            ]
        )


        print(
            "Outcome:",
            outcome
        )


        print(
            "Tool called: False"
        )


        print(
            "Final answer:",
            final_answer
        )


        print(
            "State valid:",
            state_validation[
                "valid"
            ]
        )


        print(
            "TEST PASSED:",
            passed
        )


        return {

            "test_id":
                test[
                    "test_id"
                ],

            "expected_outcome":
                test[
                    "expected_outcome"
                ],

            "actual_outcome":
                outcome,

            "passed":
                passed,

            "state_valid":
                state_validation[
                    "valid"
                ],

            "state":
                state,
        }


    # ------------------------------------------------------------
    # Request has sufficient information.
    # Attempt controlled write.
    # ------------------------------------------------------------

    ticket_id = validation[
        "ticket_id"
    ]


    reason = validation[
        "reason"
    ]


    step = increment_step(
        state
    )


    argument = {

        "ticket_id":
            ticket_id,

        "reason":
            reason,
    }


    call_id = record_tool_call(
        state,
        "create_escalation",
        argument,
        used_prior_result=False,
        dependency_note=None
    )


    print(
        "Step:",
        step
    )


    print(
        "Tool:",
        "create_escalation"
    )


    print(
        "Argument:",
        argument
    )


    try:

        result = create_escalation(
            ticket_id,
            reason
        )


        record_tool_result(
            state,
            call_id,
            "create_escalation",
            result
        )


        add_message(
            state,
            "tool",
            (
                "create_escalation returned: "
                f"{result}"
            )
        )


        outcome = (
            "CREATED"
        )


        state[
            "status"
        ] = "COMPLETED"


        state[
            "outcome"
        ] = outcome


        final_answer = (
            result[
                "message"
            ]
        )


        set_final_answer(
            state,
            final_answer
        )


        add_message(
            state,
            "assistant",
            final_answer
        )


        print(
            "Result:",
            result
        )


    except Exception as error:

        error_text = (
            f"{type(error).__name__}: "
            f"{error}"
        )


        record_error(
            state,
            call_id,
            "create_escalation",
            error_text
        )


        outcome = (
            classify_write_error(
                error_text
            )
        )


        state[
            "status"
        ] = "FAILED"


        state[
            "outcome"
        ] = outcome


        final_answer = (
            "Escalation was not created. "
            f"{error_text}"
        )


        set_final_answer(
            state,
            final_answer
        )


        add_message(
            state,
            "assistant",
            final_answer
        )


        print(
            "ERROR:",
            error_text
        )


    state_validation = (
        validate_agent_state(
            state
        )
    )


    passed = (
        outcome
        ==
        test[
            "expected_outcome"
        ]
    )


    print(
        "Outcome:",
        outcome
    )


    print(
        "Final answer:",
        state[
            "final_answer"
        ]
    )


    print(
        "State valid:",
        state_validation[
            "valid"
        ]
    )


    print(
        "TEST PASSED:",
        passed
    )


    return {

        "test_id":
            test[
                "test_id"
            ],

        "expected_outcome":
            test[
                "expected_outcome"
            ],

        "actual_outcome":
            outcome,

        "passed":
            passed,

        "state_valid":
            state_validation[
                "valid"
            ],

        "state":
            state,
    }


# ================================================================
# READ AUDIT LOG
# ================================================================

def read_audit_log():

    if not AUDIT_FILE.exists():

        return []


    records = []


    with AUDIT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            stripped = (
                line.strip()
            )


            if not stripped:

                continue


            records.append(
                json.loads(
                    stripped
                )
            )


    return records


# ================================================================
# AUDIT VALIDATION
# ================================================================

def validate_audit_records(
    records
):

    required_fields = {

        "timestamp",

        "action",

        "ticket_id",

        "reason",

        "success",

        "escalation_id",

        "error",
    }


    for record in records:

        if not required_fields.issubset(
            record.keys()
        ):

            return False


        if not record[
            "timestamp"
        ]:

            return False


        if not isinstance(
            record[
                "success"
            ],
            bool
        ):

            return False


    return True


# ================================================================
# MAIN
# ================================================================

def main():

    print(
        "\nSTAGE 8 — CONTROLLED WRITE ACTION"
    )


    print(
        "=" * 76
    )


    print(
        "Tests:",
        len(
            TEST_REQUESTS
        )
    )


    print(
        "Storage:",
        ESCALATION_FILE
    )


    print(
        "Audit log:",
        AUDIT_FILE
    )


    reset_stage8_test_storage()


    records = []


    for test in TEST_REQUESTS:

        records.append(
            run_write_request(
                test
            )
        )


    escalations = (
        list_escalations()
    )


    audit_records = (
        read_audit_log()
    )


    audit_valid = (
        validate_audit_records(
            audit_records
        )
    )


    passed = sum(

        1

        for record
        in records

        if record[
            "passed"
        ]
    )


    created = sum(

        1

        for record
        in records

        if (
            record[
                "actual_outcome"
            ]
            ==
            "CREATED"
        )
    )


    clarifications = sum(

        1

        for record
        in records

        if (
            record[
                "actual_outcome"
            ]
            ==
            "CLARIFICATION"
        )
    )


    rejected_writes = sum(

        1

        for record
        in records

        if (
            record[
                "actual_outcome"
            ]
            in {
                "DUPLICATE_REJECTED",
                "INVALID_RECORD",
                "WRITE_FAILED",
            }
        )
    )


    audit_success = sum(

        1

        for record
        in audit_records

        if record[
            "success"
        ]
    )


    audit_failure = sum(

        1

        for record
        in audit_records

        if not record[
            "success"
        ]
    )


    summary = {

        "tests":
            len(
                records
            ),

        "passed":
            passed,

        "pass_rate":
            round(
                (
                    passed
                    /
                    len(
                        records
                    )
                    *
                    100
                ),
                2
            ),

        "created":
            created,

        "clarifications":
            clarifications,

        "rejected_write_attempts":
            rejected_writes,

        "stored_escalations":
            len(
                escalations
            ),

        "audit_attempts":
            len(
                audit_records
            ),

        "audit_success":
            audit_success,

        "audit_failure":
            audit_failure,

        "audit_valid":
            audit_valid,
    }


    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    output = {

        "summary":
            summary,

        "tests":
            records,

        "escalations":
            escalations,

        "audit_records":
            audit_records,
    }


    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
            default=str
        )


    print(
        "\n"
        +
        "=" * 76
    )


    print(
        "STAGE 8 RESULTS"
    )


    print(
        "=" * 76
    )


    print(
        "Tests passed:",
        (
            f"{passed}/"
            f"{len(records)}"
        )
    )


    print(
        "Pass rate:",
        f"{summary['pass_rate']:.2f}%"
    )


    print(
        "Escalations created:",
        created
    )


    print(
        "Clarifications:",
        clarifications
    )


    print(
        "Rejected write attempts:",
        rejected_writes
    )


    print(
        "Stored escalations:",
        len(
            escalations
        )
    )


    print(
        "Audit attempts:",
        len(
            audit_records
        )
    )


    print(
        "Audit success:",
        audit_success
    )


    print(
        "Audit failure:",
        audit_failure
    )


    print(
        "Audit structure valid:",
        audit_valid
    )


    print(
        "\nSaved results:"
    )


    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":

    main()
