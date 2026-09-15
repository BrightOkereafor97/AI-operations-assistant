from copy import deepcopy


# ================================================================
# PROJECT 3 — STAGE 6
# EXPLICIT AGENT STATE
#
# This file defines the shared state object used by the agent.
#
# The state acts as the agent's structured working memory during
# one request.
# ================================================================


REQUIRED_STATE_FIELDS = [
    "user_request",
    "messages",
    "tool_calls",
    "tool_results",
    "step_count",
    "errors",
    "final_answer",
]


# ================================================================
# CREATE STATE
# ================================================================

def create_agent_state(
    user_request
):

    if not isinstance(
        user_request,
        str
    ):

        raise TypeError(
            "user_request must be a string."
        )


    if not user_request.strip():

        raise ValueError(
            "user_request cannot be empty."
        )


    return {

        "user_request":
            user_request.strip(),

        "messages":
            [],

        "tool_calls":
            [],

        "tool_results":
            [],

        "step_count":
            0,

        "errors":
            [],

        "final_answer":
            None,
    }


# ================================================================
# ADD MESSAGE
# ================================================================

def add_message(
    state,
    role,
    content
):

    if role not in {
        "user",
        "assistant",
        "tool",
        "system"
    }:

        raise ValueError(
            f"Unsupported message role: {role}"
        )


    state[
        "messages"
    ].append(
        {
            "role":
                role,

            "content":
                str(content),
        }
    )


# ================================================================
# INCREMENT STEP
# ================================================================

def increment_step(
    state
):

    state[
        "step_count"
    ] += 1

    return state[
        "step_count"
    ]


# ================================================================
# RECORD TOOL CALL
# ================================================================

def record_tool_call(
    state,
    tool_name,
    argument,
    used_prior_result=False,
    dependency_note=None
):

    call_id = (
        len(
            state[
                "tool_calls"
            ]
        )
        +
        1
    )


    call_record = {

        "call_id":
            call_id,

        "step":
            state[
                "step_count"
            ],

        "tool":
            tool_name,

        "argument":
            deepcopy(
                argument
            ),

        "used_prior_result":
            bool(
                used_prior_result
            ),

        "dependency_note":
            dependency_note,
    }


    state[
        "tool_calls"
    ].append(
        call_record
    )


    return call_id


# ================================================================
# RECORD TOOL RESULT
# ================================================================

def record_tool_result(
    state,
    call_id,
    tool_name,
    result
):

    state[
        "tool_results"
    ].append(
        {
            "call_id":
                call_id,

            "tool":
                tool_name,

            "result":
                deepcopy(
                    result
                ),
        }
    )


# ================================================================
# RECORD ERROR
# ================================================================

def record_error(
    state,
    call_id,
    tool_name,
    error
):

    state[
        "errors"
    ].append(
        {
            "call_id":
                call_id,

            "tool":
                tool_name,

            "error":
                str(error),
        }
    )


# ================================================================
# SET FINAL ANSWER
# ================================================================

def set_final_answer(
    state,
    answer
):

    state[
        "final_answer"
    ] = str(
        answer
    )


# ================================================================
# VALIDATE STATE STRUCTURE
# ================================================================

def validate_agent_state(
    state
):

    missing_fields = [

        field

        for field
        in REQUIRED_STATE_FIELDS

        if field not in state
    ]


    if missing_fields:

        return {

            "valid":
                False,

            "missing_fields":
                missing_fields,

            "errors":
                [
                    (
                        "Missing required state fields: "
                        +
                        ", ".join(
                            missing_fields
                        )
                    )
                ],
        }


    validation_errors = []


    if not isinstance(
        state[
            "user_request"
        ],
        str
    ):

        validation_errors.append(
            "user_request must be a string."
        )


    if not isinstance(
        state[
            "messages"
        ],
        list
    ):

        validation_errors.append(
            "messages must be a list."
        )


    if not isinstance(
        state[
            "tool_calls"
        ],
        list
    ):

        validation_errors.append(
            "tool_calls must be a list."
        )


    if not isinstance(
        state[
            "tool_results"
        ],
        list
    ):

        validation_errors.append(
            "tool_results must be a list."
        )


    if not isinstance(
        state[
            "step_count"
        ],
        int
    ):

        validation_errors.append(
            "step_count must be an integer."
        )


    if not isinstance(
        state[
            "errors"
        ],
        list
    ):

        validation_errors.append(
            "errors must be a list."
        )


    return {

        "valid":
            len(
                validation_errors
            )
            ==
            0,

        "missing_fields":
            [],

        "errors":
            validation_errors,
    }