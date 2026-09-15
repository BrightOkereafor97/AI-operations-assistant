import json

from pathlib import Path

from stage5_multi_tool import (
    load_planner_model,
    plan_tools,
    parse_tool_plan,
    validate_tool_plan,
    build_tool_argument,
    execute_tool,
    build_final_answer,
)

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


# ================================================================
# PROJECT 3 — STAGE 6
# EXPLICIT AGENT STATE RUNNER
#
# Goal:
# Demonstrate that all important execution information is stored
# in one explicit state object.
#
# Stage 6 DOES NOT introduce an autonomous loop.
# That happens in Stage 7.
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
    / "stage6_state_records.json"
)


# ================================================================
# REPRESENTATIVE TEST REQUESTS
# ================================================================

TEST_REQUESTS = [

    {
        "test_id":
            "STATE-001",

        "description":
            "Independent ticket and knowledge tools",

        "request":
            (
                "Check ticket TKT-005 and tell me "
                "what the company escalation policy says."
            ),
    },


    {
        "test_id":
            "STATE-002",

        "description":
            "Dependent ticket to knowledge chain",

        "request":
            (
                "Check TKT-003 and tell me what "
                "response target applies to its priority."
            ),
    },


    {
        "test_id":
            "STATE-003",

        "description":
            "Dependent claim to calculation chain",

        "request":
            (
                "Get the amount on EXP-010 "
                "and add 7200 to it."
            ),
    },


    {
        "test_id":
            "STATE-004",

        "description":
            "Three-tool request",

        "request":
            (
                "Check TKT-005, add 32000 and 8500, "
                "and tell me what our escalation policy says."
            ),
    },
]


# ================================================================
# RUN ONE REQUEST WITH EXPLICIT STATE
# ================================================================

def run_request_with_state(
    test,
    tokenizer,
    model
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


    # ------------------------------------------------------------
    # 1. INITIALIZE STATE
    # ------------------------------------------------------------

    state = create_agent_state(
        user_request
    )


    add_message(
        state,
        "user",
        user_request
    )


    print(
        "\nInitial state created."
    )


    # ------------------------------------------------------------
    # 2. PLAN TOOLS
    # ------------------------------------------------------------

    raw_plan = plan_tools(
        user_request,
        tokenizer,
        model
    )


    (
        model_labels,
        model_tools
    ) = parse_tool_plan(
        raw_plan
    )


    (
        validated_tools,
        structural_plan,
        validation_reason
    ) = validate_tool_plan(
        user_request,
        model_tools
    )


    print(
        "Raw planner output:",
        raw_plan
    )

    print(
        "Model tools:",
        model_tools
    )

    print(
        "Structural plan:",
        structural_plan
    )

    print(
        "Validated tools:",
        validated_tools
    )

    print(
        "Validation:",
        validation_reason
    )


    # ------------------------------------------------------------
    # Store planner information as an assistant message.
    #
    # We are not introducing extra required state fields yet.
    # ------------------------------------------------------------

    add_message(
        state,
        "assistant",
        (
            "Validated tool plan: "
            +
            " -> ".join(
                validated_tools
            )
        )
    )


    # ------------------------------------------------------------
    # 3. EXECUTION CONTEXT
    #
    # This remains temporary execution memory.
    #
    # The permanent execution history is recorded in state.
    # ------------------------------------------------------------

    execution_context = {}


    successful_results = []

    execution_errors = []


    # ------------------------------------------------------------
    # 4. EXECUTE EACH TOOL
    # ------------------------------------------------------------

    for tool_name in validated_tools:


        # --------------------------------------------------------
        # Every attempted tool call counts as one step.
        # --------------------------------------------------------

        current_step = (
            increment_step(
                state
            )
        )


        (
            argument,
            used_prior_result,
            dependency_note
        ) = build_tool_argument(
            tool_name,
            user_request,
            execution_context
        )


        call_id = record_tool_call(
            state,
            tool_name,
            argument,
            used_prior_result,
            dependency_note
        )


        print(
            f"\nStep {current_step}"
        )

        print(
            "Tool:",
            tool_name
        )

        print(
            "Argument:",
            argument
        )

        print(
            "Used prior result:",
            used_prior_result
        )


        if dependency_note:

            print(
                "Dependency note:",
                dependency_note
            )


        try:

            result = execute_tool(
                tool_name,
                argument
            )


            # ----------------------------------------------------
            # Make result available to later tools.
            # ----------------------------------------------------

            execution_context[
                tool_name
            ] = result


            # ----------------------------------------------------
            # Persist result in explicit agent state.
            # ----------------------------------------------------

            record_tool_result(
                state,
                call_id,
                tool_name,
                result
            )


            add_message(
                state,
                "tool",
                (
                    f"{tool_name} returned: "
                    f"{result}"
                )
            )


            successful_results.append(
                {
                    "tool":
                        tool_name,

                    "argument":
                        argument,

                    "result":
                        result,
                }
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
                tool_name,
                error_text
            )


            execution_errors.append(
                {
                    "tool":
                        tool_name,

                    "argument":
                        argument,

                    "error":
                        error_text,
                }
            )


            print(
                "ERROR:",
                error_text
            )


    # ------------------------------------------------------------
    # 5. BUILD FINAL ANSWER
    # ------------------------------------------------------------

    final_answer = build_final_answer(
        successful_results,
        execution_errors
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


    # ------------------------------------------------------------
    # 6. VALIDATE STATE
    # ------------------------------------------------------------

    validation = validate_agent_state(
        state
    )


    print(
        "\nFinal answer:",
        final_answer
    )


    print(
        "\nSTATE SUMMARY"
    )

    print(
        "-" * 76
    )

    print(
        "Step count:",
        state[
            "step_count"
        ]
    )

    print(
        "Tool calls:",
        len(
            state[
                "tool_calls"
            ]
        )
    )

    print(
        "Tool results:",
        len(
            state[
                "tool_results"
            ]
        )
    )

    print(
        "Errors:",
        len(
            state[
                "errors"
            ]
        )
    )

    print(
        "Messages:",
        len(
            state[
                "messages"
            ]
        )
    )

    print(
        "State valid:",
        validation[
            "valid"
        ]
    )


    print(
        "\nFULL STATE"
    )

    print(
        "-" * 76
    )

    print(
        json.dumps(
            state,
            indent=2,
            ensure_ascii=False,
            default=str
        )
    )


    return {

        "test_id":
            test[
                "test_id"
            ],

        "description":
            test[
                "description"
            ],

        "state_valid":
            validation[
                "valid"
            ],

        "validation_errors":
            validation[
                "errors"
            ],

        "state":
            state,
    }


# ================================================================
# MAIN
# ================================================================

def main():

    print(
        "\nSTAGE 6 — EXPLICIT AGENT STATE"
    )

    print(
        "=" * 76
    )


    tokenizer, model = (
        load_planner_model()
    )


    records = []


    for test in TEST_REQUESTS:

        record = (
            run_request_with_state(
                test,
                tokenizer,
                model
            )
        )

        records.append(
            record
        )


    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            records,
            file,
            indent=2,
            ensure_ascii=False,
            default=str
        )


    valid_states = sum(

        1

        for record in records

        if record[
            "state_valid"
        ]
    )


    print(
        "\n"
        +
        "=" * 76
    )

    print(
        "STAGE 6 RESULTS"
    )

    print(
        "=" * 76
    )


    print(
        "Requests tested:",
        len(
            records
        )
    )


    print(
        "Valid states:",
        (
            f"{valid_states}/"
            f"{len(records)}"
        )
    )


    print(
        "State validation rate:",
        (
            f"{(
                valid_states
                /
                len(records)
                *
                100
            ):.2f}%"
        )
    )


    print(
        "\nSaved state records:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":

    main()