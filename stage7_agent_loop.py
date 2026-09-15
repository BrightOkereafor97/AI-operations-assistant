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

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "stage7_agent_loop_results.json"

MAX_STEPS = 5

STOP_READY = "READY"
STOP_CLARIFICATION = "CLARIFICATION"
STOP_UNRECOVERABLE = "UNRECOVERABLE"
STOP_MAX_STEPS = "MAX_STEPS"

TEST_REQUESTS = [
    {
        "test_id": "LOOP-001",
        "description": "Independent two-tool request reaches READY",
        "request": (
            "Check ticket TKT-005 and tell me "
            "what the company escalation policy says."
        ),
        "expected_stop_reason": STOP_READY,
        "max_steps": MAX_STEPS,
    },
    {
        "test_id": "LOOP-002",
        "description": "Dependent tool chain reaches READY",
        "request": (
            "Get the amount on EXP-010 "
            "and add 7200 to it."
        ),
        "expected_stop_reason": STOP_READY,
        "max_steps": MAX_STEPS,
    },
    {
        "test_id": "LOOP-003",
        "description": "Missing ticket identifier requires clarification",
        "request": "Check the ticket and tell me who owns it.",
        "expected_stop_reason": STOP_CLARIFICATION,
        "max_steps": MAX_STEPS,
    },
    {
        "test_id": "LOOP-004",
        "description": "Missing record causes unrecoverable stop",
        "request": "Check ticket TKT-999.",
        "expected_stop_reason": STOP_UNRECOVERABLE,
        "max_steps": MAX_STEPS,
    },
    {
        "test_id": "LOOP-005",
        "description": "Safety limit stops unfinished multi-tool execution",
        "request": (
            "Check TKT-005, add 32000 and 8500, "
            "and tell me what our escalation policy says."
        ),
        "expected_stop_reason": STOP_MAX_STEPS,
        "max_steps": 2,
    },
]


def initialize_loop_state(user_request, max_steps):
    state = create_agent_state(user_request)
    state["status"] = "RUNNING"
    state["stop_reason"] = None
    state["planned_tools"] = []
    state["next_tool_index"] = 0
    state["max_steps"] = max_steps
    add_message(state, "user", user_request)
    return state


def build_execution_context_from_state(state):
    context = {}
    for item in state["tool_results"]:
        context[item["tool"]] = item["result"]
    return context


def successful_results_from_state(state):
    return [
        {
            "tool": item["tool"],
            "result": item["result"],
        }
        for item in state["tool_results"]
    ]


def errors_from_state(state):
    return [
        {
            "tool": item["tool"],
            "error": item["error"],
        }
        for item in state["errors"]
    ]


def clarification_message_for(tool_name, argument):
    if tool_name == "get_ticket" and not argument:
        return (
            "Which ticket should I check? "
            "Please provide the ticket ID, for example TKT-005."
        )

    if tool_name == "get_expense_claim" and not argument:
        return (
            "Which expense claim should I check? "
            "Please provide the claim ID, for example EXP-010."
        )

    if tool_name == "calculate_expense" and not argument:
        return (
            "Which amounts should I calculate? "
            "Please provide the required values."
        )

    if tool_name == "search_company_knowledge" and not argument:
        return (
            "What company policy or process "
            "would you like me to look up?"
        )

    return None


def stop_with_answer(state, stop_reason, status, answer):
    state["stop_reason"] = stop_reason
    state["status"] = status
    set_final_answer(state, answer)
    add_message(state, "assistant", answer)


def stop_ready(state):
    final_answer = build_final_answer(
        successful_results_from_state(state),
        errors_from_state(state),
    )
    stop_with_answer(
        state,
        STOP_READY,
        "COMPLETED",
        final_answer,
    )


def stop_for_clarification(state, message):
    stop_with_answer(
        state,
        STOP_CLARIFICATION,
        "WAITING_FOR_CLARIFICATION",
        message,
    )


def stop_unrecoverable(state, tool_name, error_text):
    answer = (
        "I could not continue because "
        f"{tool_name} failed: {error_text}"
    )
    stop_with_answer(
        state,
        STOP_UNRECOVERABLE,
        "FAILED",
        answer,
    )


def stop_max_steps(state):
    max_steps = state["max_steps"]
    answer = (
        "I stopped execution after reaching "
        f"the maximum of {max_steps} tool steps. "
        "The request was not fully completed."
    )
    stop_with_answer(
        state,
        STOP_MAX_STEPS,
        "STOPPED_MAX_STEPS",
        answer,
    )


def prepare_plan(state, tokenizer, model):
    user_request = state["user_request"]

    raw_plan = plan_tools(
        user_request,
        tokenizer,
        model,
    )

    model_labels, model_tools = parse_tool_plan(
        raw_plan
    )

    validated_tools, structural_plan, validation_reason = (
        validate_tool_plan(
            user_request,
            model_tools,
        )
    )

    state["planned_tools"] = validated_tools

    state["planner"] = {
        "raw_output": raw_plan,
        "model_labels": model_labels,
        "model_tools": model_tools,
        "structural_plan": structural_plan,
        "validation_reason": validation_reason,
    }

    add_message(
        state,
        "assistant",
        (
            "Validated tool plan: "
            + (
                " -> ".join(validated_tools)
                if validated_tools
                else "NO_TOOL"
            )
        ),
    )

    print("Raw planner output:", raw_plan)
    print("Model tools:", model_tools)
    print("Structural plan:", structural_plan)
    print("Validated tools:", validated_tools)
    print("Validation:", validation_reason)


def determine_next_action(state):
    if state["stop_reason"] is not None:
        return {"action": "STOP"}

    planned_tools = state["planned_tools"]
    next_index = state["next_tool_index"]

    if next_index >= len(planned_tools):
        return {"action": STOP_READY}

    if state["step_count"] >= state["max_steps"]:
        return {"action": STOP_MAX_STEPS}

    next_tool = planned_tools[next_index]

    execution_context = (
        build_execution_context_from_state(
            state
        )
    )

    argument, used_prior_result, dependency_note = (
        build_tool_argument(
            next_tool,
            state["user_request"],
            execution_context,
        )
    )

    clarification = clarification_message_for(
        next_tool,
        argument,
    )

    if clarification:
        return {
            "action": STOP_CLARIFICATION,
            "message": clarification,
            "tool": next_tool,
            "argument": argument,
        }

    return {
        "action": "EXECUTE_TOOL",
        "tool": next_tool,
        "argument": argument,
        "used_prior_result": used_prior_result,
        "dependency_note": dependency_note,
    }


def execute_one_step(state, decision):
    tool_name = decision["tool"]
    argument = decision["argument"]
    used_prior_result = decision["used_prior_result"]
    dependency_note = decision["dependency_note"]

    current_step = increment_step(state)

    call_id = record_tool_call(
        state,
        tool_name,
        argument,
        used_prior_result,
        dependency_note,
    )

    print(f"\nStep {current_step}")
    print("Tool:", tool_name)
    print("Argument:", argument)
    print("Used prior result:", used_prior_result)

    if dependency_note:
        print(
            "Dependency note:",
            dependency_note,
        )

    try:
        result = execute_tool(
            tool_name,
            argument,
        )

        record_tool_result(
            state,
            call_id,
            tool_name,
            result,
        )

        add_message(
            state,
            "tool",
            f"{tool_name} returned: {result}",
        )

        state["next_tool_index"] += 1

        print("Result:", result)

        return True

    except Exception as error:
        error_text = (
            f"{type(error).__name__}: {error}"
        )

        record_error(
            state,
            call_id,
            tool_name,
            error_text,
        )

        print("ERROR:", error_text)

        stop_unrecoverable(
            state,
            tool_name,
            error_text,
        )

        return False


def run_agent_loop(
    user_request,
    tokenizer,
    model,
    max_steps=MAX_STEPS,
):
    state = initialize_loop_state(
        user_request,
        max_steps,
    )

    prepare_plan(
        state,
        tokenizer,
        model,
    )

    if not state["planned_tools"]:
        stop_with_answer(
            state,
            STOP_READY,
            "COMPLETED",
            (
                "I cannot complete this request "
                "with the currently available tools."
            ),
        )
        return state

    while state["stop_reason"] is None:
        decision = determine_next_action(
            state
        )

        print(
            "\nLoop decision:",
            decision["action"],
        )

        if decision["action"] == STOP_READY:
            stop_ready(state)
            break

        if decision["action"] == STOP_CLARIFICATION:
            stop_for_clarification(
                state,
                decision["message"],
            )
            break

        if decision["action"] == STOP_MAX_STEPS:
            stop_max_steps(state)
            break

        if decision["action"] == "EXECUTE_TOOL":
            successful = execute_one_step(
                state,
                decision,
            )

            if not successful:
                break

            continue

        stop_with_answer(
            state,
            STOP_UNRECOVERABLE,
            "FAILED",
            (
                "The agent controller produced "
                "an unknown action and stopped safely."
            ),
        )
        break

    return state


def evaluate_test(test, state):
    state_validation = validate_agent_state(
        state
    )

    expected_stop = test[
        "expected_stop_reason"
    ]

    actual_stop = state[
        "stop_reason"
    ]

    stop_reason_correct = (
        expected_stop
        ==
        actual_stop
    )

    stayed_within_limit = (
        state["step_count"]
        <=
        state["max_steps"]
    )

    passed = (
        state_validation["valid"]
        and
        stop_reason_correct
        and
        stayed_within_limit
    )

    return {
        "state_valid": state_validation["valid"],
        "validation_errors": state_validation["errors"],
        "expected_stop_reason": expected_stop,
        "actual_stop_reason": actual_stop,
        "stop_reason_correct": stop_reason_correct,
        "stayed_within_limit": stayed_within_limit,
        "passed": passed,
    }


def run_test(test, tokenizer, model):
    print("\n" + "=" * 76)

    print(
        test["test_id"],
        "-",
        test["description"],
    )

    print(
        "User:",
        test["request"],
    )

    print(
        "Max steps:",
        test["max_steps"],
    )

    state = run_agent_loop(
        test["request"],
        tokenizer,
        model,
        max_steps=test["max_steps"],
    )

    evaluation = evaluate_test(
        test,
        state,
    )

    print("\nFINAL LOOP STATE")
    print("-" * 76)

    print("Status:", state["status"])
    print("Stop reason:", state["stop_reason"])
    print("Step count:", state["step_count"])
    print("Tool calls:", len(state["tool_calls"]))
    print("Tool results:", len(state["tool_results"]))
    print("Errors:", len(state["errors"]))
    print("Final answer:", state["final_answer"])
    print("State valid:", evaluation["state_valid"])
    print("Expected stop:", evaluation["expected_stop_reason"])
    print("Stop correct:", evaluation["stop_reason_correct"])
    print("Within step limit:", evaluation["stayed_within_limit"])
    print("TEST PASSED:", evaluation["passed"])

    return {
        "test_id": test["test_id"],
        "description": test["description"],
        "evaluation": evaluation,
        "state": state,
    }


def main():
    print(
        "\nSTAGE 7 — BOUNDED AGENT LOOP"
    )

    print("=" * 76)

    print(
        "Default MAX_STEPS:",
        MAX_STEPS,
    )

    print(
        "Tests:",
        len(TEST_REQUESTS),
    )

    tokenizer, model = load_planner_model()

    records = []

    for test in TEST_REQUESTS:
        records.append(
            run_test(
                test,
                tokenizer,
                model,
            )
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            records,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    passed = sum(
        1
        for record in records
        if record["evaluation"]["passed"]
    )

    stop_counts = {}

    for record in records:
        stop_reason = (
            record["state"]["stop_reason"]
        )

        stop_counts[stop_reason] = (
            stop_counts.get(
                stop_reason,
                0,
            )
            + 1
        )

    print("\n" + "=" * 76)
    print("STAGE 7 RESULTS")
    print("=" * 76)

    print(
        "Tests passed:",
        f"{passed}/{len(records)}",
    )

    print(
        "Pass rate:",
        f"{(passed / len(records) * 100):.2f}%",
    )

    print(
        "Stop reasons:",
        stop_counts,
    )

    print(
        "\nSaved results:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
