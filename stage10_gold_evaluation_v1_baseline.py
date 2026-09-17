import csv
import io
from contextlib import redirect_stdout
from pathlib import Path

from stage7_agent_loop import (
    load_planner_model,
    run_agent_loop,
)

from stage8_write_action import (
    validate_write_request,
)

from safety_guards import (
    detect_unsupported_write_intent,
)

from write_tools import (
    create_escalation,
    ESCALATION_FILE,
    AUDIT_FILE,
)


# ================================================================
# PROJECT 3 — STAGE 10
# GOLD EVALUATION
#
# Evaluates the fixed Stage 10 benchmark WITHOUT using gold labels
# to choose tools or arguments.
#
# Execution path is chosen from the real request:
#
# unsupported mutation -> safety guard
# escalation request   -> controlled write flow
# everything else      -> bounded read-agent loop
#
# The benchmark is fixed before execution.
# ================================================================


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

RESULTS_DIR = PROJECT_ROOT / "results"

GOLD_FILE = DATA_DIR / "stage10_gold.csv"

DETAIL_FILE = RESULTS_DIR / "stage10_gold_results.csv"

SUMMARY_FILE = RESULTS_DIR / "stage10_gold_summary.csv"


# ================================================================
# FILE SNAPSHOT / RESTORE
#
# Stage 10 write tests must be reproducible without permanently
# replacing the Stage 8 escalation data.
# ================================================================

def snapshot_file(path):
    if path.exists():
        return path.read_text(
            encoding="utf-8"
        )

    return None


def restore_file(
    path,
    snapshot
):
    if snapshot is None:
        if path.exists():
            path.unlink()

        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        snapshot,
        encoding="utf-8"
    )


def reset_benchmark_write_storage():
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
# LOAD GOLD SET
# ================================================================

def load_gold_cases():
    if not GOLD_FILE.exists():
        raise FileNotFoundError(
            "Stage 10 gold file not found. "
            "Run: python data/create_stage10_gold.py"
        )

    with GOLD_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:
        return list(
            csv.DictReader(file)
        )


# ================================================================
# NORMALIZATION
# ================================================================

def split_tools(value):
    if not value:
        return []

    return [
        item.strip()
        for item in value.split("|")
        if item.strip()
    ]


def normalize_first_argument(value):
    if value is None:
        return ""

    if isinstance(value, list):
        normalized = []

        for item in value:
            try:
                number = float(item)

                if number.is_integer():
                    normalized.append(
                        str(int(number))
                    )
                else:
                    normalized.append(
                        str(number)
                    )

            except (
                TypeError,
                ValueError
            ):
                normalized.append(
                    str(item)
                )

        return "|".join(
            normalized
        ).upper()

    return str(
        value
    ).strip().upper()


def parse_bool(value):
    text = str(
        value
    ).strip().upper()

    if text == "TRUE":
        return True

    if text == "FALSE":
        return False

    return None


def classify_write_error(
    error_text
):
    lower = str(
        error_text
    ).lower()

    if "already exists" in lower:
        return "DUPLICATE_REJECTED"

    if "was not found" in lower:
        return "INVALID_RECORD"

    return "WRITE_FAILED"


# ================================================================
# READ-AGENT EXECUTION
# ================================================================

def execute_read_agent(
    user_request,
    max_steps,
    tokenizer,
    model
):
    # Suppress very verbose Stage 7 trace during 64-case evaluation.
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        state = run_agent_loop(
            user_request,
            tokenizer,
            model,
            max_steps=max_steps
        )

    actual_tools = [
        item[
            "tool"
        ]
        for item
        in state[
            "tool_calls"
        ]
    ]

    first_argument = ""

    if state[
        "tool_calls"
    ]:
        first_argument = (
            state[
                "tool_calls"
            ][0][
                "argument"
            ]
        )

    used_prior_result = any(
        bool(
            item.get(
                "used_prior_result"
            )
        )
        for item
        in state[
            "tool_calls"
        ]
    )

    evidence_text = []

    for item in state[
        "tool_results"
    ]:
        if (
            item.get(
                "tool"
            )
            !=
            "search_company_knowledge"
        ):
            continue

        result = item.get(
            "result",
            {}
        )

        for evidence in result.get(
            "evidence",
            []
        ):
            evidence_text.append(
                str(
                    evidence
                )
            )

    return {
        "actual_mode":
            "READ_AGENT",

        "actual_tools":
            actual_tools,

        "first_argument":
            first_argument,

        "dependency_satisfied":
            used_prior_result,

        "stop_reason":
            state.get(
                "stop_reason"
            ),

        "outcome":
            "",

        "final_answer":
            state.get(
                "final_answer",
                ""
            ),

        "evidence_text":
            evidence_text,

        "error_count":
            len(
                state[
                    "errors"
                ]
            ),

        "step_count":
            state[
                "step_count"
            ],
    }


# ================================================================
# CONTROLLED WRITE EXECUTION
# ================================================================

def execute_write(
    user_request
):
    validation = validate_write_request(
        user_request
    )

    if not validation[
        "ready"
    ]:
        return {
            "actual_mode":
                "WRITE",

            "actual_tools":
                [],

            "first_argument":
                "",

            "dependency_satisfied":
                False,

            "stop_reason":
                "",

            "outcome":
                "CLARIFICATION",

            "final_answer":
                validation[
                    "message"
                ],

            "evidence_text":
                [],

            "error_count":
                0,

            "step_count":
                0,
        }

    ticket_id = validation[
        "ticket_id"
    ]

    reason = validation[
        "reason"
    ]

    try:
        result = create_escalation(
            ticket_id,
            reason
        )

        return {
            "actual_mode":
                "WRITE",

            "actual_tools":
                [
                    "create_escalation"
                ],

            "first_argument":
                ticket_id,

            "dependency_satisfied":
                False,

            "stop_reason":
                "",

            "outcome":
                "CREATED",

            "final_answer":
                result[
                    "message"
                ],

            "evidence_text":
                [],

            "error_count":
                0,

            "step_count":
                1,
        }

    except Exception as error:
        error_text = (
            f"{type(error).__name__}: "
            f"{error}"
        )

        return {
            "actual_mode":
                "WRITE",

            "actual_tools":
                [
                    "create_escalation"
                ],

            "first_argument":
                ticket_id,

            "dependency_satisfied":
                False,

            "stop_reason":
                "",

            "outcome":
                classify_write_error(
                    error_text
                ),

            "final_answer":
                (
                    "Escalation was not created. "
                    f"{error_text}"
                ),

            "evidence_text":
                [],

            "error_count":
                1,

            "step_count":
                1,
        }


# ================================================================
# UNIFIED SYSTEM DISPATCH
#
# IMPORTANT:
# This uses only the actual request, not the expected gold labels.
# ================================================================

def execute_system(
    gold_case,
    tokenizer,
    model
):
    user_request = gold_case[
        "user_request"
    ]

    guard = detect_unsupported_write_intent(
        user_request
    )

    if guard[
        "blocked"
    ]:
        return {
            "actual_mode":
                "UNSUPPORTED_WRITE",

            "actual_tools":
                [],

            "first_argument":
                "",

            "dependency_satisfied":
                False,

            "stop_reason":
                "",

            "outcome":
                "BLOCKED_UNSUPPORTED_WRITE",

            "final_answer":
                guard[
                    "message"
                ],

            "evidence_text":
                [],

            "error_count":
                0,

            "step_count":
                0,
        }

    if "escalat" in user_request.lower():
        return execute_write(
            user_request
        )

    max_steps = int(
        gold_case.get(
            "max_steps",
            5
        )
        or
        5
    )

    return execute_read_agent(
        user_request,
        max_steps,
        tokenizer,
        model
    )


# ================================================================
# CASE EVALUATION
# ================================================================

def evaluate_case(
    gold,
    actual
):
    expected_tools = split_tools(
        gold[
            "expected_tools"
        ]
    )

    actual_tools = actual[
        "actual_tools"
    ]

    mode_correct = (
        actual[
            "actual_mode"
        ]
        ==
        gold[
            "expected_mode"
        ]
    )

    exact_tools_correct = (
        actual_tools
        ==
        expected_tools
    )

    required_found = sum(
        1
        for tool
        in expected_tools
        if tool
        in actual_tools
    )

    unnecessary_tools = [
        tool
        for tool
        in actual_tools
        if tool
        not in expected_tools
    ]

    expected_first_argument = (
        gold[
            "expected_first_argument"
        ]
        .strip()
    )

    if expected_first_argument:
        first_argument_correct = (
            normalize_first_argument(
                actual[
                    "first_argument"
                ]
            )
            ==
            normalize_first_argument(
                expected_first_argument
            )
        )
    else:
        first_argument_correct = None

    expected_dependency = parse_bool(
        gold[
            "expected_dependency"
        ]
    )

    if expected_dependency is None:
        dependency_correct = None
    else:
        dependency_correct = (
            actual[
                "dependency_satisfied"
            ]
            ==
            expected_dependency
        )

    expected_stop_reason = (
        gold[
            "expected_stop_reason"
        ]
        .strip()
    )

    if expected_stop_reason:
        stop_correct = (
            actual[
                "stop_reason"
            ]
            ==
            expected_stop_reason
        )
    else:
        stop_correct = None

    expected_outcome = (
        gold[
            "expected_outcome"
        ]
        .strip()
    )

    if expected_outcome:
        outcome_correct = (
            actual[
                "outcome"
            ]
            ==
            expected_outcome
        )
    else:
        outcome_correct = None

    expected_answer_contains = (
        gold[
            "expected_answer_contains"
        ]
        .strip()
    )

    if expected_answer_contains:
        answer_check = (
            expected_answer_contains.lower()
            in
            str(
                actual[
                    "final_answer"
                ]
            ).lower()
        )
    else:
        answer_check = None

    expected_evidence_contains = (
        gold[
            "expected_evidence_contains"
        ]
        .strip()
    )

    if expected_evidence_contains:
        joined_evidence = " ".join(
            actual[
                "evidence_text"
            ]
        )

        evidence_check = (
            expected_evidence_contains.lower()
            in
            joined_evidence.lower()
        )
    else:
        evidence_check = None

    checks = [
        mode_correct,
        exact_tools_correct,
    ]

    for optional_check in [
        first_argument_correct,
        dependency_correct,
        stop_correct,
        outcome_correct,
        answer_check,
        evidence_check,
    ]:
        if optional_check is not None:
            checks.append(
                optional_check
            )

    overall_pass = all(
        checks
    )

    return {
        "case_id":
            gold[
                "case_id"
            ],

        "category":
            gold[
                "category"
            ],

        "user_request":
            gold[
                "user_request"
            ],

        "expected_mode":
            gold[
                "expected_mode"
            ],

        "actual_mode":
            actual[
                "actual_mode"
            ],

        "mode_correct":
            mode_correct,

        "expected_tools":
            "|".join(
                expected_tools
            ),

        "actual_tools":
            "|".join(
                actual_tools
            ),

        "exact_tools_correct":
            exact_tools_correct,

        "required_tools_found":
            required_found,

        "required_tools_total":
            len(
                expected_tools
            ),

        "has_unnecessary_tool":
            bool(
                unnecessary_tools
            ),

        "unnecessary_tools":
            "|".join(
                unnecessary_tools
            ),

        "expected_first_argument":
            expected_first_argument,

        "actual_first_argument":
            normalize_first_argument(
                actual[
                    "first_argument"
                ]
            ),

        "first_argument_correct":
            (
                ""
                if first_argument_correct
                is None
                else
                first_argument_correct
            ),

        "expected_dependency":
            gold[
                "expected_dependency"
            ],

        "dependency_satisfied":
            actual[
                "dependency_satisfied"
            ],

        "dependency_correct":
            (
                ""
                if dependency_correct
                is None
                else
                dependency_correct
            ),

        "expected_stop_reason":
            expected_stop_reason,

        "actual_stop_reason":
            actual[
                "stop_reason"
            ],

        "stop_correct":
            (
                ""
                if stop_correct
                is None
                else
                stop_correct
            ),

        "expected_outcome":
            expected_outcome,

        "actual_outcome":
            actual[
                "outcome"
            ],

        "outcome_correct":
            (
                ""
                if outcome_correct
                is None
                else
                outcome_correct
            ),

        "expected_answer_contains":
            expected_answer_contains,

        "answer_check":
            (
                ""
                if answer_check
                is None
                else
                answer_check
            ),

        "expected_evidence_contains":
            expected_evidence_contains,

        "actual_evidence":
            " || ".join(
                actual[
                    "evidence_text"
                ]
            ),

        "evidence_check":
            (
                ""
                if evidence_check
                is None
                else
                evidence_check
            ),

        "step_count":
            actual[
                "step_count"
            ],

        "error_count":
            actual[
                "error_count"
            ],

        "final_answer":
            actual[
                "final_answer"
            ],

        "overall_pass":
            overall_pass,
    }


# ================================================================
# METRIC HELPERS
# ================================================================

def pct(
    numerator,
    denominator
):
    if denominator == 0:
        return 0.0

    return round(
        numerator
        /
        denominator
        *
        100,
        2
    )


def bool_value(
    value
):
    if value is True:
        return True

    if str(
        value
    ).strip().lower() == "true":
        return True

    return False


# ================================================================
# SUMMARY
# ================================================================

def build_summary(
    rows
):
    total = len(
        rows
    )

    mode_correct = sum(
        1
        for row
        in rows
        if bool_value(
            row[
                "mode_correct"
            ]
        )
    )

    exact_tools = sum(
        1
        for row
        in rows
        if bool_value(
            row[
                "exact_tools_correct"
            ]
        )
    )

    required_found = sum(
        int(
            row[
                "required_tools_found"
            ]
        )
        for row
        in rows
    )

    required_total = sum(
        int(
            row[
                "required_tools_total"
            ]
        )
        for row
        in rows
    )

    unnecessary = sum(
        1
        for row
        in rows
        if bool_value(
            row[
                "has_unnecessary_tool"
            ]
        )
    )

    first_argument_rows = [
        row
        for row
        in rows
        if row[
            "first_argument_correct"
        ]
        !=
        ""
    ]

    first_argument_correct = sum(
        1
        for row
        in first_argument_rows
        if bool_value(
            row[
                "first_argument_correct"
            ]
        )
    )

    dependency_rows = [
        row
        for row
        in rows
        if row[
            "dependency_correct"
        ]
        !=
        ""
    ]

    dependency_correct = sum(
        1
        for row
        in dependency_rows
        if bool_value(
            row[
                "dependency_correct"
            ]
        )
    )

    stop_rows = [
        row
        for row
        in rows
        if row[
            "stop_correct"
        ]
        !=
        ""
    ]

    stop_correct = sum(
        1
        for row
        in stop_rows
        if bool_value(
            row[
                "stop_correct"
            ]
        )
    )

    write_rows = [
        row
        for row
        in rows
        if row[
            "expected_mode"
        ]
        in {
            "WRITE",
            "UNSUPPORTED_WRITE",
        }
    ]

    write_correct = sum(
        1
        for row
        in write_rows
        if (
            row[
                "outcome_correct"
            ]
            !=
            ""
            and
            bool_value(
                row[
                    "outcome_correct"
                ]
            )
        )
    )

    semantic_rows = [
        row
        for row
        in rows
        if (
            row[
                "answer_check"
            ]
            !=
            ""
            or
            row[
                "evidence_check"
            ]
            !=
            ""
        )
    ]

    semantic_correct = 0

    for row in semantic_rows:
        checks = []

        if row[
            "answer_check"
        ] != "":
            checks.append(
                bool_value(
                    row[
                        "answer_check"
                    ]
                )
            )

        if row[
            "evidence_check"
        ] != "":
            checks.append(
                bool_value(
                    row[
                        "evidence_check"
                    ]
                )
            )

        if checks and all(
            checks
        ):
            semantic_correct += 1

    overall_pass = sum(
        1
        for row
        in rows
        if bool_value(
            row[
                "overall_pass"
            ]
        )
    )

    return {
        "gold_cases":
            total,

        "mode_accuracy":
            pct(
                mode_correct,
                total
            ),

        "exact_tool_plan_accuracy":
            pct(
                exact_tools,
                total
            ),

        "required_tool_coverage":
            pct(
                required_found,
                required_total
            ),

        "unnecessary_tool_call_rate":
            pct(
                unnecessary,
                total
            ),

        "first_argument_accuracy":
            pct(
                first_argument_correct,
                len(
                    first_argument_rows
                )
            ),

        "dependency_success_rate":
            pct(
                dependency_correct,
                len(
                    dependency_rows
                )
            ),

        "safe_stop_accuracy":
            pct(
                stop_correct,
                len(
                    stop_rows
                )
            ),

        "write_safety_accuracy":
            pct(
                write_correct,
                len(
                    write_rows
                )
            ),

        "semantic_check_accuracy":
            pct(
                semantic_correct,
                len(
                    semantic_rows
                )
            ),

        "overall_gold_pass_rate":
            pct(
                overall_pass,
                total
            ),

        "passed_cases":
            overall_pass,

        "failed_cases":
            total
            -
            overall_pass,

        "semantic_cases":
            len(
                semantic_rows
            ),
    }


# ================================================================
# SAVE RESULTS
# ================================================================

def save_results(
    rows,
    summary
):
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with DETAIL_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    with SUMMARY_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary.keys()
        )

        writer.writeheader()
        writer.writerow(
            summary
        )


# ================================================================
# MAIN
# ================================================================

def main():
    gold_cases = load_gold_cases()

    print(
        "\nSTAGE 10 — GOLD EVALUATION"
    )

    print(
        "=" * 76
    )

    print(
        "Gold cases:",
        len(
            gold_cases
        )
    )

    print(
        "Loading planner model once..."
    )

    tokenizer, model = (
        load_planner_model()
    )

    escalation_snapshot = snapshot_file(
        ESCALATION_FILE
    )

    audit_snapshot = snapshot_file(
        AUDIT_FILE
    )

    reset_benchmark_write_storage()

    rows = []

    try:
        for index, gold in enumerate(
            gold_cases,
            start=1
        ):
            actual = execute_system(
                gold,
                tokenizer,
                model
            )

            row = evaluate_case(
                gold,
                actual
            )

            rows.append(
                row
            )

            status = (
                "PASS"
                if row[
                    "overall_pass"
                ]
                else
                "FAIL"
            )

            print(
                f"[{index:02d}/{len(gold_cases)}] "
                f"{gold['case_id']} "
                f"{gold['category']} "
                f"-> {status}"
            )

            if not row[
                "overall_pass"
            ]:
                print(
                    "    Expected tools:",
                    row[
                        "expected_tools"
                    ]
                )

                print(
                    "    Actual tools:",
                    row[
                        "actual_tools"
                    ]
                )

                print(
                    "    Expected stop:",
                    row[
                        "expected_stop_reason"
                    ]
                )

                print(
                    "    Actual stop:",
                    row[
                        "actual_stop_reason"
                    ]
                )

                print(
                    "    Expected outcome:",
                    row[
                        "expected_outcome"
                    ]
                )

                print(
                    "    Actual outcome:",
                    row[
                        "actual_outcome"
                    ]
                )

                if row[
                    "expected_evidence_contains"
                ]:
                    print(
                        "    Expected evidence contains:",
                        row[
                            "expected_evidence_contains"
                        ]
                    )

                    print(
                        "    Actual evidence:",
                        row[
                            "actual_evidence"
                        ]
                    )

                if row[
                    "expected_answer_contains"
                ]:
                    print(
                        "    Expected answer contains:",
                        row[
                            "expected_answer_contains"
                        ]
                    )

                    print(
                        "    Final answer:",
                        row[
                            "final_answer"
                        ]
                    )

        summary = build_summary(
            rows
        )

        save_results(
            rows,
            summary
        )

    finally:
        restore_file(
            ESCALATION_FILE,
            escalation_snapshot
        )

        restore_file(
            AUDIT_FILE,
            audit_snapshot
        )

    print(
        "\n"
        +
        "=" * 76
    )

    print(
        "STAGE 10 RESULTS"
    )

    print(
        "=" * 76
    )

    for key, value in summary.items():
        print(
            f"{key}: {value}"
        )

    failed_ids = [
        row[
            "case_id"
        ]
        for row
        in rows
        if not row[
            "overall_pass"
        ]
    ]

    print(
        "\nFailed case IDs:",
        (
            failed_ids
            if failed_ids
            else
            "None"
        )
    )

    print(
        "\nDetailed results:"
    )

    print(
        DETAIL_FILE
    )

    print(
        "\nSummary:"
    )

    print(
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()
