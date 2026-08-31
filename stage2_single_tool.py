import csv
import json
import re
from pathlib import Path

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from tools import get_ticket


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "stage2_requests.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_FILE = (
    RESULTS_DIR
    / "single_tool_tests.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "single_tool_summary.csv"
)


# ================================================================
# MODEL
# ================================================================

MODEL_NAME = "google/flan-t5-base"


# ================================================================
# LOAD MODEL
# ================================================================

def load_model():

    print(
        f"\nLoading router model: {MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = (
        AutoModelForSeq2SeqLM
        .from_pretrained(
            MODEL_NAME
        )
    )

    model.eval()

    print(
        "Router model ready."
    )

    return tokenizer, model


# ================================================================
# ROUTER PROMPT
#
# This restores the prompt style that gave us
# 100% tool-selection accuracy in V3.
# ================================================================

def create_router_prompt(
    user_request
):

    return f"""
Classify this user request.

There are only two possible outputs.

If the request asks for information about a specific support ticket,
output exactly:

CALL|get_ticket|TKT-NNN

Examples:

Request: Check ticket 5.
Output: CALL|get_ticket|TKT-NNN

Request: Who owns support case #10?
Output: CALL|get_ticket|TKT-NNN

Request: Show me TKT-019.
Output: CALL|get_ticket|TKT-NNN

Request: What is the status of ticket number 2?
Output: CALL|get_ticket|TKT-NNN

Request: Find ticket 20 and tell me its status.
Output: CALL|get_ticket|TKT-NNN


If the request is NOT about looking up a specific support ticket,
output exactly:

NO_TOOL

Examples:

Request: What is our annual leave policy?
Output: NO_TOOL

Request: Add 32000 and 8500.
Output: NO_TOOL


Request: {user_request}

Output:
""".strip()


# ================================================================
# ARGUMENT EXTRACTION PROMPT
# ================================================================

def create_argument_prompt(
    user_request
):

    return f"""
Your only job is to copy the support ticket number from the request.

Return digits only.

Do not return TKT.
Do not return words.
Do not explain.

Examples:

Request: Check support case #3 for me.
Number: 3

Request: Can you pull up TKT-019?
Number: 19

Request: Who owns ticket 10?
Number: 10

Request: Show me the details for case 7.
Number: 7

Request: Look up ticket number 12.
Number: 12

Request: I need the priority and owner for TKT-017.
Number: 17

Request: Please check what happened on support case 2.
Number: 2

Request: Find ticket 20 and tell me its status.
Number: 20


Request: {user_request}

Number:
""".strip()


# ================================================================
# MODEL GENERATION
# ================================================================

def generate_model_output(
    prompt,
    tokenizer,
    model,
    max_new_tokens=20
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    return tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    ).strip()


# ================================================================
# NORMALIZE TICKET NUMBER
# ================================================================

def normalize_ticket_id(
    value
):

    if value is None:
        return None

    text = str(value).strip()

    numbers = re.findall(
        r"\d+",
        text
    )

    if not numbers:
        return None

    number = int(
        numbers[-1]
    )

    return f"TKT-{number:03d}"


# ================================================================
# MODEL DECISION
# ================================================================

def get_model_decision(
    user_request,
    tokenizer,
    model
):

    # ------------------------------------------------------------
    # PASS 1 — ROUTING
    # ------------------------------------------------------------

    router_prompt = (
        create_router_prompt(
            user_request
        )
    )

    router_output = (
        generate_model_output(
            router_prompt,
            tokenizer,
            model,
            max_new_tokens=20
        )
    )

    router_upper = (
        router_output
        .strip()
        .upper()
    )


    # ------------------------------------------------------------
    # NO TOOL
    # ------------------------------------------------------------

    if "NO_TOOL" in router_upper:

        return {
            "tool_requested": False,
            "tool_name": None,
            "ticket_id": None,
            "router_output":
                router_output,
            "argument_output":
                "",
            "parse_error":
                None,
        }


    # ------------------------------------------------------------
    # TOOL REQUESTED
    # ------------------------------------------------------------

    if (
        "CALL" in router_upper
        or
        "GET_TICKET" in router_upper
    ):

        # --------------------------------------------------------
        # PASS 2 — ARGUMENT EXTRACTION
        # --------------------------------------------------------

        argument_prompt = (
            create_argument_prompt(
                user_request
            )
        )

        argument_output = (
            generate_model_output(
                argument_prompt,
                tokenizer,
                model,
                max_new_tokens=10
            )
        )

        ticket_id = (
            normalize_ticket_id(
                argument_output
            )
        )

        if ticket_id is None:

            return {
                "tool_requested":
                    True,

                "tool_name":
                    "get_ticket",

                "ticket_id":
                    None,

                "router_output":
                    router_output,

                "argument_output":
                    argument_output,

                "parse_error":
                    (
                        "The model selected get_ticket "
                        "but failed to extract a "
                        "valid ticket number."
                    ),
            }


        return {
            "tool_requested":
                True,

            "tool_name":
                "get_ticket",

            "ticket_id":
                ticket_id,

            "router_output":
                router_output,

            "argument_output":
                argument_output,

            "parse_error":
                None,
        }


    # ------------------------------------------------------------
    # INVALID ROUTER OUTPUT
    # ------------------------------------------------------------

    return {
        "tool_requested":
            False,

        "tool_name":
            None,

        "ticket_id":
            None,

        "router_output":
            router_output,

        "argument_output":
            "",

        "parse_error":
            (
                "Unrecognized router output: "
                f"{router_output}"
            ),
    }


# ================================================================
# EXECUTE TOOL
# ================================================================

def execute_decision(
    decision
):

    if not decision[
        "tool_requested"
    ]:

        return {
            "tool_result": None,
            "tool_error": None,
        }


    if decision[
        "tool_name"
    ] != "get_ticket":

        return {
            "tool_result":
                None,

            "tool_error":
                "Unsupported tool requested.",
        }


    if not decision[
        "ticket_id"
    ]:

        return {
            "tool_result":
                None,

            "tool_error":
                "Missing ticket_id.",
        }


    try:

        result = get_ticket(
            decision[
                "ticket_id"
            ]
        )

        return {
            "tool_result":
                result,

            "tool_error":
                None,
        }


    except Exception as error:

        return {
            "tool_result":
                None,

            "tool_error":
                (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
        }


# ================================================================
# FINAL ANSWER
# ================================================================

def build_final_answer(
    decision,
    execution
):

    if decision[
        "parse_error"
    ]:

        return (
            "The model did not produce "
            "a valid structured tool decision."
        )


    if not decision[
        "tool_requested"
    ]:

        return (
            "No ticket lookup was performed. "
            "The only available Stage 2 tool "
            "is get_ticket, so this request "
            "cannot be handled by the current tool."
        )


    if execution[
        "tool_error"
    ]:

        return (
            "The ticket lookup could not "
            "be completed: "
            + execution[
                "tool_error"
            ]
        )


    ticket = execution[
        "tool_result"
    ]


    return (
        f"{ticket['ticket_id']} is "
        f"{ticket['status']}. "
        f"Priority: {ticket['priority']}. "
        f"Owner: {ticket['owner']}. "
        f"Summary: {ticket['summary']}"
    )


# ================================================================
# LOAD TEST SET
# ================================================================

def load_test_set():

    with DATA_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        return list(
            csv.DictReader(
                file
            )
        )


# ================================================================
# MAIN EXPERIMENT
# ================================================================

def main():

    tests = load_test_set()

    print(
        f"\nStage 2 tests loaded: "
        f"{len(tests)}"
    )


    tokenizer, model = (
        load_model()
    )


    result_rows = []


    for test in tests:

        print(
            "\n"
            + "=" * 70
        )

        print(
            test[
                "request_id"
            ]
        )

        print(
            "User:",
            test[
                "user_request"
            ]
        )


        # --------------------------------------------------------
        # MODEL DECISION
        # --------------------------------------------------------

        decision = (
            get_model_decision(
                test[
                    "user_request"
                ],
                tokenizer,
                model
            )
        )


        print(
            "Router output:",
            decision[
                "router_output"
            ]
        )


        if decision[
            "tool_requested"
        ]:

            print(
                "Argument output:",
                decision[
                    "argument_output"
                ]
            )


        print(
            "Structured decision:",
            {
                "tool_requested":
                    decision[
                        "tool_requested"
                    ],

                "tool_name":
                    decision[
                        "tool_name"
                    ],

                "ticket_id":
                    decision[
                        "ticket_id"
                    ],

                "parse_error":
                    decision[
                        "parse_error"
                    ],
            }
        )


        # --------------------------------------------------------
        # PYTHON EXECUTION
        # --------------------------------------------------------

        execution = (
            execute_decision(
                decision
            )
        )


        print(
            "Tool result:",
            execution[
                "tool_result"
            ]
        )


        if execution[
            "tool_error"
        ]:

            print(
                "Tool error:",
                execution[
                    "tool_error"
                ]
            )


        # --------------------------------------------------------
        # FINAL ANSWER
        # --------------------------------------------------------

        final_answer = (
            build_final_answer(
                decision,
                execution
            )
        )


        print(
            "Final answer:",
            final_answer
        )


        # --------------------------------------------------------
        # EXPECTED BEHAVIOUR
        # --------------------------------------------------------

        expected_tool_requested = (
            test[
                "expected_tool_requested"
            ]
            == "1"
        )


        actual_tool_requested = (
            decision[
                "tool_requested"
            ]
        )


        selection_correct = int(
            expected_tool_requested
            ==
            actual_tool_requested
        )


        # --------------------------------------------------------
        # ARGUMENT EVALUATION
        # --------------------------------------------------------

        if expected_tool_requested:

            tool_name_correct = int(
                decision[
                    "tool_name"
                ]
                ==
                test[
                    "expected_tool_name"
                ]
            )

            argument_correct = int(
                decision[
                    "ticket_id"
                ]
                ==
                test[
                    "expected_ticket_id"
                ]
            )

        else:

            tool_name_correct = int(
                not actual_tool_requested
            )

            argument_correct = int(
                not actual_tool_requested
            )


        # --------------------------------------------------------
        # OVERALL PASS
        # --------------------------------------------------------

        overall_pass = int(

            selection_correct == 1

            and

            tool_name_correct == 1

            and

            argument_correct == 1

            and

            decision[
                "parse_error"
            ] is None

            and

            execution[
                "tool_error"
            ] is None
        )


        # --------------------------------------------------------
        # SAVE ROW
        # --------------------------------------------------------

        result_rows.append(
            {
                **test,

                "router_output":
                    decision[
                        "router_output"
                    ],

                "argument_output":
                    decision[
                        "argument_output"
                    ],

                "actual_tool_requested":
                    int(
                        actual_tool_requested
                    ),

                "actual_tool_name":
                    (
                        decision[
                            "tool_name"
                        ]
                        or ""
                    ),

                "actual_ticket_id":
                    (
                        decision[
                            "ticket_id"
                        ]
                        or ""
                    ),

                "parse_error":
                    (
                        decision[
                            "parse_error"
                        ]
                        or ""
                    ),

                "tool_result":
                    (
                        json.dumps(
                            execution[
                                "tool_result"
                            ],
                            ensure_ascii=False
                        )

                        if execution[
                            "tool_result"
                        ]

                        else ""
                    ),

                "tool_error":
                    (
                        execution[
                            "tool_error"
                        ]
                        or ""
                    ),

                "final_answer":
                    final_answer,

                "tool_selection_correct":
                    selection_correct,

                "tool_name_correct":
                    tool_name_correct,

                "argument_correct":
                    argument_correct,

                "overall_pass":
                    overall_pass,
            }
        )


    # ============================================================
    # SAVE DETAILED RESULTS
    # ============================================================

    with RESULTS_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=result_rows[
                0
            ].keys()
        )

        writer.writeheader()

        writer.writerows(
            result_rows
        )


    # ============================================================
    # METRICS
    # ============================================================

    total = len(
        result_rows
    )


    selection_accuracy = (
        sum(
            row[
                "tool_selection_correct"
            ]

            for row in result_rows
        )
        / total
    )


    ticket_rows = [
        row

        for row in result_rows

        if row[
            "expected_tool_requested"
        ] == "1"
    ]


    argument_accuracy = (
        sum(
            row[
                "argument_correct"
            ]

            for row in ticket_rows
        )
        / len(
            ticket_rows
        )
    )


    no_tool_rows = [
        row

        for row in result_rows

        if row[
            "expected_tool_requested"
        ] == "0"
    ]


    unnecessary_calls = sum(
        1

        for row in no_tool_rows

        if row[
            "actual_tool_requested"
        ] == 1
    )


    unnecessary_tool_rate = (
        unnecessary_calls
        / len(
            no_tool_rows
        )
    )


    overall_pass_rate = (
        sum(
            row[
                "overall_pass"
            ]

            for row in result_rows
        )
        / total
    )


    # ============================================================
    # SUMMARY
    # ============================================================

    summary = {
        "model":
            MODEL_NAME,

        "experiment":
            "V5_v3_router_plus_two_pass_argument_extraction",

        "total_requests":
            total,

        "ticket_requests":
            len(
                ticket_rows
            ),

        "no_tool_requests":
            len(
                no_tool_rows
            ),

        "tool_selection_accuracy":
            selection_accuracy,

        "argument_accuracy":
            argument_accuracy,

        "unnecessary_tool_call_rate":
            unnecessary_tool_rate,

        "overall_pass_rate":
            overall_pass_rate,
    }


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


    # ============================================================
    # DISPLAY SUMMARY
    # ============================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STAGE 2 — V5 RESULTS"
    )

    print(
        "=" * 70
    )


    print(
        f"Requests: {total}"
    )


    print(
        "Tool Selection Accuracy:",
        f"{selection_accuracy * 100:.2f}%"
    )


    print(
        "Argument Accuracy:",
        f"{argument_accuracy * 100:.2f}%"
    )


    print(
        "Unnecessary Tool Call Rate:",
        f"{unnecessary_tool_rate * 100:.2f}%"
    )


    print(
        "Overall Pass Rate:",
        f"{overall_pass_rate * 100:.2f}%"
    )


    print(
        "\nDetailed results:"
    )

    print(
        RESULTS_FILE
    )


    print(
        "\nSummary:"
    )

    print(
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()