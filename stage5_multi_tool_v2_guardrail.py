import csv
import json
import re

from pathlib import Path

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from tools import (
    get_ticket,
    get_expense_claim,
    calculate_expense,
    search_company_knowledge
)


# ================================================================
# STAGE 5 — V2 HYBRID MULTI-TOOL PLANNER
#
# V1 problem:
# The LLM sometimes produced malformed or incomplete multi-tool
# plans.
#
# V2 solution:
# 1. LLM proposes a plan.
# 2. Python inspects strong structural cues.
# 3. Python validates / overrides the proposed plan.
#
# IMPORTANT:
# V2 still DOES NOT perform true tool-to-tool chaining.
# It also still sends the whole user request to RAG.
# Those are later experiments.
# ================================================================


# ================================================================
# 1. PATHS
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

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

TEST_FILE = (
    DATA_DIR
    / "stage5_requests.csv"
)

DETAIL_FILE = (
    RESULTS_DIR
    / "stage5_results_v2.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "stage5_summary_v2.csv"
)


# ================================================================
# 2. MODEL SETTINGS
# ================================================================

MODEL_NAME = (
    "google/flan-t5-base"
)


# ================================================================
# 3. TOOL LABELS
# ================================================================

LABEL_TO_TOOL = {

    "TICKET":
        "get_ticket",

    "EXPENSE_CLAIM":
        "get_expense_claim",

    "CALCULATE":
        "calculate_expense",

    "KNOWLEDGE":
        "search_company_knowledge",

    "NO_TOOL":
        "NO_TOOL",
}


TOOL_TO_LABEL = {

    value: key

    for key, value
    in LABEL_TO_TOOL.items()
}


# ================================================================
# 4. PATTERNS
# ================================================================

TICKET_ID_PATTERN = re.compile(
    r"\bTKT-\d+\b",
    re.IGNORECASE
)

CLAIM_ID_PATTERN = re.compile(
    r"\bEXP-\d+\b",
    re.IGNORECASE
)

NUMBER_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\b"
)


# ================================================================
# 5. LOAD MODEL
# ================================================================

def load_planner_model():

    print(
        f"\nLoading Stage 5 planner model: "
        f"{MODEL_NAME}"
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        AutoModelForSeq2SeqLM
        .from_pretrained(
            MODEL_NAME
        )
    )

    model.eval()

    print(
        "Stage 5 planner model ready."
    )

    return (
        tokenizer,
        model
    )


# ================================================================
# 6. BUILD PLANNER PROMPT
# ================================================================

def build_planner_prompt(
    user_request
):

    return f"""
Choose ALL tools needed for the request.

Allowed labels:
TICKET
EXPENSE_CLAIM
CALCULATE
KNOWLEDGE
NO_TOOL

Meanings:
TICKET = read a support ticket.
EXPENSE_CLAIM = read an expense claim.
CALCULATE = perform arithmetic on expense amounts.
KNOWLEDGE = search company policy or process.
NO_TOOL = none of these tools are needed.

A request may require more than one tool.

Return only the labels in execution order.
Separate multiple labels using |.

Examples:

Request:
Check TKT-005 and tell me the escalation policy.
Answer:
TICKET|KNOWLEDGE

Request:
Check EXP-010 and tell me the travel expense policy.
Answer:
EXPENSE_CLAIM|KNOWLEDGE

Request:
Add 5000 and 9000 and tell me the expense policy.
Answer:
CALCULATE|KNOWLEDGE

Request:
Check TKT-010 and EXP-002.
Answer:
TICKET|EXPENSE_CLAIM

Request:
Get the amount on EXP-010 and add 7200 to it.
Answer:
EXPENSE_CLAIM|CALCULATE

Request:
{user_request}

Answer:
""".strip()


# ================================================================
# 7. PLAN TOOLS WITH LLM
# ================================================================

def plan_tools(
    user_request,
    tokenizer,
    model
):

    prompt = (
        build_planner_prompt(
            user_request
        )
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    print(
        "[DEBUG] Planner prompt tokens:",
        inputs[
            "input_ids"
        ].shape[1]
    )

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            num_beams=4
        )

    raw_output = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return raw_output


# ================================================================
# 8. PARSE MODEL PLAN
# ================================================================

def parse_tool_plan(
    raw_output
):

    text = (
        str(raw_output)
        .upper()
    )

    pattern = re.compile(
        r"EXPENSE_CLAIM|"
        r"CALCULATE|"
        r"KNOWLEDGE|"
        r"TICKET|"
        r"NO_TOOL"
    )

    matches = pattern.findall(
        text
    )

    ordered_labels = []

    for label in matches:

        if label not in ordered_labels:

            ordered_labels.append(
                label
            )


    real_labels = [
        label

        for label
        in ordered_labels

        if label != "NO_TOOL"
    ]


    if real_labels:

        ordered_labels = (
            real_labels
        )


    planned_tools = [

        LABEL_TO_TOOL[
            label
        ]

        for label
        in ordered_labels

        if label in LABEL_TO_TOOL
    ]


    return (
        ordered_labels,
        planned_tools
    )


# ================================================================
# 9. STRUCTURAL CUE HELPERS
# ================================================================

def find_ticket_position(
    user_request
):

    exact_match = (
        TICKET_ID_PATTERN.search(
            user_request
        )
    )

    if exact_match:

        return exact_match.start()


    patterns = [

        r"\bticket\b",

        r"\bsupport\s+case\b",

        r"\bcase\s+(?:number\s+)?#?\s*\d+\b",
    ]


    positions = []


    for pattern in patterns:

        match = re.search(
            pattern,
            user_request,
            re.IGNORECASE
        )

        if match:

            positions.append(
                match.start()
            )


    if positions:

        return min(
            positions
        )


    return None


def find_claim_position(
    user_request
):

    exact_match = (
        CLAIM_ID_PATTERN.search(
            user_request
        )
    )

    if exact_match:

        return exact_match.start()


    patterns = [

        r"\bexpense\s+claim\b",

        r"\bclaim\s+(?:number\s+)?#?\s*\d+\b",
    ]


    positions = []


    for pattern in patterns:

        match = re.search(
            pattern,
            user_request,
            re.IGNORECASE
        )

        if match:

            positions.append(
                match.start()
            )


    if positions:

        return min(
            positions
        )


    return None


def find_calculation_position(
    user_request
):

    lower_request = (
        user_request.lower()
    )


    arithmetic_patterns = [

        r"\badd\b",

        r"\bcalculate\b",

        r"\btotal\b",

        r"\bsum\b",

        r"\bplus\b",
    ]


    positions = []


    for pattern in arithmetic_patterns:

        match = re.search(
            pattern,
            lower_request
        )

        if match:

            positions.append(
                match.start()
            )


    if not positions:

        return None


    # ------------------------------------------------------------
    # Remove structured IDs before counting numbers.
    #
    # Otherwise:
    # EXP-010 could look like the number 10.
    # TKT-005 could look like the number 5.
    # ------------------------------------------------------------

    cleaned_request = (
        TICKET_ID_PATTERN.sub(
            "",
            user_request
        )
    )

    cleaned_request = (
        CLAIM_ID_PATTERN.sub(
            "",
            cleaned_request
        )
    )


    numbers = NUMBER_PATTERN.findall(
        cleaned_request
    )


    has_claim_reference = (
        find_claim_position(
            user_request
        )
        is not None
    )


    # ------------------------------------------------------------
    # Normal arithmetic:
    # At least two explicit values.
    #
    # Dependent arithmetic:
    # One explicit value + an expense claim whose amount
    # must later be retrieved.
    # ------------------------------------------------------------

    if len(numbers) >= 2:

        return min(
            positions
        )


    if (
        has_claim_reference
        and
        len(numbers) >= 1
    ):

        return min(
            positions
        )


    return None


def find_knowledge_position(
    user_request
):

    lower_request = (
        user_request.lower()
    )


    knowledge_phrases = [

        "policy",

        "process",

        "procedure",

        "escalation",

        "response target",

        "service level",

        "annual leave",

        "sick leave",

        "remote work",

        "company says",

        "company say",

        "what does the company",
    ]


    positions = []


    for phrase in knowledge_phrases:

        position = lower_request.find(
            phrase
        )

        if position >= 0:

            positions.append(
                position
            )


    if positions:

        return min(
            positions
        )


    return None


# ================================================================
# 10. BUILD STRUCTURAL PLAN
# ================================================================

def build_structural_plan(
    user_request
):

    detected = []


    ticket_position = (
        find_ticket_position(
            user_request
        )
    )


    if ticket_position is not None:

        detected.append(
            (
                ticket_position,
                "get_ticket"
            )
        )


    claim_position = (
        find_claim_position(
            user_request
        )
    )


    if claim_position is not None:

        detected.append(
            (
                claim_position,
                "get_expense_claim"
            )
        )


    calculation_position = (
        find_calculation_position(
            user_request
        )
    )


    if calculation_position is not None:

        detected.append(
            (
                calculation_position,
                "calculate_expense"
            )
        )


    knowledge_position = (
        find_knowledge_position(
            user_request
        )
    )


    if knowledge_position is not None:

        detected.append(
            (
                knowledge_position,
                "search_company_knowledge"
            )
        )


    # ------------------------------------------------------------
    # Sort tools according to where their cue first appears in
    # the user's request.
    # ------------------------------------------------------------

    detected.sort(
        key=lambda item: item[0]
    )


    structural_plan = []


    for _, tool_name in detected:

        if tool_name not in structural_plan:

            structural_plan.append(
                tool_name
            )


    return structural_plan


# ================================================================
# 11. VALIDATE MODEL PLAN
# ================================================================

def validate_tool_plan(
    user_request,
    model_tools
):

    structural_plan = (
        build_structural_plan(
            user_request
        )
    )


    # ------------------------------------------------------------
    # If strong structural evidence exists, trust the explicit
    # structure over malformed/incomplete model output.
    # ------------------------------------------------------------

    if structural_plan:

        if model_tools == structural_plan:

            reason = (
                "MODEL_PLAN_ACCEPTED"
            )

        else:

            reason = (
                "STRUCTURAL_OVERRIDE"
            )


        return (
            structural_plan,
            structural_plan,
            reason
        )


    # ------------------------------------------------------------
    # No strong structural plan was detected.
    #
    # Fall back to the model's valid parsed plan.
    # ------------------------------------------------------------

    if model_tools:

        return (
            model_tools,
            structural_plan,
            "MODEL_FALLBACK"
        )


    return (
        [],
        structural_plan,
        "NO_VALID_PLAN"
    )


# ================================================================
# 12. EXTRACT TICKET ARGUMENT
# ================================================================

def extract_ticket_argument(
    user_request
):

    exact_match = (
        TICKET_ID_PATTERN.search(
            user_request
        )
    )

    if exact_match:

        return (
            exact_match
            .group(0)
            .upper()
        )


    natural_patterns = [

        r"\bticket\s+(?:number\s+)?#?\s*(\d+)\b",

        r"\bsupport\s+case\s+(?:number\s+)?#?\s*(\d+)\b",

        r"\bcase\s+(?:number\s+)?#?\s*(\d+)\b",
    ]


    for pattern in natural_patterns:

        match = re.search(
            pattern,
            user_request,
            re.IGNORECASE
        )

        if match:

            number = int(
                match.group(1)
            )

            return (
                f"TKT-{number:03d}"
            )


    return None


# ================================================================
# 13. EXTRACT CLAIM ARGUMENT
# ================================================================

def extract_claim_argument(
    user_request
):

    exact_match = (
        CLAIM_ID_PATTERN.search(
            user_request
        )
    )

    if exact_match:

        return (
            exact_match
            .group(0)
            .upper()
        )


    natural_patterns = [

        r"\bexpense\s+claim\s+(?:number\s+)?#?\s*(\d+)\b",

        r"\bclaim\s+(?:number\s+)?#?\s*(\d+)\b",
    ]


    for pattern in natural_patterns:

        match = re.search(
            pattern,
            user_request,
            re.IGNORECASE
        )

        if match:

            number = int(
                match.group(1)
            )

            return (
                f"EXP-{number:03d}"
            )


    return None


# ================================================================
# 14. EXTRACT CALCULATION ARGUMENTS
# ================================================================

def extract_calculation_arguments(
    user_request
):

    cleaned_request = (
        TICKET_ID_PATTERN.sub(
            "",
            user_request
        )
    )

    cleaned_request = (
        CLAIM_ID_PATTERN.sub(
            "",
            cleaned_request
        )
    )


    numbers = NUMBER_PATTERN.findall(
        cleaned_request
    )


    amounts = []


    for number in numbers:

        try:

            amounts.append(
                float(number)
            )

        except ValueError:

            continue


    return amounts


# ================================================================
# 15. EXTRACT TOOL ARGUMENT
# ================================================================

def extract_argument(
    tool_name,
    user_request
):

    if tool_name == "get_ticket":

        return (
            extract_ticket_argument(
                user_request
            )
        )


    if tool_name == "get_expense_claim":

        return (
            extract_claim_argument(
                user_request
            )
        )


    if tool_name == "calculate_expense":

        return (
            extract_calculation_arguments(
                user_request
            )
        )


    if tool_name == "search_company_knowledge":

        # --------------------------------------------------------
        # V2 intentionally still sends the whole request to RAG.
        #
        # We will isolate the knowledge sub-question in a later
        # controlled experiment.
        # --------------------------------------------------------

        return user_request


    return None


# ================================================================
# 16. EXECUTE ONE TOOL
# ================================================================

def execute_tool(
    tool_name,
    argument
):

    if tool_name == "get_ticket":

        if not argument:

            raise ValueError(
                "No ticket ID was extracted."
            )

        return get_ticket(
            argument
        )


    if tool_name == "get_expense_claim":

        if not argument:

            raise ValueError(
                "No expense claim ID was extracted."
            )

        return get_expense_claim(
            argument
        )


    if tool_name == "calculate_expense":

        if not argument:

            raise ValueError(
                "No expense amounts were extracted."
            )

        return calculate_expense(
            argument
        )


    if tool_name == "search_company_knowledge":

        if not argument:

            raise ValueError(
                "No knowledge question was extracted."
            )

        return search_company_knowledge(
            argument
        )


    raise ValueError(
        f"Unknown tool: {tool_name}"
    )


# ================================================================
# 17. FORMAT TOOL RESULT
# ================================================================

def format_tool_result(
    tool_name,
    result
):

    if tool_name == "get_ticket":

        return (
            f"Ticket {result['ticket_id']} is "
            f"{result['status']}. "
            f"Priority: {result['priority']}. "
            f"Owner: {result['owner']}. "
            f"Summary: {result['summary']}"
        )


    if tool_name == "get_expense_claim":

        return (
            f"Expense claim {result['claim_id']} "
            f"was submitted by {result['employee']}. "
            f"Amount: {result['amount']}. "
            f"Status: {result['status']}."
        )


    if tool_name == "calculate_expense":

        return (
            f"Calculated expense total: "
            f"{result}"
        )


    if tool_name == "search_company_knowledge":

        return (
            f"Company knowledge: "
            f"{result.get('answer', '')}"
        )


    return str(
        result
    )


# ================================================================
# 18. BUILD FINAL ANSWER
# ================================================================

def build_final_answer(
    successful_results,
    errors
):

    answer_parts = []


    for item in successful_results:

        answer_parts.append(
            format_tool_result(
                item[
                    "tool"
                ],
                item[
                    "result"
                ]
            )
        )


    for item in errors:

        answer_parts.append(
            (
                f"{item['tool']} failed: "
                f"{item['error']}"
            )
        )


    if not answer_parts:

        return (
            "The request could not be completed "
            "with the planned tools."
        )


    return (
        " ".join(
            answer_parts
        )
    )


# ================================================================
# 19. LOAD TESTS
# ================================================================

def load_tests():

    if not TEST_FILE.exists():

        raise FileNotFoundError(
            f"Stage 5 test file not found: "
            f"{TEST_FILE}"
        )


    with TEST_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(
            file
        )

        return list(
            reader
        )


# ================================================================
# 20. EXPECTED TOOL PLAN
# ================================================================

def parse_expected_tools(
    expected_tools
):

    return [

        item.strip()

        for item
        in expected_tools.split("|")

        if item.strip()
    ]


# ================================================================
# 21. NORMALIZE ARGUMENT
# ================================================================

def normalize_argument_for_comparison(
    value
):

    if value is None:

        return ""


    if isinstance(
        value,
        list
    ):

        normalized = []


        for item in value:

            try:

                numeric = float(
                    item
                )


                if numeric.is_integer():

                    normalized.append(
                        str(
                            int(numeric)
                        )
                    )

                else:

                    normalized.append(
                        str(numeric)
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
        )


    return (
        str(value)
        .strip()
        .upper()
    )


# ================================================================
# 22. EVALUATE ONE REQUEST
# ================================================================

def evaluate_request(
    test,
    planned_tools,
    executed_calls
):

    expected_tools = (
        parse_expected_tools(
            test[
                "expected_tools"
            ]
        )
    )


    exact_plan_correct = (
        planned_tools
        ==
        expected_tools
    )


    required_found = sum(
        1

        for tool
        in expected_tools

        if tool in planned_tools
    )


    required_total = len(
        expected_tools
    )


    unnecessary_tools = [

        tool

        for tool
        in planned_tools

        if tool not in expected_tools
    ]


    has_unnecessary_tool = bool(
        unnecessary_tools
    )


    expected_first_argument = (
        test[
            "expected_first_argument"
        ]
        .strip()
    )


    actual_first_argument = None


    if executed_calls:

        actual_first_argument = (
            executed_calls[0]
            .get(
                "argument"
            )
        )


    actual_first_normalized = (
        normalize_argument_for_comparison(
            actual_first_argument
        )
    )


    expected_first_normalized = (
        normalize_argument_for_comparison(
            expected_first_argument
        )
    )


    first_argument_correct = (
        actual_first_normalized
        ==
        expected_first_normalized
    )


    tool_error_count = sum(
        1

        for call
        in executed_calls

        if call.get(
            "error"
        )
    )


    no_tool_errors = (
        tool_error_count
        ==
        0
    )


    # ------------------------------------------------------------
    # V2 still does not implement true dependency chaining.
    # ------------------------------------------------------------

    dependency_type = (
        test[
            "dependency"
        ]
        .strip()
        .lower()
    )


    if dependency_type == "dependent":

        dependency_satisfied = False

    else:

        dependency_satisfied = True


    task_complete = (
        exact_plan_correct
        and
        first_argument_correct
        and
        no_tool_errors
        and
        dependency_satisfied
    )


    return {

        "expected_tools":
            expected_tools,

        "exact_plan_correct":
            exact_plan_correct,

        "required_tools_found":
            required_found,

        "required_tools_total":
            required_total,

        "has_unnecessary_tool":
            has_unnecessary_tool,

        "unnecessary_tools":
            unnecessary_tools,

        "first_argument_correct":
            first_argument_correct,

        "dependency_satisfied":
            dependency_satisfied,

        "tool_error_count":
            tool_error_count,

        "task_complete":
            task_complete,
    }


# ================================================================
# 23. RUN ONE TEST
# ================================================================

def run_test(
    test,
    tokenizer,
    model
):

    request_id = (
        test[
            "request_id"
        ]
    )

    user_request = (
        test[
            "user_request"
        ]
    )


    print(
        "\n"
        + "=" * 76
    )

    print(
        request_id,
        "-",
        test[
            "category"
        ]
    )

    print(
        "User:",
        user_request
    )


    # ------------------------------------------------------------
    # A. MODEL PLAN
    # ------------------------------------------------------------

    raw_plan = plan_tools(
        user_request,
        tokenizer,
        model
    )


    model_labels, model_tools = (
        parse_tool_plan(
            raw_plan
        )
    )


    print(
        "Raw planner output:",
        raw_plan
    )

    print(
        "Model parsed labels:",
        model_labels
    )

    print(
        "Model planned tools:",
        model_tools
    )


    # ------------------------------------------------------------
    # B. PYTHON VALIDATION
    # ------------------------------------------------------------

    (
        validated_tools,
        structural_plan,
        validation_reason
    ) = validate_tool_plan(
        user_request,
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
    # C. EXECUTE VALIDATED PLAN
    # ------------------------------------------------------------

    executed_calls = []

    successful_results = []

    errors = []


    for tool_name in validated_tools:

        argument = extract_argument(
            tool_name,
            user_request
        )


        print(
            f"\nTool: {tool_name}"
        )

        print(
            "Argument:",
            argument
        )


        call_record = {

            "tool":
                tool_name,

            "argument":
                argument,

            "result":
                None,

            "error":
                None,
        }


        try:

            result = execute_tool(
                tool_name,
                argument
            )


            call_record[
                "result"
            ] = result


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


            call_record[
                "error"
            ] = error_text


            errors.append(
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


        executed_calls.append(
            call_record
        )


    # ------------------------------------------------------------
    # D. FINAL ANSWER
    # ------------------------------------------------------------

    final_answer = build_final_answer(
        successful_results,
        errors
    )


    print(
        "\nFinal answer:",
        final_answer
    )


    # ------------------------------------------------------------
    # E. EVALUATION
    # ------------------------------------------------------------

    evaluation = evaluate_request(
        test,
        validated_tools,
        executed_calls
    )


    print(
        "\nEvaluation:"
    )

    print(
        "Exact plan correct:",
        evaluation[
            "exact_plan_correct"
        ]
    )

    print(
        "First argument correct:",
        evaluation[
            "first_argument_correct"
        ]
    )

    print(
        "Dependency satisfied:",
        evaluation[
            "dependency_satisfied"
        ]
    )

    print(
        "Task complete:",
        evaluation[
            "task_complete"
        ]
    )


    return {

        "request_id":
            request_id,

        "category":
            test[
                "category"
            ],

        "user_request":
            user_request,

        "dependency":
            test[
                "dependency"
            ],

        "expected_tools":
            "|".join(
                evaluation[
                    "expected_tools"
                ]
            ),

        "raw_planner_output":
            raw_plan,

        "model_planned_tools":
            "|".join(
                model_tools
            ),

        "structural_plan":
            "|".join(
                structural_plan
            ),

        "validated_tools":
            "|".join(
                validated_tools
            ),

        "validation_reason":
            validation_reason,

        "executed_calls":
            json.dumps(
                executed_calls,
                ensure_ascii=False
            ),

        "final_answer":
            final_answer,

        "exact_plan_correct":
            evaluation[
                "exact_plan_correct"
            ],

        "required_tools_found":
            evaluation[
                "required_tools_found"
            ],

        "required_tools_total":
            evaluation[
                "required_tools_total"
            ],

        "has_unnecessary_tool":
            evaluation[
                "has_unnecessary_tool"
            ],

        "unnecessary_tools":
            "|".join(
                evaluation[
                    "unnecessary_tools"
                ]
            ),

        "first_argument_correct":
            evaluation[
                "first_argument_correct"
            ],

        "dependency_satisfied":
            evaluation[
                "dependency_satisfied"
            ],

        "tool_error_count":
            evaluation[
                "tool_error_count"
            ],

        "task_complete":
            evaluation[
                "task_complete"
            ],
    }


# ================================================================
# 24. SAVE DETAILED RESULTS
# ================================================================

def save_detailed_results(
    rows
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    if not rows:

        return


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


# ================================================================
# 25. BUILD SUMMARY
# ================================================================

def build_summary(
    rows
):

    total_requests = len(
        rows
    )


    exact_plan_correct = sum(
        1

        for row in rows

        if row[
            "exact_plan_correct"
        ]
    )


    first_argument_correct = sum(
        1

        for row in rows

        if row[
            "first_argument_correct"
        ]
    )


    unnecessary_requests = sum(
        1

        for row in rows

        if row[
            "has_unnecessary_tool"
        ]
    )


    task_complete = sum(
        1

        for row in rows

        if row[
            "task_complete"
        ]
    )


    required_tools_found = sum(
        row[
            "required_tools_found"
        ]

        for row in rows
    )


    required_tools_total = sum(
        row[
            "required_tools_total"
        ]

        for row in rows
    )


    dependent_rows = [

        row

        for row in rows

        if (
            row[
                "dependency"
            ]
            .strip()
            .lower()
            ==
            "dependent"
        )
    ]


    dependent_success = sum(
        1

        for row in dependent_rows

        if row[
            "dependency_satisfied"
        ]
    )


    def percentage(
        numerator,
        denominator
    ):

        if denominator == 0:

            return 0.0


        return round(
            (
                numerator
                /
                denominator
            )
            *
            100,
            2
        )


    return {

        "experiment":
            (
                "Stage5_V2_"
                "hybrid_multi_tool_guardrail"
            ),

        "requests":
            total_requests,

        "exact_plan_accuracy":
            percentage(
                exact_plan_correct,
                total_requests
            ),

        "required_tool_coverage":
            percentage(
                required_tools_found,
                required_tools_total
            ),

        "first_argument_accuracy":
            percentage(
                first_argument_correct,
                total_requests
            ),

        "unnecessary_tool_call_rate":
            percentage(
                unnecessary_requests,
                total_requests
            ),

        "dependency_success_rate":
            percentage(
                dependent_success,
                len(
                    dependent_rows
                )
            ),

        "structural_task_completion":
            percentage(
                task_complete,
                total_requests
            ),
    }


# ================================================================
# 26. SAVE SUMMARY
# ================================================================

def save_summary(
    summary
):

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
# 27. MAIN
# ================================================================

def main():

    tests = load_tests()


    print(
        "\nSTAGE 5 — V2 HYBRID MULTI-TOOL PLANNER"
    )

    print(
        "=" * 76
    )

    print(
        "Tests loaded:",
        len(
            tests
        )
    )


    tokenizer, model = (
        load_planner_model()
    )


    rows = []


    for test in tests:

        row = run_test(
            test,
            tokenizer,
            model
        )

        rows.append(
            row
        )


    save_detailed_results(
        rows
    )


    summary = build_summary(
        rows
    )


    save_summary(
        summary
    )


    print(
        "\n"
        + "=" * 76
    )

    print(
        "STAGE 5 — V2 RESULTS"
    )

    print(
        "=" * 76
    )


    print(
        "Requests:",
        summary[
            "requests"
        ]
    )

    print(
        "Exact Plan Accuracy:",
        f"{summary['exact_plan_accuracy']:.2f}%"
    )

    print(
        "Required Tool Coverage:",
        f"{summary['required_tool_coverage']:.2f}%"
    )

    print(
        "First Argument Accuracy:",
        f"{summary['first_argument_accuracy']:.2f}%"
    )

    print(
        "Unnecessary Tool Call Rate:",
        f"{summary['unnecessary_tool_call_rate']:.2f}%"
    )

    print(
        "Dependency Success Rate:",
        f"{summary['dependency_success_rate']:.2f}%"
    )

    print(
        "Structural Task Completion:",
        f"{summary['structural_task_completion']:.2f}%"
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