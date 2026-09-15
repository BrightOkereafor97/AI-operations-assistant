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
# STAGE 5 — V3 KNOWLEDGE SUB-QUESTION EXTRACTION
#
# V2 solved:
# - multi-tool planning
# - malformed model plans
# - missing required tools
#
# V3 changes ONE main behaviour:
#
# Instead of sending the whole compound request to RAG,
# extract only the company-knowledge clause.
#
# V3 still DOES NOT implement true dependency chaining.
# ================================================================


# ================================================================
# 1. PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

RESULTS_DIR = PROJECT_ROOT / "results"

TEST_FILE = DATA_DIR / "stage5_requests.csv"

DETAIL_FILE = (
    RESULTS_DIR
    / "stage5_results_v3.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "stage5_summary_v3.csv"
)


# ================================================================
# 2. MODEL SETTINGS
# ================================================================

MODEL_NAME = "google/flan-t5-base"


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
# 6. PLANNER PROMPT
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
CALCULATE = perform arithmetic.
KNOWLEDGE = search company policy or process.
NO_TOOL = none of these tools are needed.

A request may require more than one tool.

Return only labels in execution order.
Separate multiple labels with |.

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
# 7. MODEL PLAN
# ================================================================

def plan_tools(
    user_request,
    tokenizer,
    model
):

    prompt = build_planner_prompt(
        user_request
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    print(
        "[DEBUG] Planner prompt tokens:",
        inputs["input_ids"].shape[1]
    )

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            num_beams=4
        )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )


# ================================================================
# 8. PARSE MODEL PLAN
# ================================================================

def parse_tool_plan(
    raw_output
):

    text = str(
        raw_output
    ).upper()

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

    labels = []

    for label in matches:

        if label not in labels:

            labels.append(
                label
            )

    real_labels = [
        label
        for label in labels
        if label != "NO_TOOL"
    ]

    if real_labels:

        labels = real_labels

    planned_tools = [
        LABEL_TO_TOOL[label]
        for label in labels
        if label in LABEL_TO_TOOL
    ]

    return (
        labels,
        planned_tools
    )


# ================================================================
# 9. STRUCTURAL CUE DETECTION
# ================================================================

def find_ticket_position(
    user_request
):

    match = TICKET_ID_PATTERN.search(
        user_request
    )

    if match:

        return match.start()

    match = re.search(
        r"\bticket\b|\bsupport\s+case\b",
        user_request,
        re.IGNORECASE
    )

    if match:

        return match.start()

    return None


def find_claim_position(
    user_request
):

    match = CLAIM_ID_PATTERN.search(
        user_request
    )

    if match:

        return match.start()

    match = re.search(
        r"\bexpense\s+claim\b|"
        r"\bclaim\s+(?:number\s+)?#?\s*\d+\b",
        user_request,
        re.IGNORECASE
    )

    if match:

        return match.start()

    return None


def find_calculation_position(
    user_request
):

    arithmetic_match = re.search(
        r"\badd\b|"
        r"\bcalculate\b|"
        r"\btotal\b|"
        r"\bsum\b|"
        r"\bplus\b",
        user_request,
        re.IGNORECASE
    )

    if not arithmetic_match:

        return None

    cleaned = TICKET_ID_PATTERN.sub(
        "",
        user_request
    )

    cleaned = CLAIM_ID_PATTERN.sub(
        "",
        cleaned
    )

    numbers = NUMBER_PATTERN.findall(
        cleaned
    )

    claim_exists = (
        find_claim_position(
            user_request
        )
        is not None
    )

    if len(numbers) >= 2:

        return arithmetic_match.start()

    if (
        claim_exists
        and len(numbers) >= 1
    ):

        return arithmetic_match.start()

    return None


def find_knowledge_position(
    user_request
):

    lower_request = user_request.lower()

    phrases = [

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

    for phrase in phrases:

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
# 10. STRUCTURAL PLAN
# ================================================================

def build_structural_plan(
    user_request
):

    detected = []

    ticket_position = find_ticket_position(
        user_request
    )

    if ticket_position is not None:

        detected.append(
            (
                ticket_position,
                "get_ticket"
            )
        )

    claim_position = find_claim_position(
        user_request
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
# 11. VALIDATE PLAN
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
# 12. TICKET ARGUMENT
# ================================================================

def extract_ticket_argument(
    user_request
):

    match = TICKET_ID_PATTERN.search(
        user_request
    )

    if match:

        return (
            match.group(0)
            .upper()
        )

    patterns = [

        r"\bticket\s+(?:number\s+)?#?\s*(\d+)\b",

        r"\bsupport\s+case\s+(?:number\s+)?#?\s*(\d+)\b",

        r"\bcase\s+(?:number\s+)?#?\s*(\d+)\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            user_request,
            re.IGNORECASE
        )

        if match:

            return (
                f"TKT-{int(match.group(1)):03d}"
            )

    return None


# ================================================================
# 13. CLAIM ARGUMENT
# ================================================================

def extract_claim_argument(
    user_request
):

    match = CLAIM_ID_PATTERN.search(
        user_request
    )

    if match:

        return (
            match.group(0)
            .upper()
        )

    patterns = [

        r"\bexpense\s+claim\s+(?:number\s+)?#?\s*(\d+)\b",

        r"\bclaim\s+(?:number\s+)?#?\s*(\d+)\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            user_request,
            re.IGNORECASE
        )

        if match:

            return (
                f"EXP-{int(match.group(1)):03d}"
            )

    return None


# ================================================================
# 14. CALCULATION ARGUMENTS
# ================================================================

def extract_calculation_arguments(
    user_request
):

    cleaned = TICKET_ID_PATTERN.sub(
        "",
        user_request
    )

    cleaned = CLAIM_ID_PATTERN.sub(
        "",
        cleaned
    )

    numbers = NUMBER_PATTERN.findall(
        cleaned
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
# 15. STAGE 5 V3 — KNOWLEDGE SUB-QUESTION EXTRACTION
# ================================================================

def contains_knowledge_cue(
    text
):

    lower_text = text.lower()

    phrases = [

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

        "company policy",

        "employee expense",
    ]

    return any(
        phrase in lower_text
        for phrase in phrases
    )


def clean_knowledge_clause(
    clause
):

    clause = clause.strip(
        " ,."
    )

    clause = re.sub(
        r"^(?:then\s+)?tell\s+me\s+",
        "",
        clause,
        flags=re.IGNORECASE
    )

    clause = re.sub(
        r"^(?:and\s+)",
        "",
        clause,
        flags=re.IGNORECASE
    )

    clause = clause.strip()

    if not clause:

        return clause

    # ------------------------------------------------------------
    # Improve a few natural constructions without changing
    # their meaning.
    # ------------------------------------------------------------

    if re.match(
        r"^what\s+the\s+company\b",
        clause,
        re.IGNORECASE
    ):

        clause = re.sub(
            r"^what\s+the\s+company\s+",
            "What does the company ",
            clause,
            flags=re.IGNORECASE
        )

        clause = re.sub(
            r"\bsays$",
            "say",
            clause,
            flags=re.IGNORECASE
        )


    elif re.match(
        r"^what\s+our\b",
        clause,
        re.IGNORECASE
    ):

        clause = re.sub(
            r"^what\s+our\s+",
            "What does our ",
            clause,
            flags=re.IGNORECASE
        )

        clause = re.sub(
            r"\bsays$",
            "say",
            clause,
            flags=re.IGNORECASE
        )


    if not clause.endswith(
        "?"
    ):

        clause += "?"

    return clause


def extract_knowledge_question(
    user_request
):

    # ------------------------------------------------------------
    # Split compound requests into smaller clauses.
    #
    # Examples:
    #
    # "Check TKT-005 AND tell me the policy"
    #
    # becomes:
    #
    # "Check TKT-005"
    # "tell me the policy"
    # ------------------------------------------------------------

    clauses = re.split(
        r"\s*,?\s+(?:and|then)\s+",
        user_request,
        flags=re.IGNORECASE
    )

    knowledge_clauses = []

    for clause in clauses:

        if contains_knowledge_cue(
            clause
        ):

            cleaned = clean_knowledge_clause(
                clause
            )

            if cleaned:

                knowledge_clauses.append(
                    cleaned
                )

    # ------------------------------------------------------------
    # Normally there is one knowledge clause.
    # If several exist, preserve them together.
    # ------------------------------------------------------------

    if knowledge_clauses:

        return " ".join(
            knowledge_clauses
        )


    # ------------------------------------------------------------
    # Fallback:
    # If no clean clause was detected, preserve the original
    # request rather than inventing a question.
    # ------------------------------------------------------------

    return user_request


# ================================================================
# 16. EXTRACT ARGUMENT
# ================================================================

def extract_argument(
    tool_name,
    user_request
):

    if tool_name == "get_ticket":

        return extract_ticket_argument(
            user_request
        )

    if tool_name == "get_expense_claim":

        return extract_claim_argument(
            user_request
        )

    if tool_name == "calculate_expense":

        return extract_calculation_arguments(
            user_request
        )

    if tool_name == "search_company_knowledge":

        return extract_knowledge_question(
            user_request
        )

    return None


# ================================================================
# 17. EXECUTE TOOL
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
# 18. FORMAT RESULT
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
# 19. FINAL ANSWER
# ================================================================

def build_final_answer(
    successful_results,
    errors
):

    parts = []

    for item in successful_results:

        parts.append(
            format_tool_result(
                item["tool"],
                item["result"]
            )
        )

    for item in errors:

        parts.append(
            (
                f"{item['tool']} failed: "
                f"{item['error']}"
            )
        )

    if not parts:

        return (
            "The request could not be completed "
            "with the planned tools."
        )

    return " ".join(
        parts
    )


# ================================================================
# 20. LOAD TESTS
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

        return list(
            csv.DictReader(
                file
            )
        )


# ================================================================
# 21. EXPECTED TOOLS
# ================================================================

def parse_expected_tools(
    value
):

    return [
        item.strip()
        for item in value.split("|")
        if item.strip()
    ]


# ================================================================
# 22. NORMALIZE ARGUMENT
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

                number = float(
                    item
                )

                if number.is_integer():

                    normalized.append(
                        str(
                            int(number)
                        )
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
        )

    return (
        str(value)
        .strip()
        .upper()
    )


# ================================================================
# 23. EVALUATE
# ================================================================

def evaluate_request(
    test,
    planned_tools,
    executed_calls
):

    expected_tools = (
        parse_expected_tools(
            test["expected_tools"]
        )
    )

    exact_plan_correct = (
        planned_tools
        ==
        expected_tools
    )

    required_found = sum(
        1
        for tool in expected_tools
        if tool in planned_tools
    )

    unnecessary_tools = [
        tool
        for tool in planned_tools
        if tool not in expected_tools
    ]

    expected_first_argument = (
        test[
            "expected_first_argument"
        ].strip()
    )

    actual_first_argument = None

    if executed_calls:

        actual_first_argument = (
            executed_calls[0]
            .get(
                "argument"
            )
        )

    first_argument_correct = (
        normalize_argument_for_comparison(
            actual_first_argument
        )
        ==
        normalize_argument_for_comparison(
            expected_first_argument
        )
    )

    tool_error_count = sum(
        1
        for call in executed_calls
        if call.get("error")
    )

    dependency_type = (
        test["dependency"]
        .strip()
        .lower()
    )

    # ------------------------------------------------------------
    # V3 intentionally still has no dependency chaining.
    # ------------------------------------------------------------

    dependency_satisfied = (
        dependency_type
        != "dependent"
    )

    task_complete = (
        exact_plan_correct
        and first_argument_correct
        and tool_error_count == 0
        and dependency_satisfied
    )

    return {

        "expected_tools":
            expected_tools,

        "exact_plan_correct":
            exact_plan_correct,

        "required_tools_found":
            required_found,

        "required_tools_total":
            len(expected_tools),

        "has_unnecessary_tool":
            bool(
                unnecessary_tools
            ),

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
# 24. RUN TEST
# ================================================================

def run_test(
    test,
    tokenizer,
    model
):

    request_id = (
        test["request_id"]
    )

    user_request = (
        test["user_request"]
    )

    print(
        "\n"
        + "=" * 76
    )

    print(
        request_id,
        "-",
        test["category"]
    )

    print(
        "User:",
        user_request
    )


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


    final_answer = build_final_answer(
        successful_results,
        errors
    )

    print(
        "\nFinal answer:",
        final_answer
    )


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
            test["category"],

        "user_request":
            user_request,

        "dependency":
            test["dependency"],

        "expected_tools":
            "|".join(
                evaluation["expected_tools"]
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
# 25. SAVE RESULTS
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
# 26. SUMMARY
# ================================================================

def build_summary(
    rows
):

    total = len(
        rows
    )

    exact = sum(
        1
        for row in rows
        if row[
            "exact_plan_correct"
        ]
    )

    first_argument = sum(
        1
        for row in rows
        if row[
            "first_argument_correct"
        ]
    )

    unnecessary = sum(
        1
        for row in rows
        if row[
            "has_unnecessary_tool"
        ]
    )

    completed = sum(
        1
        for row in rows
        if row[
            "task_complete"
        ]
    )

    required_found = sum(
        row[
            "required_tools_found"
        ]
        for row in rows
    )

    required_total = sum(
        row[
            "required_tools_total"
        ]
        for row in rows
    )

    dependent_rows = [
        row
        for row in rows
        if (
            row["dependency"]
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


    def pct(
        numerator,
        denominator
    ):

        if denominator == 0:

            return 0.0

        return round(
            numerator
            / denominator
            * 100,
            2
        )


    return {

        "experiment":
            (
                "Stage5_V3_"
                "knowledge_subquestion_extraction"
            ),

        "requests":
            total,

        "exact_plan_accuracy":
            pct(
                exact,
                total
            ),

        "required_tool_coverage":
            pct(
                required_found,
                required_total
            ),

        "first_argument_accuracy":
            pct(
                first_argument,
                total
            ),

        "unnecessary_tool_call_rate":
            pct(
                unnecessary,
                total
            ),

        "dependency_success_rate":
            pct(
                dependent_success,
                len(dependent_rows)
            ),

        "structural_task_completion":
            pct(
                completed,
                total
            ),
    }


# ================================================================
# 27. SAVE SUMMARY
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
# 28. MAIN
# ================================================================

def main():

    tests = load_tests()

    print(
        "\nSTAGE 5 — V3 KNOWLEDGE SUB-QUESTION EXTRACTION"
    )

    print(
        "=" * 76
    )

    print(
        "Tests loaded:",
        len(tests)
    )


    tokenizer, model = (
        load_planner_model()
    )

    rows = []

    for test in tests:

        rows.append(
            run_test(
                test,
                tokenizer,
                model
            )
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
        "STAGE 5 — V3 RESULTS"
    )

    print(
        "=" * 76
    )

    print(
        "Requests:",
        summary["requests"]
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