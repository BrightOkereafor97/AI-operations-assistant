import csv
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from tools import (
    get_ticket,
    get_expense_claim,
    calculate_expense,
    search_company_knowledge,
)

# ================================================================
# STAGE 5 — V4 DEPENDENCY CHAINING
#
# V2 solved multi-tool planning with Python validation.
# V3 isolated the knowledge sub-question before calling RAG.
# V4 adds tool-to-tool state passing for dependent requests.
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

TEST_FILE = DATA_DIR / "stage5_requests.csv"
DETAIL_FILE = RESULTS_DIR / "stage5_results_v4.csv"
SUMMARY_FILE = RESULTS_DIR / "stage5_summary_v4.csv"

MODEL_NAME = "google/flan-t5-base"

LABEL_TO_TOOL = {
    "TICKET": "get_ticket",
    "EXPENSE_CLAIM": "get_expense_claim",
    "CALCULATE": "calculate_expense",
    "KNOWLEDGE": "search_company_knowledge",
    "NO_TOOL": "NO_TOOL",
}

TICKET_ID_PATTERN = re.compile(r"\bTKT-\d+\b", re.IGNORECASE)
CLAIM_ID_PATTERN = re.compile(r"\bEXP-\d+\b", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")


# ================================================================
# MODEL
# ================================================================

def load_planner_model():
    print(f"\nLoading Stage 5 planner model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()

    print("Stage 5 planner model ready.")

    return tokenizer, model


def build_planner_prompt(user_request):
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


def plan_tools(user_request, tokenizer, model):
    prompt = build_planner_prompt(user_request)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    print(
        "[DEBUG] Planner prompt tokens:",
        inputs["input_ids"].shape[1],
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            num_beams=4,
        )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )


def parse_tool_plan(raw_output):
    text = str(raw_output).upper()

    matches = re.findall(
        r"EXPENSE_CLAIM|CALCULATE|KNOWLEDGE|TICKET|NO_TOOL",
        text,
    )

    labels = []

    for label in matches:
        if label not in labels:
            labels.append(label)

    real_labels = [
        label
        for label in labels
        if label != "NO_TOOL"
    ]

    if real_labels:
        labels = real_labels

    tools = [
        LABEL_TO_TOOL[label]
        for label in labels
        if label in LABEL_TO_TOOL
    ]

    return labels, tools


# ================================================================
# STRUCTURAL PLAN VALIDATION
# ================================================================

def find_ticket_position(user_request):
    match = TICKET_ID_PATTERN.search(user_request)

    if match:
        return match.start()

    match = re.search(
        r"\bticket\b|\bsupport\s+case\b",
        user_request,
        re.IGNORECASE,
    )

    return match.start() if match else None


def find_claim_position(user_request):
    match = CLAIM_ID_PATTERN.search(user_request)

    if match:
        return match.start()

    match = re.search(
        r"\bexpense\s+claim\b|"
        r"\bclaim\s+(?:number\s+)?#?\s*\d+\b",
        user_request,
        re.IGNORECASE,
    )

    return match.start() if match else None


def find_calculation_position(user_request):
    match = re.search(
        r"\badd\b|\bcalculate\b|\btotal\b|\bsum\b|\bplus\b",
        user_request,
        re.IGNORECASE,
    )

    if not match:
        return None

    cleaned = TICKET_ID_PATTERN.sub("", user_request)
    cleaned = CLAIM_ID_PATTERN.sub("", cleaned)

    numbers = NUMBER_PATTERN.findall(cleaned)

    has_claim = (
        find_claim_position(user_request)
        is not None
    )

    if len(numbers) >= 2:
        return match.start()

    if has_claim and len(numbers) >= 1:
        return match.start()

    return None


def find_knowledge_position(user_request):
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

    positions = [
        lower_request.find(phrase)
        for phrase in phrases
        if lower_request.find(phrase) >= 0
    ]

    return min(positions) if positions else None


def build_structural_plan(user_request):
    detected = []

    candidates = [
        (
            find_ticket_position(user_request),
            "get_ticket",
        ),
        (
            find_claim_position(user_request),
            "get_expense_claim",
        ),
        (
            find_calculation_position(user_request),
            "calculate_expense",
        ),
        (
            find_knowledge_position(user_request),
            "search_company_knowledge",
        ),
    ]

    for position, tool_name in candidates:
        if position is not None:
            detected.append(
                (position, tool_name)
            )

    detected.sort(
        key=lambda item: item[0]
    )

    plan = []

    for _, tool_name in detected:
        if tool_name not in plan:
            plan.append(tool_name)

    return plan


def validate_tool_plan(
    user_request,
    model_tools,
):
    structural_plan = build_structural_plan(
        user_request
    )

    if structural_plan:
        reason = (
            "MODEL_PLAN_ACCEPTED"
            if model_tools == structural_plan
            else "STRUCTURAL_OVERRIDE"
        )

        return (
            structural_plan,
            structural_plan,
            reason,
        )

    if model_tools:
        return (
            model_tools,
            structural_plan,
            "MODEL_FALLBACK",
        )

    return (
        [],
        structural_plan,
        "NO_VALID_PLAN",
    )


# ================================================================
# BASE ARGUMENT EXTRACTION
# ================================================================

def extract_ticket_argument(user_request):
    match = TICKET_ID_PATTERN.search(
        user_request
    )

    if match:
        return match.group(0).upper()

    patterns = [
        r"\bticket\s+(?:number\s+)?#?\s*(\d+)\b",
        r"\bsupport\s+case\s+(?:number\s+)?#?\s*(\d+)\b",
        r"\bcase\s+(?:number\s+)?#?\s*(\d+)\b",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            user_request,
            re.IGNORECASE,
        )

        if match:
            return (
                f"TKT-{int(match.group(1)):03d}"
            )

    return None


def extract_claim_argument(user_request):
    match = CLAIM_ID_PATTERN.search(
        user_request
    )

    if match:
        return match.group(0).upper()

    patterns = [
        r"\bexpense\s+claim\s+(?:number\s+)?#?\s*(\d+)\b",
        r"\bclaim\s+(?:number\s+)?#?\s*(\d+)\b",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            user_request,
            re.IGNORECASE,
        )

        if match:
            return (
                f"EXP-{int(match.group(1)):03d}"
            )

    return None


def extract_calculation_arguments(user_request):
    cleaned = TICKET_ID_PATTERN.sub(
        "",
        user_request,
    )

    cleaned = CLAIM_ID_PATTERN.sub(
        "",
        cleaned,
    )

    numbers = NUMBER_PATTERN.findall(
        cleaned
    )

    return [
        float(number)
        for number in numbers
    ]


# ================================================================
# KNOWLEDGE SUB-QUESTION EXTRACTION
# ================================================================

def contains_knowledge_cue(text):
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


def clean_knowledge_clause(clause):
    clause = clause.strip(" ,.")

    clause = re.sub(
        r"^(?:then\s+)?tell\s+me\s+",
        "",
        clause,
        flags=re.IGNORECASE,
    )

    clause = re.sub(
        r"^and\s+",
        "",
        clause,
        flags=re.IGNORECASE,
    )

    clause = clause.strip()

    if not clause:
        return clause

    if re.match(
        r"^what\s+the\s+company\b",
        clause,
        re.IGNORECASE,
    ):
        clause = re.sub(
            r"^what\s+the\s+company\s+",
            "What does the company ",
            clause,
            flags=re.IGNORECASE,
        )

        clause = re.sub(
            r"\bsays$",
            "say",
            clause,
            flags=re.IGNORECASE,
        )

    elif re.match(
        r"^what\s+our\b",
        clause,
        re.IGNORECASE,
    ):
        clause = re.sub(
            r"^what\s+our\s+",
            "What does our ",
            clause,
            flags=re.IGNORECASE,
        )

        clause = re.sub(
            r"\bsays$",
            "say",
            clause,
            flags=re.IGNORECASE,
        )

    if not clause.endswith("?"):
        clause += "?"

    return clause


def extract_knowledge_question(user_request):
    clauses = re.split(
        r"\s*,?\s+(?:and|then)\s+",
        user_request,
        flags=re.IGNORECASE,
    )

    knowledge_clauses = []

    for clause in clauses:
        if contains_knowledge_cue(clause):
            cleaned = clean_knowledge_clause(
                clause
            )

            if cleaned:
                knowledge_clauses.append(
                    cleaned
                )

    if knowledge_clauses:
        return " ".join(
            knowledge_clauses
        )

    return user_request


# ================================================================
# V4 — DEPENDENCY CHAINING
# ================================================================

def knowledge_depends_on_ticket(
    user_request,
    base_question,
):
    text = (
        f"{user_request} "
        f"{base_question}"
    ).lower()

    phrases = [
        "its priority",
        "that priority",
        "this priority",
        "applies to its priority",
    ]

    return any(
        phrase in text
        for phrase in phrases
    )


def build_knowledge_argument(
    user_request,
    execution_context,
):
    base_question = extract_knowledge_question(
        user_request
    )

    ticket_result = execution_context.get(
        "get_ticket"
    )

    if (
        ticket_result
        and
        knowledge_depends_on_ticket(
            user_request,
            base_question,
        )
    ):
        priority = ticket_result.get(
            "priority"
        )

        if priority:
            return (
                (
                    "What response target applies to "
                    f"{priority} priority support requests?"
                ),
                True,
                (
                    "Used ticket priority "
                    f"'{priority}' from the previous tool result."
                ),
            )

    return (
        base_question,
        False,
        None,
    )


def calculation_depends_on_claim(
    user_request,
):
    has_claim = bool(
        CLAIM_ID_PATTERN.search(
            user_request
        )
    )

    has_arithmetic = bool(
        re.search(
            r"\badd\b|\bplus\b|\btotal\b|\bcalculate\b|\bsum\b",
            user_request,
            re.IGNORECASE,
        )
    )

    references_claim_amount = bool(
        re.search(
            r"\bamount\b.*\bEXP-\d+\b|"
            r"\bEXP-\d+\b.*\bamount\b|"
            r"\badd\b.*\bto\s+it\b",
            user_request,
            re.IGNORECASE,
        )
    )

    return (
        has_claim
        and has_arithmetic
        and references_claim_amount
    )


def build_calculation_argument(
    user_request,
    execution_context,
):
    base_amounts = (
        extract_calculation_arguments(
            user_request
        )
    )

    claim_result = (
        execution_context.get(
            "get_expense_claim"
        )
    )

    if (
        claim_result
        and calculation_depends_on_claim(
            user_request
        )
    ):
        claim_amount = claim_result.get(
            "amount"
        )

        if claim_amount is not None:
            try:
                claim_amount = float(
                    claim_amount
                )
            except (
                TypeError,
                ValueError,
            ):
                return (
                    base_amounts,
                    False,
                    None,
                )

            return (
                [
                    claim_amount,
                    *base_amounts,
                ],
                True,
                (
                    "Used expense-claim amount "
                    f"{claim_amount} from the previous tool result."
                ),
            )

    return (
        base_amounts,
        False,
        None,
    )


def build_tool_argument(
    tool_name,
    user_request,
    execution_context,
):
    if tool_name == "get_ticket":
        return (
            extract_ticket_argument(
                user_request
            ),
            False,
            None,
        )

    if tool_name == "get_expense_claim":
        return (
            extract_claim_argument(
                user_request
            ),
            False,
            None,
        )

    if tool_name == "calculate_expense":
        return build_calculation_argument(
            user_request,
            execution_context,
        )

    if tool_name == "search_company_knowledge":
        return build_knowledge_argument(
            user_request,
            execution_context,
        )

    return (
        None,
        False,
        None,
    )


# ================================================================
# TOOL EXECUTION
# ================================================================

def execute_tool(
    tool_name,
    argument,
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
# FINAL ANSWER
# ================================================================

def format_tool_result(
    tool_name,
    result,
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

    return str(result)


def build_final_answer(
    successful_results,
    errors,
):
    parts = []

    for item in successful_results:
        parts.append(
            format_tool_result(
                item["tool"],
                item["result"],
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

    return " ".join(parts)


# ================================================================
# TEST LOADING / EVALUATION
# ================================================================

def load_tests():
    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Stage 5 test file not found: "
            f"{TEST_FILE}"
        )

    with TEST_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def parse_expected_tools(value):
    return [
        item.strip()
        for item in value.split("|")
        if item.strip()
    ]


def normalize_argument_for_comparison(
    value,
):
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
                ValueError,
            ):
                normalized.append(
                    str(item)
                )

        return "|".join(normalized)

    return str(value).strip().upper()


def evaluate_request(
    test,
    planned_tools,
    executed_calls,
):
    expected_tools = parse_expected_tools(
        test["expected_tools"]
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
        test["expected_first_argument"]
        .strip()
    )

    actual_first_argument = None

    if executed_calls:
        actual_first_argument = (
            executed_calls[0]
            .get("argument")
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

    if dependency_type == "dependent":
        dependency_satisfied = any(
            call.get("used_prior_result")
            for call in executed_calls
        )
    else:
        dependency_satisfied = True

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
            bool(unnecessary_tools),

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
# RUN ONE TEST
# ================================================================

def run_test(
    test,
    tokenizer,
    model,
):
    request_id = test["request_id"]
    user_request = test["user_request"]

    print(
        "\n"
        + "=" * 76
    )

    print(
        request_id,
        "-",
        test["category"],
    )

    print(
        "User:",
        user_request,
    )

    raw_plan = plan_tools(
        user_request,
        tokenizer,
        model,
    )

    model_labels, model_tools = (
        parse_tool_plan(
            raw_plan
        )
    )

    print(
        "Raw planner output:",
        raw_plan,
    )

    print(
        "Model parsed labels:",
        model_labels,
    )

    print(
        "Model planned tools:",
        model_tools,
    )

    (
        validated_tools,
        structural_plan,
        validation_reason,
    ) = validate_tool_plan(
        user_request,
        model_tools,
    )

    print(
        "Structural plan:",
        structural_plan,
    )

    print(
        "Validated tools:",
        validated_tools,
    )

    print(
        "Validation:",
        validation_reason,
    )

    execution_context = {}
    executed_calls = []
    successful_results = []
    errors = []

    for tool_name in validated_tools:
        (
            argument,
            used_prior_result,
            dependency_note,
        ) = build_tool_argument(
            tool_name,
            user_request,
            execution_context,
        )

        print(
            f"\nTool: {tool_name}"
        )

        print(
            "Argument:",
            argument,
        )

        print(
            "Used prior result:",
            used_prior_result,
        )

        if dependency_note:
            print(
                "Dependency note:",
                dependency_note,
            )

        call_record = {
            "tool":
                tool_name,

            "argument":
                argument,

            "used_prior_result":
                used_prior_result,

            "dependency_note":
                dependency_note,

            "result":
                None,

            "error":
                None,
        }

        try:
            result = execute_tool(
                tool_name,
                argument,
            )

            call_record["result"] = result

            execution_context[
                tool_name
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
                result,
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
                error_text,
            )

        executed_calls.append(
            call_record
        )

    final_answer = build_final_answer(
        successful_results,
        errors,
    )

    print(
        "\nFinal answer:",
        final_answer,
    )

    evaluation = evaluate_request(
        test,
        validated_tools,
        executed_calls,
    )

    print("\nEvaluation:")

    print(
        "Exact plan correct:",
        evaluation[
            "exact_plan_correct"
        ],
    )

    print(
        "First argument correct:",
        evaluation[
            "first_argument_correct"
        ],
    )

    print(
        "Dependency satisfied:",
        evaluation[
            "dependency_satisfied"
        ],
    )

    print(
        "Task complete:",
        evaluation[
            "task_complete"
        ],
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
                ensure_ascii=False,
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
# SAVE RESULTS / SUMMARY
# ================================================================

def save_detailed_results(rows):
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    with DETAIL_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


def percentage(
    numerator,
    denominator,
):
    if denominator == 0:
        return 0.0

    return round(
        numerator
        / denominator
        * 100,
        2,
    )


def build_summary(rows):
    total = len(rows)

    exact = sum(
        1
        for row in rows
        if row["exact_plan_correct"]
    )

    first_argument = sum(
        1
        for row in rows
        if row["first_argument_correct"]
    )

    unnecessary = sum(
        1
        for row in rows
        if row["has_unnecessary_tool"]
    )

    completed = sum(
        1
        for row in rows
        if row["task_complete"]
    )

    required_found = sum(
        row["required_tools_found"]
        for row in rows
    )

    required_total = sum(
        row["required_tools_total"]
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

    return {
        "experiment":
            "Stage5_V4_dependency_chaining",

        "requests":
            total,

        "exact_plan_accuracy":
            percentage(
                exact,
                total,
            ),

        "required_tool_coverage":
            percentage(
                required_found,
                required_total,
            ),

        "first_argument_accuracy":
            percentage(
                first_argument,
                total,
            ),

        "unnecessary_tool_call_rate":
            percentage(
                unnecessary,
                total,
            ),

        "dependency_success_rate":
            percentage(
                dependent_success,
                len(dependent_rows),
            ),

        "structural_task_completion":
            percentage(
                completed,
                total,
            ),
    }


def save_summary(summary):
    with SUMMARY_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary.keys(),
        )

        writer.writeheader()
        writer.writerow(summary)


# ================================================================
# MAIN
# ================================================================

def main():
    tests = load_tests()

    print(
        "\nSTAGE 5 — V4 DEPENDENCY CHAINING"
    )

    print(
        "=" * 76
    )

    print(
        "Tests loaded:",
        len(tests),
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
                model,
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
        "STAGE 5 — V4 RESULTS"
    )

    print(
        "=" * 76
    )

    print(
        "Requests:",
        summary["requests"],
    )

    print(
        "Exact Plan Accuracy:",
        f"{summary['exact_plan_accuracy']:.2f}%",
    )

    print(
        "Required Tool Coverage:",
        f"{summary['required_tool_coverage']:.2f}%",
    )

    print(
        "First Argument Accuracy:",
        f"{summary['first_argument_accuracy']:.2f}%",
    )

    print(
        "Unnecessary Tool Call Rate:",
        f"{summary['unnecessary_tool_call_rate']:.2f}%",
    )

    print(
        "Dependency Success Rate:",
        f"{summary['dependency_success_rate']:.2f}%",
    )

    print(
        "Structural Task Completion:",
        f"{summary['structural_task_completion']:.2f}%",
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
