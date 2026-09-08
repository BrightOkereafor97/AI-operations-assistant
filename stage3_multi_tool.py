import csv
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from tools import (
    get_ticket,
    calculate_expense,
    search_company_knowledge,
    get_expense_claim,
)


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FILE = PROJECT_ROOT / "data" / "stage3_requests.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = RESULTS_DIR / "routing_results.csv"
SUMMARY_FILE = RESULTS_DIR / "routing_summary.csv"


# ================================================================
# MODEL
# ================================================================

MODEL_NAME = "google/flan-t5-base"


# ================================================================
# ROUTE OPTIONS
# ================================================================

ROUTE_OPTIONS = {
    "KNOWLEDGE": "search_company_knowledge",
    "TICKET": "get_ticket",
    "CALCULATE": "calculate_expense",
    "EXPENSE_CLAIM": "get_expense_claim",
    "NO_TOOL": "NO_TOOL",
}


# ================================================================
# LOAD MODEL
# ================================================================

def load_model():

    print(f"\nLoading routing model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    print("Routing model ready.")

    return tokenizer, model


# ================================================================
# GENERAL MODEL GENERATION
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
# CONSTRAINED ROUTER PROMPT
# ================================================================

def create_constrained_router_prompt(
    user_request
):

    return f"""
Choose exactly one category for the request.

KNOWLEDGE:
Company policy, rules, procedures or internal knowledge.
Examples: annual leave, sick leave, remote work, refunds,
travel policy, escalation policy.

TICKET:
Read or look up a specific support ticket.
Examples: ticket 10, TKT-017, support case #3.

CALCULATE:
Add, total or sum expense amounts.

EXPENSE_CLAIM:
Read or look up a specific existing expense claim.
Examples: EXP-010, claim 4.

NO_TOOL:
The request cannot be handled by these read-only tools.
Use for greetings, general knowledge, creative writing,
sending, deleting, approving, modifying or updating.

Important:

Read ticket = TICKET.
Delete/update ticket = NO_TOOL.

Read claim = EXPENSE_CLAIM.
Approve/update claim = NO_TOOL.

Add expense amounts = CALCULATE.
Company policy question = KNOWLEDGE.

Request:
{user_request}

Category:
""".strip()


# ================================================================
# SCORE ROUTE CANDIDATE
# ================================================================

def score_route_candidate(
    prompt,
    candidate,
    tokenizer,
    model
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    target = tokenizer(
        candidate,
        return_tensors="pt"
    )

    labels = target["input_ids"].clone()

    labels[
        labels == tokenizer.pad_token_id
    ] = -100

    with torch.no_grad():

        outputs = model(
            **inputs,
            labels=labels
        )

    score = -float(
        outputs.loss.item()
    )

    return score


# ================================================================
# MODEL ROUTING
# ================================================================

def route_request(
    user_request,
    tokenizer,
    model
):

    prompt = create_constrained_router_prompt(
        user_request
    )

    token_count = len(
        tokenizer(prompt)["input_ids"]
    )

    print(
        f"[DEBUG] Router prompt tokens: {token_count}"
    )

    scores = {}

    for candidate in ROUTE_OPTIONS:

        score = score_route_candidate(
            prompt,
            candidate,
            tokenizer,
            model
        )

        scores[candidate] = score

    selected_category = max(
        scores,
        key=scores.get
    )

    selected_tool = ROUTE_OPTIONS[
        selected_category
    ]

    sorted_scores = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    trace = " | ".join(
        f"{name}={score:.4f}"
        for name, score in sorted_scores
    )

    raw_route = (
        f"SELECTED={selected_category}"
        f" | {trace}"
    )

    return raw_route, selected_tool


# ================================================================
# ROUTING VALIDATION / SAFETY GUARDRAIL
# ================================================================

WRITE_ACTION_PATTERN = re.compile(
    r"\b("
    r"approve|approved|reject|rejected|"
    r"delete|remove|send|email|"
    r"update|modify|change|edit|"
    r"create|close|reopen"
    r")\b",
    re.IGNORECASE,
)


TICKET_ID_PATTERN = re.compile(
    r"\bTKT-\d+\b",
    re.IGNORECASE,
)


TICKET_NUMBER_PATTERN = re.compile(
    r"\b("
    r"ticket|support\s+case|case"
    r")"
    r"\s*(?:number|#)?\s*"
    r"\d+\b",
    re.IGNORECASE,
)


CLAIM_ID_PATTERN = re.compile(
    r"\bEXP-\d+\b",
    re.IGNORECASE,
)


CLAIM_NUMBER_PATTERN = re.compile(
    r"\b("
    r"expense\s+claim|claim"
    r")"
    r"\s*(?:number|#)?\s*"
    r"\d+\b",
    re.IGNORECASE,
)


ARITHMETIC_WORD_PATTERN = re.compile(
    r"\b("
    r"add|total|sum|calculate|plus"
    r")\b",
    re.IGNORECASE,
)


COMPANY_KNOWLEDGE_PATTERN = re.compile(
    r"\b("
    r"company|policy|policies|procedure|procedures|"
    r"annual\s+leave|sick\s+leave|remote\s+work|"
    r"refund|refunds|travel\s+expenses?|travel\s+policy|"
    r"escalation\s+policy|escalation\s+process|"
    r"requesting\s+annual\s+leave"
    r")\b",
    re.IGNORECASE,
)


def contains_write_action(
    user_request
):

    return bool(
        WRITE_ACTION_PATTERN.search(
            user_request
        )
    )


def contains_ticket_reference(
    user_request
):

    return bool(
        TICKET_ID_PATTERN.search(
            user_request
        )
        or
        TICKET_NUMBER_PATTERN.search(
            user_request
        )
    )


def contains_claim_reference(
    user_request
):

    return bool(
        CLAIM_ID_PATTERN.search(
            user_request
        )
        or
        CLAIM_NUMBER_PATTERN.search(
            user_request
        )
    )


def contains_expense_calculation(
    user_request
):

    if not ARITHMETIC_WORD_PATTERN.search(
        user_request
    ):
        return False

    numbers = re.findall(
        r"\d+(?:\.\d+)?",
        user_request
    )

    return len(numbers) >= 2


def contains_company_knowledge_cue(
    user_request
):

    return bool(
        COMPANY_KNOWLEDGE_PATTERN.search(
            user_request
        )
    )


# ================================================================
# VALIDATE MODEL ROUTE
# ================================================================

def validate_route(
    user_request,
    model_tool
):

    # ------------------------------------------------------------
    # WRITE ACTIONS
    # ------------------------------------------------------------

    if contains_write_action(
        user_request
    ):

        return (
            "NO_TOOL",
            "SAFETY_OVERRIDE: write/action request detected"
        )

    # ------------------------------------------------------------
    # TICKET REFERENCE
    # ------------------------------------------------------------

    if contains_ticket_reference(
        user_request
    ):

        return (
            "get_ticket",
            "STRUCTURAL_OVERRIDE: ticket reference detected"
        )

    # ------------------------------------------------------------
    # EXPENSE CLAIM REFERENCE
    # ------------------------------------------------------------

    if contains_claim_reference(
        user_request
    ):

        return (
            "get_expense_claim",
            "STRUCTURAL_OVERRIDE: expense claim reference detected"
        )

    # ------------------------------------------------------------
    # CALCULATION
    # ------------------------------------------------------------

    if contains_expense_calculation(
        user_request
    ):

        return (
            "calculate_expense",
            "STRUCTURAL_OVERRIDE: expense arithmetic detected"
        )

    # ------------------------------------------------------------
    # INVALID TICKET CALL
    # ------------------------------------------------------------

    if model_tool == "get_ticket":

        return (
            "NO_TOOL",
            (
                "VALIDATION_OVERRIDE: "
                "ticket tool selected without ticket reference"
            )
        )

    # ------------------------------------------------------------
    # INVALID CLAIM CALL
    # ------------------------------------------------------------

    if model_tool == "get_expense_claim":

        return (
            "NO_TOOL",
            (
                "VALIDATION_OVERRIDE: "
                "claim tool selected without claim reference"
            )
        )

    # ------------------------------------------------------------
    # INVALID CALCULATOR CALL
    # ------------------------------------------------------------

    if model_tool == "calculate_expense":

        return (
            "NO_TOOL",
            (
                "VALIDATION_OVERRIDE: "
                "calculator selected without valid arithmetic request"
            )
        )

    # ------------------------------------------------------------
    # COMPANY KNOWLEDGE VALIDATION
    # ------------------------------------------------------------

    if model_tool == "search_company_knowledge":

        if not contains_company_knowledge_cue(
            user_request
        ):

            return (
                "NO_TOOL",
                (
                    "VALIDATION_OVERRIDE: "
                    "knowledge tool selected without company-knowledge cue"
                )
            )

    return (
        model_tool,
        "MODEL_ROUTE_ACCEPTED"
    )


# ================================================================
# TICKET ARGUMENT PROMPT
# ================================================================

def create_ticket_argument_prompt(
    user_request
):

    return f"""
Extract the support ticket number.

Return ONLY the number.

Examples:

Request:
Check ticket TKT-005.

Number:
5

Request:
Who owns ticket 10?

Number:
10

Request:
Check support case #3.

Number:
3

Request:
Show me ticket number 12.

Number:
12

Request:
What is the status of TKT-017?

Number:
17

Request:
{user_request}

Number:
""".strip()


# ================================================================
# EXPENSE CLAIM ARGUMENT PROMPT
# ================================================================

def create_claim_argument_prompt(
    user_request
):

    return f"""
Extract the expense claim number.

Return ONLY the number.

Examples:

Request:
What happened to EXP-010?

Number:
10

Request:
Check claim number 4.

Number:
4

Request:
What is the status of EXP-014?

Number:
14

Request:
Who submitted expense claim 7?

Number:
7

Request:
Show claim EXP-002.

Number:
2

Request:
{user_request}

Number:
""".strip()


# ================================================================
# CALCULATION ARGUMENT PROMPT
# ================================================================

def create_calculation_argument_prompt(
    user_request
):

    return f"""
Extract ONLY the expense amounts that need to be added.

Return the numbers separated by commas.

Do not calculate the total.
Do not explain anything.

Examples:

Request:
Add 32000 hotel and 8500 transport.

Output:
32000,8500

Request:
Calculate the total of 18500, 7200 and 45000.

Output:
18500,7200,45000

Request:
What is 12000 hotel plus 3500 taxi?

Output:
12000,3500

Request:
Please total these expenses: 5000, 9000 and 2500.

Output:
5000,9000,2500

Request:
Add the expense amounts 15000 and 4200.

Output:
15000,4200

Request:
{user_request}

Output:
""".strip()


# ================================================================
# NORMALIZATION HELPERS
# ================================================================

def extract_number(
    value
):

    numbers = re.findall(
        r"\d+",
        str(value)
    )

    if not numbers:
        return None

    return int(
        numbers[-1]
    )


def normalize_ticket_id(
    value
):

    number = extract_number(
        value
    )

    if number is None:
        return None

    return f"TKT-{number:03d}"


def normalize_claim_id(
    value
):

    number = extract_number(
        value
    )

    if number is None:
        return None

    return f"EXP-{number:03d}"


def normalize_amounts(
    value
):

    numbers = re.findall(
        r"\d+(?:\.\d+)?",
        str(value)
    )

    if not numbers:
        return None

    return [
        float(number)
        for number in numbers
    ]


# ================================================================
# EXPLICIT ID EXTRACTION
# ================================================================

def extract_explicit_ticket_id(
    user_request
):

    match = TICKET_ID_PATTERN.search(
        user_request
    )

    if not match:
        return None

    return match.group(0).upper()


def extract_explicit_claim_id(
    user_request
):

    match = CLAIM_ID_PATTERN.search(
        user_request
    )

    if not match:
        return None

    return match.group(0).upper()


# ================================================================
# ARGUMENT EXTRACTION
# ================================================================

def extract_argument(
    tool_name,
    user_request,
    tokenizer,
    model
):

    # ------------------------------------------------------------
    # COMPANY KNOWLEDGE
    # ------------------------------------------------------------

    if tool_name == "search_company_knowledge":

        return (
            user_request,
            user_request,
            None
        )

    # ------------------------------------------------------------
    # TICKET
    # ------------------------------------------------------------

    if tool_name == "get_ticket":

        # First preserve an exact ID such as TKT-017.

        explicit_id = extract_explicit_ticket_id(
            user_request
        )

        if explicit_id:

            return (
                explicit_id,
                explicit_id,
                None
            )

        # Otherwise ask the LLM to extract the number.

        prompt = create_ticket_argument_prompt(
            user_request
        )

        raw_argument = generate_model_output(
            prompt,
            tokenizer,
            model,
            max_new_tokens=10
        )

        argument = normalize_ticket_id(
            raw_argument
        )

        if argument is None:

            return (
                raw_argument,
                None,
                "Could not extract ticket_id."
            )

        return (
            raw_argument,
            argument,
            None
        )

    # ------------------------------------------------------------
    # EXPENSE CLAIM
    # ------------------------------------------------------------

    if tool_name == "get_expense_claim":

        # First preserve an exact ID such as EXP-014.

        explicit_id = extract_explicit_claim_id(
            user_request
        )

        if explicit_id:

            return (
                explicit_id,
                explicit_id,
                None
            )

        # Otherwise ask the LLM to extract the claim number.

        prompt = create_claim_argument_prompt(
            user_request
        )

        raw_argument = generate_model_output(
            prompt,
            tokenizer,
            model,
            max_new_tokens=10
        )

        argument = normalize_claim_id(
            raw_argument
        )

        if argument is None:

            return (
                raw_argument,
                None,
                "Could not extract claim_id."
            )

        return (
            raw_argument,
            argument,
            None
        )

    # ------------------------------------------------------------
    # CALCULATION
    # ------------------------------------------------------------

    if tool_name == "calculate_expense":

        prompt = create_calculation_argument_prompt(
            user_request
        )

        raw_argument = generate_model_output(
            prompt,
            tokenizer,
            model,
            max_new_tokens=30
        )

        argument = normalize_amounts(
            raw_argument
        )

        if not argument:

            return (
                raw_argument,
                None,
                "Could not extract expense amounts."
            )

        return (
            raw_argument,
            argument,
            None
        )

    # ------------------------------------------------------------
    # NO TOOL
    # ------------------------------------------------------------

    if tool_name == "NO_TOOL":

        return (
            "",
            None,
            None
        )

    # ------------------------------------------------------------
    # UNKNOWN TOOL
    # ------------------------------------------------------------

    return (
        "",
        None,
        "Unknown routing output."
    )


# ================================================================
# EXECUTION
# ================================================================

def execute_tool(
    tool_name,
    argument
):

    try:

        if tool_name == "get_ticket":

            return (
                get_ticket(argument),
                None
            )

        if tool_name == "get_expense_claim":

            return (
                get_expense_claim(argument),
                None
            )

        if tool_name == "calculate_expense":

            return (
                calculate_expense(argument),
                None
            )

        if tool_name == "search_company_knowledge":

            return (
                search_company_knowledge(argument),
                None
            )

        if tool_name == "NO_TOOL":

            return (
                None,
                None
            )

        return (
            None,
            "Unknown tool."
        )

    except Exception as error:

        return (
            None,
            f"{type(error).__name__}: {error}"
        )


# ================================================================
# FINAL ANSWER
# ================================================================

def build_final_answer(
    tool_name,
    tool_result,
    tool_error
):

    if tool_error:

        return (
            "The request could not be completed: "
            + tool_error
        )

    if tool_name == "NO_TOOL":

        return (
            "This request cannot be completed "
            "using the available Stage 3 read-only tools."
        )

    if tool_name == "get_ticket":

        return (
            f"{tool_result['ticket_id']} is "
            f"{tool_result['status']}. "
            f"Priority: {tool_result['priority']}. "
            f"Owner: {tool_result['owner']}. "
            f"Summary: {tool_result['summary']}"
        )

    if tool_name == "get_expense_claim":

        return (
            f"{tool_result['claim_id']} was submitted by "
            f"{tool_result['employee']}. "
            f"Category: {tool_result['category']}. "
            f"Amount: {tool_result['amount']}. "
            f"Status: {tool_result['status']}."
        )

    if tool_name == "calculate_expense":

        return (
            f"Calculated expense total: "
            f"{tool_result}"
        )

    if tool_name == "search_company_knowledge":

        if isinstance(
            tool_result,
            dict
        ):

            return tool_result.get(
                "answer",
                str(tool_result)
            )

        return str(
            tool_result
        )

    return str(
        tool_result
    )


# ================================================================
# LOAD TESTS
# ================================================================

def load_tests():

    with DATA_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        return list(
            csv.DictReader(file)
        )


# ================================================================
# FORMAT ARGUMENT FOR EVALUATION
# ================================================================

def argument_to_string(
    tool_name,
    argument
):

    if argument is None:
        return ""

    if tool_name == "calculate_expense":

        values = []

        for value in argument:

            if float(value).is_integer():

                values.append(
                    str(int(value))
                )

            else:

                values.append(
                    str(value)
                )

        return ",".join(values)

    return str(argument)


# ================================================================
# MAIN
# ================================================================

def main():

    tests = load_tests()

    print(
        f"\nStage 3 tests loaded: {len(tests)}"
    )

    tokenizer, model = load_model()

    rows = []

    for test in tests:

        print(
            "\n"
            + "=" * 70
        )

        print(
            test["request_id"],
            "-",
            test["category"]
        )

        print(
            "User:",
            test["user_request"]
        )

        # ========================================================
        # 1. MODEL ROUTING
        # ========================================================

        raw_route, model_tool = route_request(
            test["user_request"],
            tokenizer,
            model
        )

        print(
            "Model router output:",
            raw_route
        )

        print(
            "Model selected tool:",
            model_tool
        )

        # ========================================================
        # 2. ROUTE VALIDATION
        # ========================================================

        (
            tool_name,
            validation_reason
        ) = validate_route(
            test["user_request"],
            model_tool
        )

        print(
            "Validated tool:",
            tool_name
        )

        print(
            "Validation:",
            validation_reason
        )

        # ========================================================
        # 3. ARGUMENT EXTRACTION
        # ========================================================

        (
            raw_argument,
            argument,
            argument_error
        ) = extract_argument(
            tool_name,
            test["user_request"],
            tokenizer,
            model
        )

        print(
            "Raw argument:",
            raw_argument
        )

        print(
            "Structured argument:",
            argument
        )

        # ========================================================
        # 4. EXECUTION
        # ========================================================

        if argument_error:

            tool_result = None
            tool_error = argument_error

        else:

            (
                tool_result,
                tool_error
            ) = execute_tool(
                tool_name,
                argument
            )

        print(
            "Tool result:",
            tool_result
        )

        if tool_error:

            print(
                "Tool error:",
                tool_error
            )

        # ========================================================
        # 5. FINAL ANSWER
        # ========================================================

        final_answer = build_final_answer(
            tool_name,
            tool_result,
            tool_error
        )

        print(
            "Final answer:",
            final_answer
        )

        # ========================================================
        # 6. EVALUATION
        # ========================================================

        tool_selection_correct = int(
            tool_name
            ==
            test["expected_tool"]
        )

        actual_argument_string = argument_to_string(
            tool_name,
            argument
        )

        if test["expected_tool"] == "NO_TOOL":

            argument_correct = int(
                tool_name == "NO_TOOL"
            )

        else:

            argument_correct = int(
                actual_argument_string
                ==
                test["expected_argument"]
            )

        unnecessary_tool_call = int(
            test["expected_tool"] == "NO_TOOL"
            and
            tool_name != "NO_TOOL"
        )

        task_completed = int(
            tool_selection_correct == 1
            and
            argument_correct == 1
            and
            tool_error is None
        )

        rows.append(
            {
                **test,

                "raw_router_output":
                    raw_route,

                "model_selected_tool":
                    model_tool,

                "validated_tool":
                    tool_name,

                "validation_reason":
                    validation_reason,

                "raw_argument":
                    raw_argument,

                "actual_argument":
                    actual_argument_string,

                "tool_result":
                    (
                        json.dumps(
                            tool_result,
                            ensure_ascii=False,
                            default=str
                        )
                        if tool_result is not None
                        else ""
                    ),

                "tool_error":
                    tool_error or "",

                "final_answer":
                    final_answer,

                "tool_selection_correct":
                    tool_selection_correct,

                "argument_correct":
                    argument_correct,

                "unnecessary_tool_call":
                    unnecessary_tool_call,

                "task_completed":
                    task_completed,
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
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    # ============================================================
    # METRICS
    # ============================================================

    total = len(rows)

    tool_selection_accuracy = (
        sum(
            row["tool_selection_correct"]
            for row in rows
        )
        / total
    )

    argument_rows = [
        row
        for row in rows
        if row["expected_tool"] != "NO_TOOL"
    ]

    argument_accuracy = (
        sum(
            row["argument_correct"]
            for row in argument_rows
        )
        / len(argument_rows)
    )

    no_tool_rows = [
        row
        for row in rows
        if row["expected_tool"] == "NO_TOOL"
    ]

    unnecessary_tool_call_rate = (
        sum(
            row["unnecessary_tool_call"]
            for row in no_tool_rows
        )
        / len(no_tool_rows)
    )

    task_completion_rate = (
        sum(
            row["task_completed"]
            for row in rows
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
            "V5_2_guardrail_plus_explicit_id_preservation",

        "total_requests":
            total,

        "tool_selection_accuracy":
            tool_selection_accuracy,

        "argument_accuracy":
            argument_accuracy,

        "unnecessary_tool_call_rate":
            unnecessary_tool_call_rate,

        "task_completion_rate":
            task_completion_rate,
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
        writer.writerow(summary)

    # ============================================================
    # FINAL OUTPUT
    # ============================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STAGE 3 — V5.2 GUARDRAIL + EXPLICIT ID RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Requests: {total}"
    )

    print(
        "Tool Selection Accuracy:",
        f"{tool_selection_accuracy * 100:.2f}%"
    )

    print(
        "Argument Accuracy:",
        f"{argument_accuracy * 100:.2f}%"
    )

    print(
        "Unnecessary Tool Call Rate:",
        f"{unnecessary_tool_call_rate * 100:.2f}%"
    )

    print(
        "Task Completion Rate:",
        f"{task_completion_rate * 100:.2f}%"
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


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()