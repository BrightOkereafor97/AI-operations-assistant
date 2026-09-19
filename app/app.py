import inspect
import json
import re
import sys

from pathlib import Path

import streamlit as st


# ================================================================
# 1. PROJECT PATH SETUP
# ================================================================

APP_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

PROJECT_ROOT = (
    APP_DIR
    .parent
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ================================================================
# 2. IMPORT EXISTING PROJECT 3 COMPONENTS
# ================================================================

from stage5_multi_tool import (
    load_planner_model,
)

from stage7_agent_loop import (
    run_agent_loop,
)

from stage8_write_action import (
    validate_write_request,
    extract_ticket_ids,
    extract_reason,
)

from write_tools import (
    create_escalation,
    list_escalations,
)

from state import (
    create_agent_state,
)


# ================================================================
# 3. STREAMLIT PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title=(
        "Operations Assistant"
    ),
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ================================================================
# 4. UI CONSTANTS
# ================================================================

DEFAULT_MAX_STEPS = 5

WRITE_ACTION_PATTERN = re.compile(
    (
        r"\bescalate\b"
        r"|"
        r"\bcreate\s+(?:an\s+)?escalation\b"
    ),
    re.IGNORECASE,
)

UNSUPPORTED_WRITE_PATTERN = re.compile(
    (
        r"^\s*"
        r"(approve|delete|remove|pay|"
        r"update|close|reopen|assign)"
        r"\b"
    ),
    re.IGNORECASE,
)


# ================================================================
# 5. LOAD PLANNER MODEL ONCE
#
# Streamlit reruns the Python script whenever the user interacts
# with the page.
#
# Without caching, FLAN-T5 would reload on every interaction.
#
# st.cache_resource keeps the model in memory.
# ================================================================

@st.cache_resource
def get_planner_model():

    loaded = (
        load_planner_model()
    )

    if (
        not isinstance(
            loaded,
            tuple
        )
        or
        len(loaded) != 2
    ):

        raise RuntimeError(
            (
                "load_planner_model() did not "
                "return the expected "
                "(tokenizer, model) pair."
            )
        )

    tokenizer, model = loaded

    return (
        tokenizer,
        model
    )


# ================================================================
# 6. GENERIC JSON-SAFE CONVERSION
#
# Some results may contain Path objects, sets, NumPy values,
# or other Python objects that Streamlit cannot serialize directly.
# ================================================================

def make_json_safe(
    value
):

    if value is None:

        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        )
    ):

        return value

    if isinstance(
        value,
        Path
    ):

        return str(
            value
        )

    if isinstance(
        value,
        dict
    ):

        return {
            str(key):
                make_json_safe(
                    item
                )
            for (
                key,
                item
            ) in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        )
    ):

        return [
            make_json_safe(
                item
            )
            for item in value
        ]

    if hasattr(
        value,
        "item"
    ):

        try:

            return value.item()

        except Exception:

            pass

    return str(
        value
    )


# ================================================================
# 7. REQUEST-TYPE DETECTION
# ================================================================

def is_write_request(
    user_request
):

    return bool(
        WRITE_ACTION_PATTERN.search(
            user_request
        )
    )


def is_unsupported_write_request(
    user_request
):

    return bool(
        UNSUPPORTED_WRITE_PATTERN.search(
            user_request
        )
    )


# ================================================================
# 8. NORMALIZE TOOL RESULT INTO USER-FACING TEXT
# ================================================================

def result_to_message(
    result
):

    if result is None:

        return (
            "The action completed but "
            "returned no result."
        )

    if isinstance(
        result,
        str
    ):

        return result

    if isinstance(
        result,
        dict
    ):

        preferred_keys = [
            "message",
            "answer",
            "error",
            "detail",
            "status_message",
        ]

        for key in preferred_keys:

            value = (
                result.get(
                    key
                )
            )

            if value:

                return str(
                    value
                )

        escalation_id = (
            result.get(
                "escalation_id"
            )
        )

        if escalation_id:

            return (
                "Escalation created successfully. "
                f"Escalation ID: {escalation_id}"
            )

        return json.dumps(
            make_json_safe(
                result
            ),
            indent=2,
        )

    return str(
        result
    )


# ================================================================
# 9. DETERMINE SUCCESS FROM WRITE-TOOL RESULT
# ================================================================

def write_result_succeeded(
    result
):

    if isinstance(
        result,
        dict
    ):

        if "success" in result:

            return bool(
                result[
                    "success"
                ]
            )

        status = str(
            result.get(
                "status",
                ""
            )
        ).lower()

        outcome = str(
            result.get(
                "outcome",
                ""
            )
        ).lower()

        combined = (
            status
            +
            " "
            +
            outcome
        )

        failure_terms = [
            "fail",
            "reject",
            "invalid",
            "duplicate",
            "error",
        ]

        if any(
            term in combined
            for term in failure_terms
        ):

            return False

    return True


# ================================================================
# 10. CALL CREATE_ESCALATION SAFELY
#
# We inspect the existing function rather than assuming a particular
# keyword signature.
#
# Expected conceptual inputs:
#     ticket_id
#     reason
# ================================================================

def invoke_create_escalation(
    ticket_id,
    reason
):

    signature = (
        inspect.signature(
            create_escalation
        )
    )

    kwargs = {}

    positional_values = []

    for (
        parameter_name,
        parameter
    ) in signature.parameters.items():

        lowered = (
            parameter_name
            .lower()
        )

        if lowered in [
            "ticket_id",
            "ticket",
            "ticketid",
        ]:

            kwargs[
                parameter_name
            ] = ticket_id

            continue

        if lowered in [
            "reason",
            "escalation_reason",
            "description",
        ]:

            kwargs[
                parameter_name
            ] = reason

            continue

        if (
            parameter.default
            is
            inspect.Parameter.empty
        ):

            positional_values.append(
                parameter_name
            )

    # ------------------------------------------------------------
    # Preferred path:
    # the tool uses named ticket/reason arguments.
    # ------------------------------------------------------------

    if kwargs:

        unresolved_required = []

        for name in positional_values:

            if name not in kwargs:

                unresolved_required.append(
                    name
                )

        if not unresolved_required:

            return create_escalation(
                **kwargs
            )

    # ------------------------------------------------------------
    # Fallback:
    # Stage 8 was designed around ticket_id + reason, so if the
    # function has exactly two required positional parameters,
    # call them positionally.
    # ------------------------------------------------------------

    required_parameters = [
        parameter
        for parameter
        in signature.parameters.values()
        if (
            parameter.default
            is
            inspect.Parameter.empty
            and
            parameter.kind
            in [
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ]
        )
    ]

    if len(
        required_parameters
    ) == 2:

        return create_escalation(
            ticket_id,
            reason,
        )

    raise RuntimeError(
        (
            "The UI could not safely determine "
            "the create_escalation() argument structure."
        )
    )


# ================================================================
# 11. BUILD NORMALIZED WRITE STATE
#
# Stage 7 returns an agent state.
#
# Stage 8 was originally written as a benchmark runner, so the UI
# creates a compatible display-state around the actual Stage 8
# validation and write tool.
# ================================================================

def build_write_state(
    user_request
):

    state = (
        create_agent_state(
            user_request
        )
    )

    state[
        "status"
    ] = "RUNNING"

    state[
        "outcome"
    ] = None

    state[
        "stop_reason"
    ] = None

    state[
        "planned_tools"
    ] = [
        "create_escalation"
    ]

    # ------------------------------------------------------------
    # Validate BEFORE write execution.
    # ------------------------------------------------------------

    validation = (
        validate_write_request(
            user_request
        )
    )

    state[
        "write_validation"
    ] = make_json_safe(
        validation
    )

    if not validation.get(
        "ready",
        False,
    ):

        final_answer = str(
            validation.get(
                "message",
                (
                    "More information is required "
                    "before this action can be performed."
                )
            )
        )

        state[
            "status"
        ] = (
            "WAITING_FOR_CLARIFICATION"
        )

        state[
            "outcome"
        ] = validation.get(
            "outcome",
            "CLARIFICATION",
        )

        state[
            "stop_reason"
        ] = "CLARIFICATION"

        state[
            "final_answer"
        ] = final_answer

        return state

    # ------------------------------------------------------------
    # Extract validated ticket ID.
    # ------------------------------------------------------------

    ticket_id = (
        validation.get(
            "ticket_id"
        )
    )

    if not ticket_id:

        ticket_ids = (
            extract_ticket_ids(
                user_request
            )
        )

        if (
            isinstance(
                ticket_ids,
                list
            )
            and
            len(ticket_ids) == 1
        ):

            ticket_id = (
                ticket_ids[0]
            )

        elif isinstance(
            ticket_ids,
            str
        ):

            ticket_id = (
                ticket_ids
            )

    # ------------------------------------------------------------
    # Extract validated reason.
    # ------------------------------------------------------------

    reason = (
        validation.get(
            "reason"
        )
    )

    if not reason:

        extracted_reason = (
            extract_reason(
                user_request
            )
        )

        if isinstance(
            extracted_reason,
            str
        ):

            reason = (
                extracted_reason
            )

        elif isinstance(
            extracted_reason,
            dict
        ):

            reason = (
                extracted_reason.get(
                    "reason"
                )
            )

    # ------------------------------------------------------------
    # Final UI-side safety check.
    # ------------------------------------------------------------

    if (
        not ticket_id
        or
        not reason
    ):

        state[
            "status"
        ] = (
            "WAITING_FOR_CLARIFICATION"
        )

        state[
            "outcome"
        ] = "CLARIFICATION"

        state[
            "stop_reason"
        ] = "CLARIFICATION"

        state[
            "final_answer"
        ] = (
            "Please provide both a ticket ID "
            "and a clear escalation reason."
        )

        return state

    arguments = {
        "ticket_id":
            ticket_id,

        "reason":
            reason,
    }

    # ------------------------------------------------------------
    # Record the tool call for the Stage 11 audit panel.
    # ------------------------------------------------------------

    state[
        "tool_calls"
    ].append(
        {
            "tool_name":
                "create_escalation",

            "arguments":
                arguments,
        }
    )

    state[
        "step_count"
    ] += 1

    # ------------------------------------------------------------
    # Execute the real Stage 8 write tool.
    # ------------------------------------------------------------

    try:

        result = (
            invoke_create_escalation(
                ticket_id,
                reason,
            )
        )

        safe_result = (
            make_json_safe(
                result
            )
        )

        state[
            "tool_results"
        ].append(
            {
                "tool_name":
                    "create_escalation",

                "result":
                    safe_result,
            }
        )

        final_answer = (
            result_to_message(
                result
            )
        )

        success = (
            write_result_succeeded(
                result
            )
        )

        if success:

            state[
                "status"
            ] = "COMPLETED"

            state[
                "outcome"
            ] = "WRITE_COMPLETED"

            state[
                "stop_reason"
            ] = "READY"

        else:

            state[
                "status"
            ] = "FAILED"

            state[
                "outcome"
            ] = "WRITE_REJECTED"

            state[
                "stop_reason"
            ] = "UNRECOVERABLE"

        state[
            "final_answer"
        ] = final_answer

        return state

    except Exception as exc:

        error_text = (
            f"{type(exc).__name__}: {exc}"
        )

        state[
            "errors"
        ].append(
            error_text
        )

        state[
            "status"
        ] = "FAILED"

        state[
            "outcome"
        ] = "WRITE_FAILED"

        state[
            "stop_reason"
        ] = "UNRECOVERABLE"

        state[
            "final_answer"
        ] = (
            "The write action was stopped safely. "
            f"{error_text}"
        )

        return state


# ================================================================
# 12. UNSUPPORTED WRITE STATE
# ================================================================

def build_unsupported_write_state(
    user_request
):

    state = (
        create_agent_state(
            user_request
        )
    )

    state[
        "status"
    ] = "BLOCKED"

    state[
        "outcome"
    ] = "UNSUPPORTED_WRITE"

    state[
        "stop_reason"
    ] = "UNRECOVERABLE"

    state[
        "final_answer"
    ] = (
        "That write action is not available. "
        "The only state-changing action currently "
        "supported is creating a controlled ticket escalation."
    )

    state[
        "planned_tools"
    ] = []

    return state


# ================================================================
# 13. MAIN REQUEST DISPATCHER
# ================================================================

def execute_request(
    user_request,
    tokenizer,
    model,
    max_steps,
):

    if is_unsupported_write_request(
        user_request
    ):

        return (
            build_unsupported_write_state(
                user_request
            ),
            "unsupported_write",
        )

    if is_write_request(
        user_request
    ):

        return (
            build_write_state(
                user_request
            ),
            "write_action",
        )

    state = (
        run_agent_loop(
            user_request,
            tokenizer,
            model,
            max_steps=max_steps,
        )
    )

    return (
        state,
        "agent_loop",
    )


# ================================================================
# 14. RECURSIVE EVIDENCE / SOURCE EXTRACTION
# ================================================================

SOURCE_KEYS = {
    "source",
    "sources",
    "document",
    "document_name",
    "document_title",
    "generation_source",
    "generation_sources",
    "focused_evidence",
    "evidence",
    "chunks",
    "retrieval_scores",
    "retrieval_details",
    "chunk_id",
    "chunk_similarity",
    "sentence_similarity",
    "adjusted_evidence_score",
}


def collect_source_information(
    value,
    path="root",
    collected=None,
):

    if collected is None:

        collected = []

    if isinstance(
        value,
        dict
    ):

        for (
            key,
            item
        ) in value.items():

            current_path = (
                f"{path}.{key}"
            )

            if (
                key
                in
                SOURCE_KEYS
            ):

                collected.append(
                    {
                        "path":
                            current_path,

                        "key":
                            key,

                        "value":
                            make_json_safe(
                                item
                            ),
                    }
                )

            collect_source_information(
                item,
                current_path,
                collected,
            )

    elif isinstance(
        value,
        (
            list,
            tuple,
        )
    ):

        for (
            index,
            item
        ) in enumerate(
            value
        ):

            collect_source_information(
                item,
                f"{path}[{index}]",
                collected,
            )

    return collected


# ================================================================
# 15. DISPLAY STATUS BADGE
# ================================================================

def render_status(
    state
):

    stop_reason = (
        state.get(
            "stop_reason"
        )
    )

    status = (
        state.get(
            "status",
            "UNKNOWN"
        )
    )

    if stop_reason == "READY":

        st.success(
            f"Status: {status} | Stop reason: READY"
        )

        return

    if stop_reason == "CLARIFICATION":

        st.warning(
            (
                f"Status: {status} | "
                "Stop reason: CLARIFICATION"
            )
        )

        return

    if stop_reason == "MAX_STEPS":

        st.warning(
            (
                f"Status: {status} | "
                "Stop reason: MAX_STEPS"
            )
        )

        return

    if stop_reason == "UNRECOVERABLE":

        st.error(
            (
                f"Status: {status} | "
                "Stop reason: UNRECOVERABLE"
            )
        )

        return

    st.info(
        (
            f"Status: {status} | "
            f"Stop reason: {stop_reason}"
        )
    )


# ================================================================
# 16. DISPLAY EXECUTION METRICS
# ================================================================

def render_execution_metrics(
    state
):

    tool_calls = (
        state.get(
            "tool_calls",
            []
        )
    )

    errors = (
        state.get(
            "errors",
            []
        )
    )

    step_count = (
        state.get(
            "step_count",
            0
        )
    )

    col1, col2, col3, col4 = (
        st.columns(
            4
        )
    )

    with col1:

        st.metric(
            "Steps",
            step_count,
        )

    with col2:

        st.metric(
            "Tool Calls",
            len(
                tool_calls
            ),
        )

    with col3:

        st.metric(
            "Errors",
            len(
                errors
            ),
        )

    with col4:

        st.metric(
            "Stop Reason",
            (
                state.get(
                    "stop_reason"
                )
                or
                "—"
            ),
        )


# ================================================================
# 17. DISPLAY TOOL CALLS
# ================================================================

def render_tool_calls(
    state
):

    tool_calls = (
        state.get(
            "tool_calls",
            []
        )
    )

    if not tool_calls:

        st.info(
            "No tools were called."
        )

        return

    for (
        index,
        tool_call
    ) in enumerate(
        tool_calls,
        start=1,
    ):

        if isinstance(
            tool_call,
            dict
        ):

            tool_name = (
                tool_call.get(
                    "tool_name"
                )
                or
                tool_call.get(
                    "name"
                )
                or
                f"Tool call {index}"
            )

        else:

            tool_name = (
                f"Tool call {index}"
            )

        with st.expander(
            (
                f"Tool Call {index}: "
                f"{tool_name}"
            ),
            expanded=True,
        ):

            st.json(
                make_json_safe(
                    tool_call
                )
            )


# ================================================================
# 18. DISPLAY TOOL RESULTS / OBSERVATIONS
# ================================================================

def render_tool_results(
    state
):

    tool_results = (
        state.get(
            "tool_results",
            []
        )
    )

    if not tool_results:

        st.info(
            "No tool observations are available."
        )

        return

    for (
        index,
        tool_result
    ) in enumerate(
        tool_results,
        start=1,
    ):

        with st.expander(
            f"Observation {index}",
            expanded=True,
        ):

            st.json(
                make_json_safe(
                    tool_result
                )
            )


# ================================================================
# 19. DISPLAY SOURCES / EVIDENCE
# ================================================================

def render_sources(
    state
):

    source_information = (
        collect_source_information(
            state.get(
                "tool_results",
                []
            )
        )
    )

    if not source_information:

        st.info(
            (
                "No knowledge-source evidence was "
                "used for this request."
            )
        )

        return

    for (
        index,
        item
    ) in enumerate(
        source_information,
        start=1,
    ):

        with st.expander(
            (
                f"Evidence {index}: "
                f"{item['key']}"
            )
        ):

            st.caption(
                item[
                    "path"
                ]
            )

            value = (
                item[
                    "value"
                ]
            )

            if isinstance(
                value,
                (
                    dict,
                    list,
                )
            ):

                st.json(
                    value
                )

            else:

                st.write(
                    value
                )


# ================================================================
# 20. DISPLAY ERRORS
# ================================================================

def render_errors(
    state
):

    errors = (
        state.get(
            "errors",
            []
        )
    )

    if not errors:

        st.success(
            "No execution errors recorded."
        )

        return

    for (
        index,
        error
    ) in enumerate(
        errors,
        start=1,
    ):

        st.error(
            (
                f"Error {index}: "
                f"{error}"
            )
        )


# ================================================================
# 21. SESSION HISTORY
# ================================================================

if (
    "request_history"
    not in
    st.session_state
):

    st.session_state[
        "request_history"
    ] = []


# ================================================================
# 22. SIDEBAR
# ================================================================

with st.sidebar:

    st.title(
        "Agent Controls"
    )

    st.write(
        (
            "Configure the bounded agent loop "
            "and inspect the system safely."
        )
    )

    max_steps = st.slider(
        "Maximum agent steps",
        min_value=1,
        max_value=10,
        value=DEFAULT_MAX_STEPS,
        step=1,
    )

    st.divider()

    st.subheader(
        "Example Requests"
    )

    st.code(
        "Check ticket TKT-005."
    )

    st.code(
        (
            "Get the amount on EXP-010 "
            "and add 7200 to it."
        )
    )

    st.code(
        (
            "What does the company "
            "refund policy say?"
        )
    )

    st.code(
        (
            "Check TKT-003 and tell me "
            "what response target applies "
            "to its priority."
        )
    )

    st.code(
        (
            "Check the ticket and tell me "
            "who owns it."
        )
    )

    st.code(
        (
            "Escalate TKT-017 because "
            "the issue is blocking "
            "customer operations."
        )
    )

    st.divider()

    st.caption(
        (
            "The only enabled state-changing "
            "tool is controlled ticket escalation."
        )
    )

    if st.button(
        "Clear Session History"
    ):

        st.session_state[
            "request_history"
        ] = []

        st.rerun()


# ================================================================
# 23. PAGE HEADER
# ================================================================

st.title(
    "Operations Assistant"
)

st.write(
    (
        "A bounded tool-using AI agent for "
        "ticket lookup, expense claims, calculations, "
        "company knowledge, and controlled escalations."
    )
)

st.caption(
    (
        "Project 3 — Stage 11: "
        "Application / UI Layer"
    )
)


# ================================================================
# 24. LOAD MODEL
# ================================================================

try:

    with st.spinner(
        "Loading agent planner..."
    ):

        tokenizer, planner_model = (
            get_planner_model()
        )

except Exception as exc:

    st.error(
        (
            "The planner model could not be loaded.\n\n"
            f"{type(exc).__name__}: {exc}"
        )
    )

    st.stop()


# ================================================================
# 25. USER REQUEST FORM
#
# Using a form prevents execution while the user is still typing.
# ================================================================

with st.form(
    "agent_request_form",
    clear_on_submit=False,
):

    user_request = st.text_area(
        "What would you like the operations assistant to do?",
        height=120,
        placeholder=(
            "Example: Check TKT-003 and tell me "
            "what response target applies to its priority."
        ),
    )

    submitted = st.form_submit_button(
        "Run Assistant",
        type="primary",
    )


# ================================================================
# 26. EXECUTE REQUEST
# ================================================================

if submitted:

    cleaned_request = (
        user_request.strip()
    )

    if not cleaned_request:

        st.warning(
            "Please enter a request."
        )

    else:

        with st.spinner(
            "Agent is working..."
        ):

            try:

                state, execution_mode = (
                    execute_request(
                        cleaned_request,
                        tokenizer,
                        planner_model,
                        max_steps,
                    )
                )

                safe_state = (
                    make_json_safe(
                        state
                    )
                )

                history_record = {
                    "request":
                        cleaned_request,

                    "execution_mode":
                        execution_mode,

                    "state":
                        safe_state,
                }

                st.session_state[
                    "request_history"
                ].append(
                    history_record
                )

            except Exception as exc:

                safe_state = {
                    "user_request":
                        cleaned_request,

                    "messages":
                        [],

                    "tool_calls":
                        [],

                    "tool_results":
                        [],

                    "step_count":
                        0,

                    "errors": [
                        (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )
                    ],

                    "final_answer":
                        (
                            "The assistant stopped because "
                            "an application-level error occurred."
                        ),

                    "status":
                        "FAILED",

                    "outcome":
                        "APPLICATION_ERROR",

                    "stop_reason":
                        "UNRECOVERABLE",
                }

                history_record = {
                    "request":
                        cleaned_request,

                    "execution_mode":
                        "application_error",

                    "state":
                        safe_state,
                }

                st.session_state[
                    "request_history"
                ].append(
                    history_record
                )


# ================================================================
# 27. DISPLAY MOST RECENT EXECUTION
# ================================================================

if st.session_state[
    "request_history"
]:

    latest = (
        st.session_state[
            "request_history"
        ][-1]
    )

    latest_state = (
        latest[
            "state"
        ]
    )

    st.divider()

    st.subheader(
        "Latest Result"
    )

    st.caption(
        (
            "Execution mode: "
            f"{latest['execution_mode']}"
        )
    )

    render_status(
        latest_state
    )

    render_execution_metrics(
        latest_state
    )

    st.markdown(
        "### Final Answer"
    )

    final_answer = (
        latest_state.get(
            "final_answer"
        )
    )

    if final_answer:

        st.write(
            final_answer
        )

    else:

        st.info(
            "No final answer was produced."
        )

    (
        tab_tools,
        tab_results,
        tab_sources,
        tab_errors,
        tab_state,
    ) = st.tabs(
        [
            "Tool Calls",
            "Observations",
            "Sources & Evidence",
            "Errors",
            "Full Debug State",
        ]
    )

    with tab_tools:

        render_tool_calls(
            latest_state
        )

    with tab_results:

        render_tool_results(
            latest_state
        )

    with tab_sources:

        render_sources(
            latest_state
        )

    with tab_errors:

        render_errors(
            latest_state
        )

    with tab_state:

        st.json(
            latest_state
        )


# ================================================================
# 28. REQUEST HISTORY
# ================================================================

if st.session_state[
    "request_history"
]:

    st.divider()

    st.subheader(
        "Session History"
    )

    for (
        index,
        record
    ) in enumerate(
        reversed(
            st.session_state[
                "request_history"
            ]
        ),
        start=1,
    ):

        state = (
            record[
                "state"
            ]
        )

        label = (
            f"{index}. "
            f"{record['request'][:80]}"
        )

        with st.expander(
            label
        ):

            st.write(
                "**Request:**"
            )

            st.write(
                record[
                    "request"
                ]
            )

            st.write(
                "**Execution mode:**",
                record[
                    "execution_mode"
                ]
            )

            st.write(
                "**Stop reason:**",
                state.get(
                    "stop_reason"
                )
            )

            st.write(
                "**Final answer:**"
            )

            st.write(
                state.get(
                    "final_answer"
                )
            )


# ================================================================
# 29. WRITE-ACTION STORAGE VIEW
# ================================================================

st.divider()

with st.expander(
    "Stored Escalations"
):

    try:

        escalations = (
            list_escalations()
        )

        if escalations:

            st.json(
                make_json_safe(
                    escalations
                )
            )

        else:

            st.info(
                "No escalations are currently stored."
            )

    except Exception as exc:

        st.warning(
            (
                "Escalation storage could not be displayed: "
                f"{type(exc).__name__}: {exc}"
            )
        )