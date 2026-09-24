import sys

from functools import lru_cache
from pathlib import Path
from typing import Any


# ================================================================
# PROJECT PATH
#
# agent_service.py:
#
# backend/services/agent_service.py
#
# parents[2] therefore points to the Project 3 root.
# ================================================================

SERVICE_FILE = (
    Path(__file__)
    .resolve()
)

PROJECT_ROOT = (
    SERVICE_FILE
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ================================================================
# EXISTING PROJECT 3 AGENT
#
# We are reusing the tested system.
# ================================================================

from stage5_multi_tool import (
    load_planner_model,
)

from stage7_agent_loop import (
    run_agent_loop,
)


# ================================================================
# JSON-SAFE CONVERSION
# ================================================================

def make_json_safe(
    value: Any,
) -> Any:

    if value is None:

        return None


    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):

        return value


    if isinstance(
        value,
        Path,
    ):

        return str(
            value
        )


    if isinstance(
        value,
        dict,
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
        ),
    ):

        return [
            make_json_safe(
                item
            )
            for item in value
        ]


    if hasattr(
        value,
        "item",
    ):

        try:

            return value.item()

        except Exception:

            pass


    return str(
        value
    )


# ================================================================
# LOAD AGENT RUNTIME ONCE
#
# The planner model is expensive to load.
#
# lru_cache keeps one loaded instance in this Python process.
# ================================================================

@lru_cache(
    maxsize=1
)
def get_agent_runtime():

    print(
        "\nLoading production agent runtime..."
    )

    tokenizer, model = (
        load_planner_model()
    )

    print(
        "Production agent runtime ready."
    )

    return (
        tokenizer,
        model,
    )


# ================================================================
# RUN EXISTING AGENT
# ================================================================

def execute_agent_request(
    user_request: str,
    max_steps: int,
) -> dict[str, Any]:

    tokenizer, model = (
        get_agent_runtime()
    )

    state = (
        run_agent_loop(
            user_request,
            tokenizer,
            model,
            max_steps=max_steps,
        )
    )

    safe_state = (
        make_json_safe(
            state
        )
    )

    return safe_state