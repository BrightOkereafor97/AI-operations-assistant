from typing import Any

from pydantic import (
    BaseModel,
    Field,
)

from backend.config import (
    DEFAULT_MAX_STEPS,
    MAX_ALLOWED_STEPS,
)


# ================================================================
# HEALTH RESPONSE
# ================================================================

class HealthResponse(
    BaseModel
):

    status: str

    service: str

    version: str

    timestamp: str


# ================================================================
# AGENT REQUEST
# ================================================================

class AgentRunRequest(
    BaseModel
):

    request: str = Field(
        ...,
        min_length=1,
        description=(
            "Natural-language operations "
            "request for the agent."
        ),
        examples=[
            (
                "Check TKT-003 and tell me "
                "what response target applies "
                "to its priority."
            )
        ],
    )

    max_steps: int = Field(
        default=DEFAULT_MAX_STEPS,
        ge=1,
        le=MAX_ALLOWED_STEPS,
        description=(
            "Maximum number of tool execution "
            "steps allowed for this request."
        ),
    )


# ================================================================
# AGENT RESPONSE
# ================================================================

class AgentRunResponse(
    BaseModel
):

    request: str

    status: str

    stop_reason: str | None

    final_answer: str | None

    step_count: int

    planned_tools: list[Any]

    tool_calls: list[Any]

    tool_results: list[Any]

    errors: list[Any]

    full_state: dict[str, Any]