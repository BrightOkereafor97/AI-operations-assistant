from fastapi import (
    APIRouter,
    HTTPException,
)

from backend.schemas import (
    AgentRunRequest,
    AgentRunResponse,
)

from backend.services.agent_service import (
    execute_agent_request,
)


# ================================================================
# ROUTER
# ================================================================

router = APIRouter(
    prefix="/api/agent",
    tags=[
        "Agent"
    ],
)


# ================================================================
# RUN AGENT
# ================================================================

@router.post(
    "/run",
    response_model=AgentRunResponse,
)
def run_agent(
    payload: AgentRunRequest,
):

    user_request = (
        payload.request
        .strip()
    )


    if not user_request:

        raise HTTPException(
            status_code=400,
            detail=(
                "The request cannot be empty."
            ),
        )


    try:

        state = (
            execute_agent_request(
                user_request,
                payload.max_steps,
            )
        )


        return AgentRunResponse(
            request=user_request,

            status=str(
                state.get(
                    "status",
                    "UNKNOWN",
                )
            ),

            stop_reason=(
                state.get(
                    "stop_reason"
                )
            ),

            final_answer=(
                state.get(
                    "final_answer"
                )
            ),

            step_count=int(
                state.get(
                    "step_count",
                    0,
                )
            ),

            planned_tools=(
                state.get(
                    "planned_tools",
                    [],
                )
            ),

            tool_calls=(
                state.get(
                    "tool_calls",
                    [],
                )
            ),

            tool_results=(
                state.get(
                    "tool_results",
                    [],
                )
            ),

            errors=(
                state.get(
                    "errors",
                    [],
                )
            ),

            full_state=state,
        )


    except HTTPException:

        raise


    except Exception as exc:

        print(
            "\n[AGENT API ERROR]",
            type(exc).__name__,
            str(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The agent could not complete "
                "the API request because an "
                "internal server error occurred. "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc