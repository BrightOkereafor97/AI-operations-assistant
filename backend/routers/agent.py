import logging

from fastapi import APIRouter, HTTPException

from backend.schemas import AgentRunRequest, AgentRunResponse
from backend.services.agent_service import execute_agent_request


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/agent",
    tags=["Agent"],
)

PUBLIC_ERROR = (
    "The agent could not complete your request. "
    "Please check the record ID or request and try again. "
    "If the problem continues, contact the administrator."
)


@router.post(
    "/run",
    response_model=AgentRunResponse,
)
def run_agent(payload: AgentRunRequest):
    user_request = payload.request.strip()

    if not user_request:
        raise HTTPException(
            status_code=400,
            detail="The request cannot be empty.",
        )

    try:
        state = execute_agent_request(
            user_request,
            payload.max_steps,
        )

        agent_status = str(state.get("status", "UNKNOWN"))
        stop_reason = state.get("stop_reason")
        internal_errors = state.get("errors", [])

        failed = (
            agent_status == "FAILED"
            or stop_reason == "UNRECOVERABLE"
            or bool(internal_errors)
        )

        if failed:
            # Details stay in server logs, not the API response.
            logger.error(
                "Agent execution failed: status=%s stop_reason=%s "
                "errors=%s",
                agent_status,
                stop_reason,
                internal_errors,
            )

        return AgentRunResponse(
            request=user_request,
            status="FAILED" if failed else agent_status,
            stop_reason=stop_reason,
            final_answer=(
                PUBLIC_ERROR
                if failed
                else state.get("final_answer")
            ),
            step_count=int(state.get("step_count", 0)),
            planned_tools=(
                [] if failed else state.get("planned_tools", [])
            ),
            tool_calls=(
                [] if failed else state.get("tool_calls", [])
            ),
            tool_results=(
                [] if failed else state.get("tool_results", [])
            ),
            errors=(
                [{"error": PUBLIC_ERROR}]
                if failed
                else []
            ),
            # Keep the existing response shape for compatibility.
            # Do not send internal messages or planner state.
            full_state={},
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception("Unhandled agent API error")

        raise HTTPException(
            status_code=500,
            detail=PUBLIC_ERROR,
        ) from None