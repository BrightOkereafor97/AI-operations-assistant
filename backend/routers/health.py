from datetime import (
    datetime,
    timezone,
)

from pathlib import (
    Path,
)

from fastapi import (
    APIRouter,
)

from sqlalchemy import (
    text,
)

from backend.config import (
    APP_NAME,
    APP_VERSION,
)

from backend.database import (
    engine,
)

from backend.schemas import (
    ComponentHealth,
    HealthResponse,
)

from tools import (
    calculate_expense,
    get_expense_claim,
    get_ticket,
    search_company_knowledge,
)


# ================================================================
# ROUTER
# ================================================================

router = APIRouter(
    prefix="/api",
    tags=[
        "Health"
    ],
)


# ================================================================
# PROJECT PATH
# ================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

RAG_ROOT = (
    PROJECT_ROOT
    / "rag"
)

RAG_SRC = (
    RAG_ROOT
    / "src"
)

RAG_EMBEDDINGS = (
    RAG_ROOT
    / "results"
    / "embeddings"
)


# ================================================================
# POSTGRESQL CHECK
# ================================================================

def check_postgresql():

    try:

        with engine.connect() as connection:

            connection.execute(
                text(
                    "SELECT 1"
                )
            )


        return ComponentHealth(
            status="connected",
            detail=(
                "Database connection successful."
            ),
        )


    except Exception as error:

        return ComponentHealth(
            status="unavailable",
            detail=(
                f"{type(error).__name__}: "
                f"{error}"
            ),
        )


# ================================================================
# RAG CHECK
#
# This is deliberately lightweight.
#
# We verify that the packaged RAG runtime and vector-store files
# exist without loading the embedding/generation models.
# ================================================================

def check_rag():

    src_ready = (
        RAG_SRC.exists()
        and
        (
            RAG_SRC
            / "generate.py"
        ).exists()
        and
        (
            RAG_SRC
            / "retrieve.py"
        ).exists()
    )


    embedding_ready = False


    if RAG_EMBEDDINGS.exists():

        has_vectors = any(
            RAG_EMBEDDINGS.glob(
                "*.npy"
            )
        )

        has_metadata = any(
            RAG_EMBEDDINGS.glob(
                "*.csv"
            )
        )

        embedding_ready = (
            has_vectors
            and
            has_metadata
        )


    if (
        src_ready
        and
        embedding_ready
    ):

        return ComponentHealth(
            status="available",
            detail=(
                "Packaged RAG runtime and "
                "vector store are available."
            ),
        )


    return ComponentHealth(
        status="unavailable",
        detail=(
            "Required RAG runtime or "
            "embedding files are missing."
        ),
    )


# ================================================================
# AGENT TOOL CHECK
# ================================================================

def check_agent_tools():

    required_tools = [
        get_ticket,
        get_expense_claim,
        calculate_expense,
        search_company_knowledge,
    ]


    if all(
        callable(tool)
        for tool in required_tools
    ):

        return ComponentHealth(
            status="available",
            detail=(
                "Required read-only agent "
                "tools are available."
            ),
        )


    return ComponentHealth(
        status="unavailable",
        detail=(
            "One or more required tools "
            "are unavailable."
        ),
    )


# ================================================================
# HEALTH ENDPOINT
# ================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
)
def health_check():

    components = {

        "fastapi":
            ComponentHealth(
                status="ready",
                detail=(
                    "FastAPI service is running."
                ),
            ),

        "postgresql":
            check_postgresql(),

        "rag_knowledge":
            check_rag(),

        "agent_tools":
            check_agent_tools(),
    }


    expected_states = {

        "fastapi":
            "ready",

        "postgresql":
            "connected",

        "rag_knowledge":
            "available",

        "agent_tools":
            "available",
    }


    healthy = all(

        components[name].status
        ==
        expected_status

        for (
            name,
            expected_status
        )
        in expected_states.items()
    )


    overall_status = (
        "healthy"
        if healthy
        else "degraded"
    )


    return HealthResponse(

        status=
            overall_status,

        service=
            APP_NAME,

        version=
            APP_VERSION,

        timestamp=(
            datetime.now(
                timezone.utc
            )
            .isoformat()
        ),

        components=
            components,
    )