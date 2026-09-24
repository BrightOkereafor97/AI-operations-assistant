from datetime import (
    datetime,
    timezone,
)

from fastapi import (
    APIRouter,
)

from backend.config import (
    APP_NAME,
    APP_VERSION,
)

from backend.schemas import (
    HealthResponse,
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
# HEALTH CHECK
# ================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
)
def health_check():

    return HealthResponse(
        status="healthy",

        service=APP_NAME,

        version=APP_VERSION,

        timestamp=(
            datetime.now(
                timezone.utc
            )
            .isoformat()
        ),
    )