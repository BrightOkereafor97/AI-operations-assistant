from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.auth import require_access_token

from backend.config import (
    ALLOWED_ORIGINS,
    APP_NAME,
    APP_VERSION,
)

from backend.routers.agent import router as agent_router
from backend.routers.health import router as health_router


# ================================================================
# FASTAPI APPLICATION
# ================================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Production API layer for the "
        "AI Operations Assistant."
    ),
)


# ================================================================
# CORS
# ================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# ROUTERS
# ================================================================

# Health checks remain available for AWS monitoring.
app.include_router(health_router)

# All routes in the agent router require a valid access token.
app.include_router(
    agent_router,
    dependencies=[Depends(require_access_token)],
)


# ================================================================
# ROOT
# ================================================================

@app.get("/", tags=["Root"])
def root():
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "message": "AI Operations Assistant API is running.",
    }