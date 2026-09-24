# ================================================================
# APPLICATION CONFIGURATION
# ================================================================

APP_NAME = "AI Operations Assistant API"

APP_VERSION = "1.2.0"


# ================================================================
# AGENT SETTINGS
# ================================================================

DEFAULT_MAX_STEPS = 5

MAX_ALLOWED_STEPS = 10


# ================================================================
# CORS
#
# During local development:
#
# Next.js  -> localhost:3000
# FastAPI  -> localhost:8000
#
# Later the AWS Amplify frontend domain will be added through
# environment variables / production configuration.
# ================================================================

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]