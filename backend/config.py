
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
# CORS CONFIGURATION
# ================================================================
#
# Allow the local Next.js development frontend and the
# deployed AWS Amplify frontend to access FastAPI.
#
# Origins must not end with a trailing slash.
# ================================================================

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://production-web.d1c7vj8bsta3cp.amplifyapp.com",
]
