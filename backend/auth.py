from typing import Any

import jwt
from jwt import PyJWKClient
from jwt.exceptions import (
    PyJWKClientConnectionError,
    PyJWTError,
)

from fastapi import Depends, HTTPException, status
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)


# These are public identifiers, not passwords or secrets.
COGNITO_USER_POOL_ID = "eu-west-1_c5VlvzE6e"
COGNITO_APP_CLIENT_ID = "5bjde7k6tj65g17kk96vdqpert"

COGNITO_ISSUER = (
    "https://cognito-idp.eu-west-1.amazonaws.com/"
    f"{COGNITO_USER_POOL_ID}"
)

COGNITO_JWKS_URL = (
    f"{COGNITO_ISSUER}/.well-known/jwks.json"
)


# Cache Cognito's public signing keys.
# The client refreshes the key set when necessary.
jwks_client = PyJWKClient(
    COGNITO_JWKS_URL,
    cache_jwk_set=True,
    lifespan=300,
    timeout=5,
)

bearer_scheme = HTTPBearer(auto_error=False)


def unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="A valid Cognito access token is required.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_access_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        bearer_scheme
    ),
) -> dict[str, Any]:
    """
    Verify a Cognito access token before allowing API access.

    This is a synchronous dependency so signing-key network
    requests run outside FastAPI's asynchronous event loop.
    """
    if credentials is None:
        raise unauthorized()

    if credentials.scheme.lower() != "bearer":
        raise unauthorized()

    token = credentials.credentials

    try:
        header = jwt.get_unverified_header(token)

        # Never select the allowed algorithm from an incoming token.
        if header.get("alg") != "RS256":
            raise unauthorized()

        key_id = header.get("kid")

        if not isinstance(key_id, str) or not key_id:
            raise unauthorized()

        signing_key = jwks_client.get_signing_key_from_jwt(token)

        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=COGNITO_ISSUER,
            options={
                "require": [
                    "exp",
                    "iat",
                    "iss",
                    "sub",
                    "client_id",
                    "token_use",
                ],
                # Cognito access tokens identify the app through
                # client_id. This app does not use resource binding.
                "verify_aud": False,
            },
        )

        if claims.get("token_use") != "access":
            raise unauthorized()

        if claims.get("client_id") != COGNITO_APP_CLIENT_ID:
            raise unauthorized()

        subject = claims.get("sub")

        if not isinstance(subject, str) or not subject.strip():
            raise unauthorized()

        return claims

    except PyJWKClientConnectionError:
        # Fail closed if Cognito's keys cannot be retrieved.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Sign-in verification is temporarily unavailable. "
                "Please try again."
            ),
        ) from None

    except PyJWTError:
        # Do not expose tokens or internal verification details.
        raise unauthorized() from None