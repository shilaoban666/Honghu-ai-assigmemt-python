"""JWT authentication — mirrors JwtService.java + JwtAuthenticationFilter.java."""

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings

security_scheme = HTTPBearer(auto_error=False)


class JwtPrincipal:
    """Mirrors JwtPrincipal.java."""
    def __init__(self, user_id: str, username: str, role: str):
        self.user_id = user_id
        self.username = username
        self.role = role


def issue_token(user_id: str, username: str, role: str) -> dict:
    """Issue a signed JWT. Mirrors JwtService.issue()."""
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=max(1, settings.security.jwt.ttl_hours))
    payload = {
        "iss": settings.security.jwt.issuer,
        "sub": user_id,
        "username": username,
        "role": role,
        "iat": now,
        "exp": expires_at,
    }
    token = jwt.encode(payload, settings.security.jwt.secret, algorithm="HS256")
    return {
        "token": token,
        "token_type": "Bearer",
        "expires_at": expires_at.isoformat(),
    }


def parse_token(token: str) -> JwtPrincipal | None:
    """Parse and validate JWT. Mirrors JwtService.parse()."""
    if not token:
        return None
    try:
        claims = jwt.decode(token, settings.security.jwt.secret, algorithms=["HS256"])
        return JwtPrincipal(
            user_id=claims["sub"],
            username=claims.get("username", ""),
            role=claims.get("role", "USER"),
        )
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme),
) -> JwtPrincipal:
    """FastAPI dependency: extract current user from JWT or dev header fallback."""
    # Try JWT first
    if credentials and credentials.credentials:
        principal = parse_token(credentials.credentials)
        if principal:
            return principal

    # Dev header fallback (mirrors dev-header-fallback in Java)
    if settings.security.dev_header_fallback:
        user_id = request.headers.get("X-User-Id")
        if user_id:
            return JwtPrincipal(user_id=user_id, username="dev-user", role="ADMIN")

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing authentication")


async def get_optional_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme),
) -> JwtPrincipal | None:
    """Like get_current_user but returns None instead of raising 401."""
    if credentials and credentials.credentials:
        principal = parse_token(credentials.credentials)
        if principal:
            return principal
    if settings.security.dev_header_fallback:
        user_id = request.headers.get("X-User-Id")
        if user_id:
            return JwtPrincipal(user_id=user_id, username="dev-user", role="ADMIN")
    return None
