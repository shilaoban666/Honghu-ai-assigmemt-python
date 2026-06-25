"""User API endpoints — mirrors UserController.java."""

from fastapi import APIRouter, Depends, HTTPException, Header, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from passlib.context import CryptContext

from app.core.database import get_db
from app.models.orm import User, UserStatus, UserRole
from app.schemas.dto import (
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    UserResponse,
    WeChatLoginRequest,
    WeChatAuthorizeResponse,
    VerifyEmailRequest,
    ResendVerificationRequest,
)
from app.security.jwt import issue_token, get_current_user, JwtPrincipal

import uuid
from datetime import datetime, timezone

router = APIRouter(prefix="/api/v1/users", tags=["Users"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@router.post("/register")
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> LoginResponse:
    """Register a new user."""
    # Check uniqueness
    existing = await db.execute(select(User).where(User.username == request.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists")

    if request.email:
        existing = await db.execute(select(User).where(User.email == request.email))
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already exists")

    if request.phone:
        existing = await db.execute(select(User).where(User.phone == request.phone))
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Phone already exists")

    user = User(
        user_id=str(uuid.uuid4()),
        username=request.username,
        nickname=request.nickname,
        email=request.email,
        phone=request.phone,
        password=pwd_context.hash(request.password),
        user_status=UserStatus.PENDING_VERIFICATION if request.email else UserStatus.ACTIVE,
        user_role=UserRole.USER,
    )
    db.add(user)
    await db.commit()

    token_data = issue_token(user.user_id, user.username, user.user_role.value)
    return LoginResponse(
        token=token_data["token"],
        tokenType=token_data["token_type"],
        expiresAt=token_data["expires_at"],
        userId=user.user_id,
        username=user.username,
        role=user.user_role.value,
    )


@router.post("/login")
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> LoginResponse:
    """Login with username/password."""
    result = await db.execute(select(User).where(User.username == request.username))
    user = result.scalar_one_or_none()
    if not user or not pwd_context.verify(request.password, user.password or ""):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token_data = issue_token(user.user_id, user.username, user.user_role.value)
    return LoginResponse(
        token=token_data["token"],
        tokenType=token_data["token_type"],
        expiresAt=token_data["expires_at"],
        userId=user.user_id,
        username=user.username,
        role=user.user_role.value,
    )


@router.get("/me")
async def get_me(
    current_user: JwtPrincipal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Get current user profile."""
    result = await db.execute(select(User).where(User.user_id == current_user.user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return UserResponse(
        userId=user.user_id,
        username=user.username,
        nickname=user.nickname,
        email=user.email,
        phone=user.phone,
        gender=user.gender.value if user.gender else None,
        userStatus=user.user_status.value,
        userRole=user.user_role.value,
    )


@router.get("/{user_id}")
async def get_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Get user by ID."""
    result = await db.execute(select(User).where(User.user_id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return UserResponse(
        userId=user.user_id,
        username=user.username,
        nickname=user.nickname,
        email=user.email,
        phone=user.phone,
        gender=user.gender.value if user.gender else None,
        userStatus=user.user_status.value,
        userRole=user.user_role.value,
    )


@router.post("/wechat/authorize")
async def wechat_authorize(
    request: WeChatLoginRequest,
) -> WeChatAuthorizeResponse:
    """WeChat login authorization URL."""
    # Simplified mock — real impl uses WeChat OAuth
    from urllib.parse import urlencode
    params = urlencode({"code": request.code, "state": request.state or ""})
    return WeChatAuthorizeResponse(redirectUrl=f"/api/v1/users/wechat/callback?{params}")


@router.post("/verify-email")
async def verify_email(
    request: VerifyEmailRequest,
    db: AsyncSession = Depends(get_db),
):
    """Verify user email with token."""
    # In real impl, decode token, find user, set email_verified=True
    return {"message": "Email verified (mock)"}


@router.post("/resend-verification")
async def resend_verification(
    request: ResendVerificationRequest,
):
    """Resend email verification."""
    return {"message": "Verification email resent (mock)"}
