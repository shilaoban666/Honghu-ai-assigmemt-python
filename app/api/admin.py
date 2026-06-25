"""Admin API endpoints — mirrors AdminController.java."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.orm import (
    AiModelDefinition,
    AiProvider,
    AiModelPricing,
    User,
    UserRole,
    UserStatus,
    RoleQuotaConfig,
)
from app.security.jwt import get_current_user, JwtPrincipal

router = APIRouter(prefix="/api/v1/admin", tags=["Admin"])


def require_admin(current_user: JwtPrincipal = Depends(get_current_user)):
    if current_user.role != "ADMIN":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user


# ── Model Management ─────────────────────────────────────────────

@router.get("/models")
async def list_models(
    db: AsyncSession = Depends(get_db),
    _admin: JwtPrincipal = Depends(require_admin),
):
    """List all AI model definitions."""
    result = await db.execute(select(AiModelDefinition).order_by(AiModelDefinition.model_code))
    models = result.scalars().all()
    return [
        {
            "modelCode": m.model_code,
            "displayName": m.display_name,
            "providerCode": m.provider_code,
            "apiModelName": m.api_model_name,
            "level": m.level,
            "score": m.score,
            "localModel": m.local_model,
            "supportsStream": m.supports_stream,
            "enabled": m.enabled,
            "description": m.description,
        }
        for m in models
    ]


@router.post("/models")
async def create_model(
    body: dict,
    db: AsyncSession = Depends(get_db),
    _admin: JwtPrincipal = Depends(require_admin),
):
    """Create a new AI model definition."""
    model = AiModelDefinition(
        model_code=body["modelCode"],
        display_name=body.get("displayName", body["modelCode"]),
        provider_code=body.get("providerCode", "deepseek-cloud"),
        api_model_name=body.get("apiModelName", body["modelCode"]),
        level=body.get("level", 0),
        score=body.get("score", 0),
        local_model=body.get("localModel", False),
        supports_stream=body.get("supportsStream", True),
        enabled=body.get("enabled", True),
        description=body.get("description"),
    )
    db.add(model)
    await db.commit()
    return {"message": "Model created", "modelCode": model.model_code}


@router.put("/models/{model_code}")
async def update_model(
    model_code: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
    _admin: JwtPrincipal = Depends(require_admin),
):
    """Update an AI model definition."""
    result = await db.execute(select(AiModelDefinition).where(AiModelDefinition.model_code == model_code))
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    for field in ["displayName", "providerCode", "apiModelName", "level", "score", "enabled", "description"]:
        snake = "".join(["_" + c.lower() if c.isupper() else c for c in field]).lstrip("_")
        if snake in body:
            setattr(model, snake, body[snake])

    await db.commit()
    return {"message": "Model updated"}


@router.delete("/models/{model_code}")
async def delete_model(
    model_code: str,
    db: AsyncSession = Depends(get_db),
    _admin: JwtPrincipal = Depends(require_admin),
):
    """Delete an AI model definition."""
    result = await db.execute(select(AiModelDefinition).where(AiModelDefinition.model_code == model_code))
    model = result.scalar_one_or_none()
    if model:
        await db.delete(model)
        await db.commit()
    return {"message": "Model deleted"}


# ── Provider Management ──────────────────────────────────────────

@router.get("/providers")
async def list_providers(
    db: AsyncSession = Depends(get_db),
    _admin: JwtPrincipal = Depends(require_admin),
):
    """List all AI providers."""
    result = await db.execute(select(AiProvider).order_by(AiProvider.provider_code))
    providers = result.scalars().all()
    return [
        {
            "providerCode": p.provider_code,
            "displayName": p.display_name,
            "providerType": p.provider_type.value if p.provider_type else None,
            "baseUrl": p.base_url,
            "chatCompletionsPath": p.chat_completions_path,
            "useApiKey": p.use_api_key,
            "enabled": p.enabled,
        }
        for p in providers
    ]


@router.post("/providers")
async def create_provider(
    body: dict,
    db: AsyncSession = Depends(get_db),
    _admin: JwtPrincipal = Depends(require_admin),
):
    """Create a new AI provider."""
    provider = AiProvider(
        provider_code=body["providerCode"],
        display_name=body.get("displayName", body["providerCode"]),
        provider_type=body.get("providerType", "OPENAI_COMPATIBLE"),
        base_url=body.get("baseUrl", ""),
        chat_completions_path=body.get("chatCompletionsPath", "/v1/chat/completions"),
        use_api_key=body.get("useApiKey", True),
        api_key_cipher=body.get("apiKey"),  # In real impl: encrypt with AES-GCM
        api_key_header=body.get("apiKeyHeader", "Authorization"),
        api_key_prefix=body.get("apiKeyPrefix", "Bearer "),
        enabled=body.get("enabled", True),
    )
    db.add(provider)
    await db.commit()
    return {"message": "Provider created", "providerCode": provider.provider_code}


# ── User Management ──────────────────────────────────────────────

@router.get("/users")
async def list_users(
    db: AsyncSession = Depends(get_db),
    _admin: JwtPrincipal = Depends(require_admin),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
):
    """List all users (admin)."""
    offset = (page - 1) * page_size
    result = await db.execute(
        select(User).order_by(User.created_at.desc()).offset(offset).limit(page_size)
    )
    users = result.scalars().all()
    count_result = await db.execute(select(User))
    total = len(count_result.scalars().all())
    return {
        "items": [
            {
                "userId": u.user_id,
                "username": u.username,
                "nickname": u.nickname,
                "email": u.email,
                "userRole": u.user_role.value if u.user_role else "USER",
                "userStatus": u.user_status.value if u.user_status else "ACTIVE",
                "createdAt": u.created_at.isoformat() if u.created_at else None,
            }
            for u in users
        ],
        "total": total,
        "page": page,
        "pageSize": page_size,
    }


@router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
    _admin: JwtPrincipal = Depends(require_admin),
):
    """Update user role."""
    result = await db.execute(select(User).where(User.user_id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    new_role = body.get("role", "USER")
    try:
        user.user_role = UserRole(new_role)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid role: {new_role}")
    await db.commit()
    return {"message": f"User role updated to {new_role}"}


# ── Quota Management ─────────────────────────────────────────────

@router.get("/quota-config")
async def list_quota_configs(
    db: AsyncSession = Depends(get_db),
    _admin: JwtPrincipal = Depends(require_admin),
):
    """List role quota configs."""
    result = await db.execute(select(RoleQuotaConfig))
    configs = result.scalars().all()
    return [
        {
            "role": c.role,
            "dailyLimit": float(c.daily_limit) if c.daily_limit else None,
            "monthlyLimit": float(c.monthly_limit) if c.monthly_limit else None,
            "concurrentRequests": c.concurrent_requests,
            "description": c.description,
        }
        for c in configs
    ]


@router.put("/quota-config/{role}")
async def update_quota_config(
    role: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
    _admin: JwtPrincipal = Depends(require_admin),
):
    """Update role quota config."""
    result = await db.execute(select(RoleQuotaConfig).where(RoleQuotaConfig.role == role))
    config = result.scalar_one_or_none()
    if not config:
        config = RoleQuotaConfig(role=role)
        db.add(config)

    if "dailyLimit" in body:
        config.daily_limit = body["dailyLimit"]
    if "monthlyLimit" in body:
        config.monthly_limit = body["monthlyLimit"]
    if "concurrentRequests" in body:
        config.concurrent_requests = body["concurrentRequests"]
    if "description" in body:
        config.description = body["description"]

    await db.commit()
    return {"message": "Quota config updated"}
