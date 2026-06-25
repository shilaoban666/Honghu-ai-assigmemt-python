"""Capability / Skill marketplace API endpoints — full implementation."""

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, async_session_factory
from app.models.orm import Skill, UserSkillInstall, SessionSkillSetting, SkillSource
from app.schemas.dto import CapabilityDto, CapabilityPageDto
from app.skill.resolver import SkillResolver

router = APIRouter(prefix="/api/v1/capabilities", tags=["Capabilities"])

skill_resolver = SkillResolver(async_session_factory)


@router.get("")
async def list_capabilities(
    db: AsyncSession = Depends(get_db),
    user_id: str | None = Header(None, alias="X-User-Id"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100, alias="pageSize"),
    category: str | None = None,
    search: str | None = None,
) -> CapabilityPageDto:
    """List capabilities (skills) with pagination and optional search/category filter."""
    offset = (page - 1) * page_size

    # Base query
    query = select(Skill).where(Skill.enabled == True)

    if category:
        query = query.where(Skill.category == category)
    if search:
        query = query.where(
            (Skill.name.ilike(f"%{search}%")) |
            (Skill.display_name.ilike(f"%{search}%")) |
            (Skill.description.ilike(f"%{search}%"))
        )

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar_one()

    # Fetch page
    query = query.order_by(Skill.install_count.desc()).offset(offset).limit(page_size)
    result = await db.execute(query)
    skills = result.scalars().all()

    # Check which skills the current user has installed
    installed_ids: set[str] = set()
    if user_id:
        install_result = await db.execute(
            select(UserSkillInstall.skill_id).where(
                UserSkillInstall.user_id == user_id,
                UserSkillInstall.enabled == True,
            )
        )
        installed_ids = {row[0] for row in install_result.all()}

    return CapabilityPageDto(
        items=[
            CapabilityDto(
                skillId=s.skill_id,
                name=s.name,
                displayName=s.display_name,
                description=s.description,
                source=s.source.value if s.source else "MARKET",
                iconUrl=s.icon_url,
                version=s.version,
                author=s.author,
                category=s.category,
                tags=s.tags.split(",") if s.tags else None,
                installed=s.skill_id in installed_ids,
                enabled=s.enabled,
                installCount=s.install_count,
            )
            for s in skills
        ],
        total=total,
        page=page,
        pageSize=page_size,
    )


@router.post("/{skill_id}/install")
async def install_skill(
    skill_id: str,
    user_id: str | None = Header(None, alias="X-User-Id"),
    db: AsyncSession = Depends(get_db),
):
    """Install a skill for the current user."""
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    # Check skill exists
    skill_result = await db.execute(select(Skill).where(Skill.skill_id == skill_id))
    skill = skill_result.scalar_one_or_none()
    if not skill:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Skill not found")

    # Check if already installed
    existing = await db.execute(
        select(UserSkillInstall).where(
            UserSkillInstall.user_id == user_id,
            UserSkillInstall.skill_id == skill_id,
        )
    )
    install = existing.scalar_one_or_none()

    if install:
        install.enabled = True
    else:
        install = UserSkillInstall(
            user_id=user_id,
            skill_id=skill_id,
            enabled=True,
        )
        db.add(install)
        skill.install_count = (skill.install_count or 0) + 1

    await db.commit()
    return {"message": "Skill installed", "skillId": skill_id}


@router.post("/{skill_id}/uninstall")
async def uninstall_skill(
    skill_id: str,
    user_id: str | None = Header(None, alias="X-User-Id"),
    db: AsyncSession = Depends(get_db),
):
    """Uninstall a skill for the current user."""
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    result = await db.execute(
        select(UserSkillInstall).where(
            UserSkillInstall.user_id == user_id,
            UserSkillInstall.skill_id == skill_id,
        )
    )
    install = result.scalar_one_or_none()
    if install:
        install.enabled = False
        await db.commit()

    return {"message": "Skill uninstalled", "skillId": skill_id}


@router.get("/installed")
async def list_installed(
    user_id: str | None = Header(None, alias="X-User-Id"),
    db: AsyncSession = Depends(get_db),
):
    """List skills installed by the current user."""
    if not user_id:
        return {"skills": []}

    result = await db.execute(
        select(UserSkillInstall, Skill)
        .join(Skill, UserSkillInstall.skill_id == Skill.skill_id)
        .where(
            UserSkillInstall.user_id == user_id,
            UserSkillInstall.enabled == True,
        )
    )
    rows = result.all()
    return {
        "skills": [
            {
                "skillId": skill.skill_id,
                "name": skill.name,
                "displayName": skill.display_name,
                "description": skill.description,
                "source": skill.source.value if skill.source else None,
                "category": skill.category,
                "installedAt": install.installed_at.isoformat() if install.installed_at else None,
            }
            for install, skill in rows
        ]
    }


@router.post("/sessions/{session_id}/skills/{skill_id}/toggle")
async def toggle_session_skill(
    session_id: str,
    skill_id: str,
    body: dict,
    db: AsyncSession = Depends(get_db),
):
    """Enable or disable a skill for a specific session."""
    enabled = body.get("enabled", True)

    result = await db.execute(
        select(SessionSkillSetting).where(
            SessionSkillSetting.session_id == session_id,
            SessionSkillSetting.skill_id == skill_id,
        )
    )
    setting = result.scalar_one_or_none()

    if setting:
        setting.enabled = enabled
    else:
        setting = SessionSkillSetting(
            session_id=session_id,
            skill_id=skill_id,
            enabled=enabled,
        )
        db.add(setting)

    await db.commit()
    return {"message": f"Skill {'enabled' if enabled else 'disabled'} for session"}
