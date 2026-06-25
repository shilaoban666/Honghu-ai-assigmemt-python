"""Skill Resolver — mirrors SkillResolverService.java.

Resolves which tools are available for a given user/session context.
Combines: built-in tools, installed skills, session-level toggles,
and wraps everything with audit logging.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.tools import BaseTool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.orm import SessionSkillSetting, UserSkillInstall, Skill
from app.skill.builtin_tools import BUILTIN_TOOLS

logger = logging.getLogger(__name__)


class SkillResolver:
    """Resolves the set of active tools for a given session context."""

    def __init__(self, db_session_factory):
        self.db_factory = db_session_factory

    async def resolve_tools(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> list[BaseTool]:
        """Resolve the complete set of tools available for this session.

        Resolution order (matches Java):
        1. Built-in tools (always available)
        2. User-installed skills from marketplace
        3. Session-level skill toggles (can override user installs)
        """
        tools: list[BaseTool] = list(BUILTIN_TOOLS)  # Start with builtins

        if not user_id and not session_id:
            # Guest/anon: only builtins
            return tools

        async with self.db_factory() as db:
            # Get session skill settings (session-level overrides)
            session_skills: dict[str, bool] = {}
            if session_id:
                result = await db.execute(
                    select(SessionSkillSetting).where(
                        SessionSkillSetting.session_id == session_id
                    )
                )
                for setting in result.scalars().all():
                    session_skills[setting.skill_id] = setting.enabled

            # Get user-installed skills
            if user_id:
                result = await db.execute(
                    select(UserSkillInstall).where(
                        UserSkillInstall.user_id == user_id,
                        UserSkillInstall.enabled == True,
                    )
                )
                installs = result.scalars().all()
                installed_skill_ids = [inst.skill_id for inst in installs]

                if installed_skill_ids:
                    # Fetch skill definitions
                    skill_result = await db.execute(
                        select(Skill).where(Skill.skill_id.in_(installed_skill_ids))
                    )
                    skills = {s.skill_id: s for s in skill_result.scalars().all()}

                    for skill_id in installed_skill_ids:
                        # Check session override
                        if skill_id in session_skills:
                            if not session_skills[skill_id]:
                                continue  # Disabled at session level

                        skill = skills.get(skill_id)
                        if not skill or not skill.enabled:
                            continue

                        # Create tool wrapper based on source type
                        if skill.source and skill.source.value == "BUILTIN":
                            continue  # Already included
                        elif skill.source and skill.source.value == "MCP":
                            # MCP tool — creates a tool wrapper
                            pass  # TODO: Phase 3b — MCP tool instantiation
                        elif skill.source and skill.source.value == "CLI":
                            # CLI tool
                            pass  # TODO: Phase 3b — CLI tool instantiation

        return tools

    async def resolve_for_chat(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> list[BaseTool]:
        """Shortcut for chat resolution — returns tools ready for model binding."""
        return await self.resolve_tools(user_id, session_id)


class ToolAuditLogger:
    """Logs tool invocations for audit trail — mirrors AuditingToolCallback.java."""

    def __init__(self, db_session_factory):
        self.db_factory = db_session_factory

    async def log_invocation(
        self,
        tool_name: str,
        skill_id: str | None,
        arguments: dict,
        result: str,
        success: bool,
        error: str | None,
        latency_ms: int,
        user_id: str | None = None,
        session_id: str | None = None,
        chat_id: int | None = None,
    ):
        """Record a tool invocation to the database."""
        try:
            async with self.db_factory() as db:
                from app.models.orm import ToolInvocationLog
                import json

                log_entry = ToolInvocationLog(
                    tool_id=tool_name,
                    skill_id=skill_id,
                    user_id=user_id,
                    session_id=session_id,
                    chat_id=chat_id,
                    arguments=json.dumps(arguments, ensure_ascii=False) if arguments else None,
                    result=result[:2000] if result else None,
                    success=success,
                    error_message=error,
                    latency_ms=latency_ms,
                )
                db.add(log_entry)
                await db.commit()
        except Exception as e:
            logger.warning(f"Failed to log tool invocation: {e}")


class ToolGuard:
    """Danger level guard for tools — mirrors ToolGuard.java."""

    def __init__(self, db_session_factory):
        self.db_factory = db_session_factory

    async def check_access(
        self,
        tool_name: str,
        danger_level: str,
        user_id: str | None,
        session_id: str | None,
    ) -> tuple[bool, str]:
        """Check if a tool can be executed. Returns (allowed, reason)."""
        # CRITICAL tools are always blocked
        if danger_level == "CRITICAL":
            return False, "CRITICAL danger level — execution blocked"

        # HIGH danger tools require admin or explicit approval
        if danger_level == "HIGH":
            if not user_id or not session_id:
                return False, "HIGH danger tools require authentication and explicit approval"
            # Check for session-level approval
            async with self.db_factory() as db:
                from app.models.orm import SessionToolApproval
                result = await db.execute(
                    select(SessionToolApproval).where(
                        SessionToolApproval.session_id == session_id,
                        SessionToolApproval.tool_id == tool_name,
                        SessionToolApproval.approved == True,
                    )
                )
                approval = result.scalar_one_or_none()
                if not approval:
                    return False, "HIGH danger tool requires user approval"
            return True, ""

        # MEDIUM and LOW are allowed
        return True, ""
