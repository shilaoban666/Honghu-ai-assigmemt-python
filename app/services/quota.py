"""Quota service — mirrors QuotaService.java.

Checks daily/monthly quota limits before AI calls.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.orm import (
    AiUsageEvent,
    UsageEventStatus,
    User,
    UserRole,
    RoleQuotaConfig,
    PlanEntitlement,
    EntitlementType,
    Workspace,
)
from app.schemas.dto import QuotaCheckResult, QuotaSnapshot

STANDARD_TOKENS_PER_CNY = Decimal("1000000")
ONE_HUNDRED = Decimal("100")


class QuotaService:
    """Checks quota before AI calls and provides quota snapshots."""

    async def check_before_call(
        self,
        db: AsyncSession,
        user_id: str | None,
        workspace_id: str | None,
    ) -> QuotaCheckResult:
        """Check if user/workspace is within quota limits before allowing a call."""
        if not user_id:
            return QuotaCheckResult(
                allowed=True,
                dailyUsed=Decimal("0"),
                monthlyUsed=Decimal("0"),
                unlimited=True,
                resetAt=datetime.now(timezone.utc) + timedelta(days=1),
            )

        result = await db.execute(select(User).where(User.user_id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            raise ValueError(f"User not found: {user_id}")

        # Determine limits
        daily_limit, monthly_limit = await self._resolve_limits(db, user, workspace_id)
        if daily_limit is None and monthly_limit is None:
            return QuotaCheckResult(
                allowed=True,
                dailyUsed=Decimal("0"),
                monthlyUsed=Decimal("0"),
                unlimited=True,
                resetAt=datetime.now(timezone.utc) + timedelta(days=1),
            )

        now = datetime.now(timezone.utc)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        daily_used = await self._used_cost(db, user_id, workspace_id, day_start)
        monthly_used = await self._used_cost(db, user_id, workspace_id, month_start)

        if daily_limit is not None and daily_used >= daily_limit:
            return QuotaCheckResult(
                allowed=False,
                reason="DAILY_LIMIT_EXCEEDED",
                dailyUsed=daily_used,
                dailyLimit=daily_limit,
                monthlyUsed=monthly_used,
                monthlyLimit=monthly_limit,
                resetAt=day_start + timedelta(days=1),
            )

        if monthly_limit is not None and monthly_used >= monthly_limit:
            return QuotaCheckResult(
                allowed=False,
                reason="MONTHLY_LIMIT_EXCEEDED",
                dailyUsed=daily_used,
                dailyLimit=daily_limit,
                monthlyUsed=monthly_used,
                monthlyLimit=monthly_limit,
                resetAt=(now.replace(day=1) + timedelta(days=32)).replace(day=1, hour=0, minute=0, second=0),
            )

        return QuotaCheckResult(
            allowed=True,
            dailyUsed=daily_used,
            dailyLimit=daily_limit,
            monthlyUsed=monthly_used,
            monthlyLimit=monthly_limit,
            resetAt=day_start + timedelta(days=1),
        )

    async def _resolve_limits(
        self, db: AsyncSession, user: User, workspace_id: str | None
    ) -> tuple[Decimal | None, Decimal | None]:
        """Resolve daily/monthly limits for a user/workspace context."""
        role = user.user_role or UserRole.USER
        config_result = await db.execute(
            select(RoleQuotaConfig).where(RoleQuotaConfig.role == role.value)
        )
        config = config_result.scalar_one_or_none()
        daily = Decimal(str(config.daily_limit)) if config and config.daily_limit else None
        monthly = Decimal(str(config.monthly_limit)) if config and config.monthly_limit else None
        return daily, monthly

    async def _used_cost(
        self, db: AsyncSession, user_id: str, workspace_id: str | None, since: datetime
    ) -> Decimal:
        """Calculate total billed cost since a given time."""
        result = await db.execute(
            select(func.coalesce(func.sum(AiUsageEvent.cost_billed), 0)).where(
                AiUsageEvent.user_id == user_id,
                AiUsageEvent.status == UsageEventStatus.SUCCESS,
                AiUsageEvent.created_at >= since,
            )
        )
        total = result.scalar_one()
        return Decimal(str(total))
