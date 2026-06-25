"""Billing service — mirrors BillingService.java.

Calculates cost for AI calls based on model pricing and token usage.
"""

from decimal import Decimal
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.orm import AiModelPricing
from app.schemas.dto import CostBreakdown, TokenUsage

MILLION = Decimal("1000000")


class BillingService:
    """Calculates cost for AI model calls."""

    async def calculate(
        self,
        db: AsyncSession,
        model_code: str,
        usage: TokenUsage | None,
        at: datetime | None = None,
    ) -> CostBreakdown:
        """Calculate cost for a single AI call."""
        at = at or datetime.now(timezone.utc)

        result = await db.execute(
            select(AiModelPricing).where(
                AiModelPricing.model_code == model_code,
                AiModelPricing.enabled == True,
                AiModelPricing.effective_from <= at,
            ).order_by(AiModelPricing.effective_from.desc()).limit(1)
        )
        pricing = result.scalar_one_or_none()
        if not pricing:
            raise ValueError(f"No active pricing found for model: {model_code}")

        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        cached_prompt_tokens = usage.cached_prompt_tokens if usage else 0
        regular_prompt_tokens = max(0, prompt_tokens - cached_prompt_tokens)

        prompt_price = Decimal(str(pricing.prompt_price_per_million or 0))
        completion_price = Decimal(str(pricing.completion_price_per_million or 0))
        cached_price = Decimal(str(pricing.cached_input_price_per_million or pricing.prompt_price_per_million or 0))
        surcharge = Decimal(str(pricing.request_surcharge or 0))
        markup = Decimal(str(pricing.markup_ratio or 1))

        prompt_cost = prompt_price * Decimal(regular_prompt_tokens) / MILLION
        cached_cost = cached_price * Decimal(cached_prompt_tokens) / MILLION
        completion_cost = completion_price * Decimal(completion_tokens) / MILLION

        vendor_cost = prompt_cost + cached_cost + completion_cost + surcharge
        billed_cost = vendor_cost * markup

        return CostBreakdown(
            vendorCost=vendor_cost,
            billedCost=billed_cost,
            pricingId=pricing.id,
            currency=pricing.currency or "CNY",
        )
