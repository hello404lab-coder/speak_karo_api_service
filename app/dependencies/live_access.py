"""Gemini Live: subscription tier gate (after require_active_plan)."""
from fastapi import Depends, HTTPException, status

from app.core.config import settings
from app.dependencies.subscription import require_active_plan
from app.models.user import User
from app.services.subscription_service import plan_rank, resolve_user_plan


def require_gemini_live_plan(
    user: User = Depends(require_active_plan),
) -> User:
    """
    Enforce GEMINI_LIVE_MIN_PLAN: user effective plan must rank >= configured minimum.
    Called after require_active_plan (daily free limits already checked).
    """
    effective = resolve_user_plan(user)
    minimum = settings.gemini_live_min_plan
    if plan_rank(effective) < plan_rank(minimum):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Gemini Live is not available on your current plan.",
        )
    return user
