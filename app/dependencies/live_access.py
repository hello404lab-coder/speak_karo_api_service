"""Gemini Live: subscription tier gate (after require_active_plan)."""
from fastapi import Depends, HTTPException, status

from app.core.config import settings
from app.dependencies.subscription import require_active_plan
from app.models.user import User
from app.services.subscription_service import resolve_user_plan

_PLAN_RANK = {"free": 0, "trial": 1, "premium": 2}


def require_gemini_live_plan(
    user: User = Depends(require_active_plan),
) -> User:
    """
    Enforce GEMINI_LIVE_MIN_PLAN: user effective plan must rank >= configured minimum.
    Called after require_active_plan (daily free limits already checked).
    """
    effective = resolve_user_plan(user)
    minimum = settings.gemini_live_min_plan
    if _PLAN_RANK.get(effective, 0) < _PLAN_RANK.get(minimum, 0):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Gemini Live is not available on your current plan.",
        )
    return user
