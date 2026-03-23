"""Subscription API: status, start-trial, activate-premium."""
import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies.auth import get_current_user, limiter
from app.models.user import User
from app.schemas.subscription import (
    ActivatePremiumRequest,
    StartTrialRequest,
    SubscriptionStatusResponse,
    UsageTodayResponse,
)
from app.services.subscription_service import (
    FREE_MAX_CHATS_PER_DAY,
    FREE_MAX_VOICE_PER_DAY,
    get_usage_today_for_display,
    resolve_user_plan,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _now_utc() -> datetime:
    return datetime.utcnow()


@router.get("/status", response_model=SubscriptionStatusResponse)
@limiter.limit("30/minute")
async def subscription_status(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SubscriptionStatusResponse:
    """Return current resolved plan, expiration dates, and today's usage with limits."""
    plan = resolve_user_plan(current_user)
    usage_data = get_usage_today_for_display(current_user.id, db)
    chat_limit: int | None = FREE_MAX_CHATS_PER_DAY if plan == "free" else None
    voice_limit: int | None = FREE_MAX_VOICE_PER_DAY if plan == "free" else None
    usage = UsageTodayResponse(
        date=usage_data["date"],
        chat_count=usage_data["chat_count"],
        voice_count=usage_data["voice_count"],
        chat_limit=chat_limit,
        voice_limit=voice_limit,
        minutes_used=usage_data["minutes_used"],
    )
    return SubscriptionStatusResponse(
        plan=plan,
        trial_expires_at=current_user.trial_expires_at,
        subscription_expires_at=current_user.subscription_expires_at,
        usage=usage,
    )


@router.post("/start-trial")
@limiter.limit("10/minute")
async def start_trial(
    request: Request,
    body: StartTrialRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Activate trial after payment verification. Trial is 3 days; one per user."""
    if not body.payment_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payment must be verified to start trial",
        )
    if current_user.is_trial_used:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Trial already used",
        )
    now = _now_utc()
    current_user.plan = "trial"
    current_user.trial_expires_at = now + timedelta(days=3)
    current_user.is_trial_used = True
    db.commit()
    db.refresh(current_user)
    logger.info("User %s started trial", current_user.id)
    return {"message": "Trial started", "plan": "trial"}


@router.post("/activate-premium")
@limiter.limit("10/minute")
async def activate_premium(
    request: Request,
    body: ActivatePremiumRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Activate premium after Google Play / Apple purchase verification (placeholder)."""
    # Placeholder: no real IAP verification
    now = _now_utc()
    current_user.plan = "premium"
    current_user.subscription_expires_at = now + timedelta(days=30)
    db.commit()
    db.refresh(current_user)
    logger.info("User %s activated premium", current_user.id)
    return {"message": "Premium activated", "plan": "premium"}
