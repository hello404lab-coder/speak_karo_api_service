"""Subscription and usage limit logic."""
import logging
from datetime import date, datetime, timedelta
from typing import Literal

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.usage import Usage
from app.models.user import User

logger = logging.getLogger(__name__)

SUBSCRIPTION_REQUIRED_DETAIL = "subscription_required"
SUBSCRIPTION_REQUIRED_MESSAGE = "Upgrade to continue unlimited practice"

FREE_MAX_CHATS_PER_DAY = 5
FREE_MAX_VOICE_PER_DAY = 1


def resolve_user_plan(user: User) -> str:
    """
    Resolve effective plan from user fields. Premium takes precedence over trial;
    expired subscriptions yield free.
    """
    now = datetime.utcnow()
    if user.subscription_expires_at and user.subscription_expires_at > now:
        return "premium"
    if user.trial_expires_at and user.trial_expires_at > now:
        return "trial"
    return "free"


def get_usage_today(user_id: str, db: Session) -> dict[str, int]:
    """Return today's chat_count and voice_count for the user (0 if no row)."""
    today = date.today()
    row = (
        db.query(Usage)
        .filter(Usage.user_id == user_id, Usage.date == today)
        .first()
    )
    if not row:
        return {"chat_count": 0, "voice_count": 0}
    return {
        "chat_count": getattr(row, "chat_count", 0) or 0,
        "voice_count": getattr(row, "voice_count", 0) or 0,
    }


def get_usage_today_for_display(user_id: str, db: Session) -> dict[str, int | float | str]:
    """
    Return today's usage for display: date (ISO), chat_count, voice_count, minutes_used.
    Single query; use 0 / 0.0 and today's date when no row exists.
    """
    today = date.today()
    row = (
        db.query(Usage)
        .filter(Usage.user_id == user_id, Usage.date == today)
        .first()
    )
    if not row:
        return {
            "date": today.isoformat(),
            "chat_count": 0,
            "voice_count": 0,
            "minutes_used": 0.0,
        }
    return {
        "date": row.date.isoformat(),
        "chat_count": getattr(row, "chat_count", 0) or 0,
        "voice_count": getattr(row, "voice_count", 0) or 0,
        "minutes_used": float(getattr(row, "minutes_used", 0.0) or 0.0),
    }


def check_usage_limit(user: User, usage_today: dict[str, int]) -> None:
    """
    Raise 402 if free user has exceeded daily limits. Trial and premium pass.
    """
    plan = resolve_user_plan(user)
    if plan in ("trial", "premium"):
        return
    chat_count = usage_today.get("chat_count", 0)
    voice_count = usage_today.get("voice_count", 0)
    if chat_count >= FREE_MAX_CHATS_PER_DAY or voice_count >= FREE_MAX_VOICE_PER_DAY:
        logger.info("User %s usage limit reached", user.id)
        raise HTTPException(
            status_code=402,
            detail={
                "error": SUBSCRIPTION_REQUIRED_DETAIL,
                "message": SUBSCRIPTION_REQUIRED_MESSAGE,
            },
            headers={"X-Error-Code": SUBSCRIPTION_REQUIRED_DETAIL},
        )


def update_usage_stats(
    user_id: str,
    db: Session,
    duration_seconds: float = 0.0,
    usage_type: Literal["chat", "voice"] = "chat",
) -> None:
    """Increment daily usage: request_count and either chat_count or voice_count."""
    today = date.today()
    usage = (
        db.query(Usage)
        .filter(Usage.user_id == user_id, Usage.date == today)
        .first()
    )
    if not usage:
        usage = Usage(
            user_id=user_id,
            date=today,
            minutes_used=0.0,
            request_count=0,
            chat_count=0,
            voice_count=0,
        )
        db.add(usage)
    usage.request_count += 1
    usage.minutes_used += duration_seconds / 60.0
    if usage_type == "chat":
        usage.chat_count += 1
    else:
        usage.voice_count += 1
    db.commit()
