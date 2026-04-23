"""Subscription and usage limit logic."""
import logging
from datetime import date, datetime, timedelta
from typing import Any, Literal

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.usage import Usage
from app.models.user import User

logger = logging.getLogger(__name__)

SUBSCRIPTION_REQUIRED_DETAIL = "subscription_required"
SUBSCRIPTION_REQUIRED_MESSAGE = "Upgrade to continue unlimited practice"

FREE_MAX_CHATS_PER_DAY = 5
FREE_MAX_VOICE_PER_DAY = 1
FREE_PLAN = "free"
TRIAL_PLAN = "trial"
PLUS_PLAN = "vuvl_plus"
PRO_PLAN = "vuvl_pro"
LEGACY_PREMIUM_PLAN = "premium"
PAID_PLAN_CODES = (PLUS_PLAN, PRO_PLAN)
ALL_PLAN_CODES = (FREE_PLAN, TRIAL_PLAN, PLUS_PLAN, PRO_PLAN)
PLAN_RANK = {
    FREE_PLAN: 0,
    TRIAL_PLAN: 1,
    PLUS_PLAN: 2,
    PRO_PLAN: 3,
}
PLAN_CATALOG: dict[str, dict[str, Any]] = {
    PLUS_PLAN: {
        "code": PLUS_PLAN,
        "name": "VUVL Plus",
        "amount": 19900,
        "currency": "INR",
        "interval": "monthly",
        "feature_flags": {
            "unlimited_practice": True,
            "gemini_live_access": False,
        },
    },
    PRO_PLAN: {
        "code": PRO_PLAN,
        "name": "VUVL Pro",
        "amount": 49900,
        "currency": "INR",
        "interval": "monthly",
        "feature_flags": {
            "unlimited_practice": True,
            "gemini_live_access": True,
        },
    },
}


def normalize_plan_code(plan: str | None) -> str:
    """Normalize persisted/legacy plan values into current public codes."""
    if not plan:
        return FREE_PLAN
    lowered = str(plan).strip().lower()
    if lowered == LEGACY_PREMIUM_PLAN:
        return PLUS_PLAN
    if lowered in ALL_PLAN_CODES:
        return lowered
    return FREE_PLAN


def normalize_paid_plan_code(plan: str | None) -> str:
    """Normalize a paid plan value, defaulting old/unknown paid rows to Plus."""
    normalized = normalize_plan_code(plan)
    if normalized in PAID_PLAN_CODES:
        return normalized
    return PLUS_PLAN


def is_paid_plan(plan: str | None) -> bool:
    """Return True when a plan represents an active paid subscription tier."""
    return normalize_plan_code(plan) in PAID_PLAN_CODES


def plan_rank(plan: str | None) -> int:
    """Resolve plan ordering for authorization gates."""
    return PLAN_RANK.get(normalize_plan_code(plan), 0)


def resolve_user_plan(user: User) -> str:
    """
    Resolve effective plan from user fields. Paid tiers take precedence over trial;
    expired subscriptions yield free.
    """
    now = datetime.utcnow()
    if user.subscription_expires_at and user.subscription_expires_at > now:
        return normalize_paid_plan_code(getattr(user, "plan", None))
    if user.trial_expires_at and user.trial_expires_at > now:
        return TRIAL_PLAN
    return FREE_PLAN


def get_or_create_today_usage(user_id: str, db: Session) -> Usage:
    """Return today's Usage row, creating it if missing (caller commits)."""
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
    return usage


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
    Raise 402 if free user has exceeded daily limits. Trial and paid plans pass.
    """
    plan = resolve_user_plan(user)
    if plan != FREE_PLAN:
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


def finalize_live_session_usage(
    user_id: str,
    db: Session,
    duration_seconds: float,
    *,
    increment_request_count: bool = False,
) -> None:
    """
    Record a completed Gemini Live session on today's Usage row: add minutes and one voice_count.
    Does not increment chat_count. request_count is optional (default off to avoid inflating vs REST).
    """
    usage = get_or_create_today_usage(user_id, db)
    usage.minutes_used += max(0.0, float(duration_seconds)) / 60.0
    usage.voice_count += 1
    if increment_request_count:
        usage.request_count += 1
    db.commit()
