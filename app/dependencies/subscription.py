"""Subscription dependency: require active plan and enforce usage limits."""
from fastapi import Depends

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.models.user import User
from app.services.subscription_service import (
    check_usage_limit,
    get_usage_today,
    resolve_user_plan,
)
from sqlalchemy.orm import Session


def require_active_plan(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> User:
    """
    Dependency that ensures the user is within usage limits. Raises 402 if free
    user has exceeded daily chat/voice limits. Returns the user otherwise.
    """
    usage_today = get_usage_today(current_user.id, db)
    check_usage_limit(current_user, usage_today)
    return current_user
