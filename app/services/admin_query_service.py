"""Read-only admin queries for users, usage, and recent activity."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from sqlalchemy import case, func, literal, or_
from sqlalchemy.orm import Session

from app.models.usage import Conversation, Message, Usage
from app.models.user import User

UserSortBy = Literal["created_at", "last_activity_at", "email"]
SortOrder = Literal["asc", "desc"]


def _resolved_plan_expression(now: datetime):
    normalized_paid_plan = case(
        (func.lower(func.coalesce(User.plan, "")) == "vuvl_pro", literal("vuvl_pro")),
        else_=literal("vuvl_plus"),
    )
    return case(
        (
            (User.subscription_expires_at.is_not(None)) & (User.subscription_expires_at > now),
            normalized_paid_plan,
        ),
        (
            (User.trial_expires_at.is_not(None)) & (User.trial_expires_at > now),
            literal("trial"),
        ),
        else_=literal("free"),
    )


def _usage_totals_subquery(db: Session):
    return (
        db.query(
            Usage.user_id.label("user_id"),
            func.coalesce(func.sum(Usage.request_count), 0).label("total_request_count"),
            func.coalesce(func.sum(Usage.chat_count), 0).label("total_chat_count"),
            func.coalesce(func.sum(Usage.voice_count), 0).label("total_voice_count"),
            func.coalesce(func.sum(Usage.minutes_used), 0.0).label("total_minutes_used"),
        )
        .group_by(Usage.user_id)
        .subquery()
    )


def _last_activity_subquery(db: Session):
    return (
        db.query(
            Conversation.user_id.label("user_id"),
            func.max(Conversation.updated_at).label("last_activity_at"),
        )
        .group_by(Conversation.user_id)
        .subquery()
    )


def list_users_for_admin(
    db: Session,
    *,
    page: int,
    page_size: int,
    search: str | None = None,
    provider: str | None = None,
    plan: str | None = None,
    onboarding_completed: bool | None = None,
    sort_by: UserSortBy = "created_at",
    sort_order: SortOrder = "desc",
) -> dict:
    """Return paginated users with usage aggregates for the admin panel."""
    now = datetime.utcnow()
    offset = (page - 1) * page_size
    resolved_plan = _resolved_plan_expression(now).label("resolved_plan")
    usage_totals = _usage_totals_subquery(db)
    last_activity = _last_activity_subquery(db)

    query = (
        db.query(
            User.id,
            User.email,
            User.name,
            User.nickname,
            User.provider,
            User.onboarding_completed,
            User.onboarding_step,
            User.created_at,
            User.updated_at,
            resolved_plan,
            last_activity.c.last_activity_at,
            func.coalesce(usage_totals.c.total_request_count, 0).label("total_request_count"),
            func.coalesce(usage_totals.c.total_chat_count, 0).label("total_chat_count"),
            func.coalesce(usage_totals.c.total_voice_count, 0).label("total_voice_count"),
            func.coalesce(usage_totals.c.total_minutes_used, 0.0).label("total_minutes_used"),
        )
        .outerjoin(usage_totals, usage_totals.c.user_id == User.id)
        .outerjoin(last_activity, last_activity.c.user_id == User.id)
    )

    if search and search.strip():
        search_term = f"%{search.strip()}%"
        query = query.filter(
            or_(
                User.email.ilike(search_term),
                User.name.ilike(search_term),
                User.nickname.ilike(search_term),
            )
        )
    if provider:
        query = query.filter(User.provider == provider)
    if onboarding_completed is not None:
        query = query.filter(User.onboarding_completed == onboarding_completed)
    if plan:
        query = query.filter(resolved_plan == plan)

    total = query.order_by(None).count()

    sort_column = {
        "created_at": User.created_at,
        "last_activity_at": last_activity.c.last_activity_at,
        "email": User.email,
    }[sort_by]
    if sort_order == "asc":
        query = query.order_by(sort_column.is_(None), sort_column.asc(), User.id.asc())
    else:
        query = query.order_by(sort_column.is_(None), sort_column.desc(), User.id.desc())

    rows = query.offset(offset).limit(page_size).all()
    items = [
        {
            "id": row.id,
            "email": row.email,
            "name": row.name,
            "nickname": row.nickname,
            "provider": row.provider,
            "onboarding_completed": row.onboarding_completed,
            "onboarding_step": row.onboarding_step,
            "plan": row.resolved_plan,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "last_activity_at": row.last_activity_at,
            "total_request_count": int(row.total_request_count or 0),
            "total_chat_count": int(row.total_chat_count or 0),
            "total_voice_count": int(row.total_voice_count or 0),
            "total_minutes_used": float(row.total_minutes_used or 0.0),
        }
        for row in rows
    ]
    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": items,
    }


def get_user_detail_for_admin(db: Session, user_id: str) -> dict | None:
    """Return a complete admin detail payload for one user."""
    now = datetime.utcnow()
    today = date.today()
    usage_totals = _usage_totals_subquery(db)
    last_activity = _last_activity_subquery(db)
    resolved_plan = _resolved_plan_expression(now).label("resolved_plan")

    row = (
        db.query(
            User,
            resolved_plan,
            last_activity.c.last_activity_at,
            func.coalesce(usage_totals.c.total_request_count, 0).label("total_request_count"),
            func.coalesce(usage_totals.c.total_chat_count, 0).label("total_chat_count"),
            func.coalesce(usage_totals.c.total_voice_count, 0).label("total_voice_count"),
            func.coalesce(usage_totals.c.total_minutes_used, 0.0).label("total_minutes_used"),
        )
        .outerjoin(usage_totals, usage_totals.c.user_id == User.id)
        .outerjoin(last_activity, last_activity.c.user_id == User.id)
        .filter(User.id == user_id)
        .first()
    )
    if not row:
        return None

    user = row[0]
    today_usage = (
        db.query(Usage)
        .filter(Usage.user_id == user_id, Usage.date == today)
        .first()
    )

    usage_history_rows = (
        db.query(Usage)
        .filter(Usage.user_id == user_id)
        .order_by(Usage.date.desc())
        .limit(30)
        .all()
    )
    usage_history = [
        {
            "date": usage.date.isoformat(),
            "request_count": int(usage.request_count or 0),
            "chat_count": int(usage.chat_count or 0),
            "voice_count": int(usage.voice_count or 0),
            "minutes_used": float(usage.minutes_used or 0.0),
        }
        for usage in reversed(usage_history_rows)
    ]

    message_counts = (
        db.query(
            Message.conversation_id.label("conversation_id"),
            func.count(Message.id).label("message_count"),
        )
        .group_by(Message.conversation_id)
        .subquery()
    )
    conversations = (
        db.query(
            Conversation.id,
            Conversation.title,
            Conversation.created_at,
            Conversation.updated_at,
            func.coalesce(message_counts.c.message_count, 0).label("message_count"),
        )
        .outerjoin(message_counts, message_counts.c.conversation_id == Conversation.id)
        .filter(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        .limit(10)
        .all()
    )
    conversation_ids = [conversation.id for conversation in conversations]
    latest_message_by_conversation: dict[str, Message] = {}
    if conversation_ids:
        latest_messages = (
            db.query(Message)
            .filter(Message.conversation_id.in_(conversation_ids))
            .order_by(Message.conversation_id.asc(), Message.created_at.desc(), Message.id.desc())
            .all()
        )
        for message in latest_messages:
            latest_message_by_conversation.setdefault(message.conversation_id, message)

    recent_activity = [
        {
            "id": conversation.id,
            "title": conversation.title,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "message_count": int(conversation.message_count or 0),
            "last_user_message": (
                latest_message_by_conversation[conversation.id].user_message
                if conversation.id in latest_message_by_conversation
                else None
            ),
            "last_ai_reply": (
                latest_message_by_conversation[conversation.id].ai_reply
                if conversation.id in latest_message_by_conversation
                else None
            ),
        }
        for conversation in conversations
    ]

    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "provider": user.provider,
        "nickname": user.nickname,
        "native_language": user.native_language,
        "native_language_code": user.native_language_code,
        "student_type": user.student_type,
        "occupation": user.occupation,
        "goal": user.goal,
        "english_level": user.english_level,
        "onboarding_completed": user.onboarding_completed,
        "onboarding_step": user.onboarding_step,
        "plan": row.resolved_plan,
        "trial_expires_at": user.trial_expires_at,
        "subscription_expires_at": user.subscription_expires_at,
        "is_trial_used": user.is_trial_used,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
        "last_activity_at": row.last_activity_at,
        "usage_summary": {
            "today": {
                "date": today.isoformat(),
                "request_count": int(getattr(today_usage, "request_count", 0) or 0),
                "chat_count": int(getattr(today_usage, "chat_count", 0) or 0),
                "voice_count": int(getattr(today_usage, "voice_count", 0) or 0),
                "minutes_used": float(getattr(today_usage, "minutes_used", 0.0) or 0.0),
            },
            "totals": {
                "request_count": int(row.total_request_count or 0),
                "chat_count": int(row.total_chat_count or 0),
                "voice_count": int(row.total_voice_count or 0),
                "minutes_used": float(row.total_minutes_used or 0.0),
            },
            "last_activity_at": row.last_activity_at,
        },
        "usage_history": usage_history,
        "recent_activity": recent_activity,
    }
