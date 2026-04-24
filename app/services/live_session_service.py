"""Sync helpers for Gemini Live session lifecycle (DB + usage)."""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.live_session import LiveSession
from app.services.subscription_service import finalize_live_session_usage

logger = logging.getLogger(__name__)

STATUS_ACTIVE = "active"
STATUS_ENDED = "ended"
STATUS_AUTO_ENDED = "auto_ended"


def get_active_live_session(db: Session, user_id: str) -> Optional[LiveSession]:
    """One open session per user (ended_at is null and status active)."""
    return (
        db.query(LiveSession)
        .filter(
            LiveSession.user_id == user_id,
            LiveSession.ended_at.is_(None),
            LiveSession.status == STATUS_ACTIVE,
        )
        .first()
    )


def _utc_naive_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _aware_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def compute_duration_seconds(started: datetime, end_utc: datetime, cap: float) -> float:
    started_aware = _aware_utc(started)
    raw = (end_utc - started_aware).total_seconds()
    return min(max(0.0, raw), max(0.0, float(cap)))


def end_live_session_for_user(
    db: Session,
    *,
    session_id: str,
    user_id: str,
    source: str,
) -> Optional[Tuple[bool, str, float, str]]:
    """
    Close a session for user_id or return idempotent payload if already closed.

    Returns None if session not found for this user.
    Otherwise (was_idempotent, session_id, duration_seconds, ended_at_iso_z).
    """
    row = (
        db.query(LiveSession)
        .filter(LiveSession.id == session_id, LiveSession.user_id == user_id)
        .first()
    )
    if not row:
        return None

    if row.ended_at is not None:
        ended_aware = _aware_utc(row.ended_at)
        dur = float(row.duration_seconds or 0.0)
        return (
            True,
            row.id,
            dur,
            ended_aware.isoformat().replace("+00:00", "Z"),
        )

    now_utc = datetime.now(timezone.utc)
    cap = float(max(0, settings.gemini_live_max_session_duration_seconds))
    duration_seconds = compute_duration_seconds(row.started_at, now_utc, cap)
    ended_naive = now_utc.replace(tzinfo=None)

    row.ended_at = ended_naive
    row.duration_seconds = duration_seconds
    row.status = STATUS_ENDED if source == "client" else STATUS_AUTO_ENDED
    row.last_seen_at = ended_naive
    db.add(row)
    db.commit()

    finalize_live_session_usage(user_id, db, duration_seconds, increment_request_count=False)
    logger.info(
        "live_session_closed user_id=%s session_id=%s source=%s duration_seconds=%s",
        user_id,
        row.id,
        source,
        duration_seconds,
    )
    return (
        False,
        row.id,
        duration_seconds,
        now_utc.isoformat().replace("+00:00", "Z"),
    )


def touch_live_session_heartbeat(db: Session, session_id: str, user_id: str) -> Optional[LiveSession]:
    """Update last_seen_at for an active session. Returns None if not found or not active."""
    row = (
        db.query(LiveSession)
        .filter(
            LiveSession.id == session_id,
            LiveSession.user_id == user_id,
            LiveSession.ended_at.is_(None),
            LiveSession.status == STATUS_ACTIVE,
        )
        .first()
    )
    if not row:
        return None
    row.last_seen_at = _utc_naive_now()
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def is_session_token_stale(row: LiveSession, stale_seconds: int) -> bool:
    """True if heartbeat is too old to mint tokens (open session only)."""
    now = datetime.now(timezone.utc)
    thresh = now - timedelta(seconds=max(1, stale_seconds))
    thresh_naive = thresh.replace(tzinfo=None)
    ref = row.last_seen_at or row.started_at
    if ref.tzinfo is not None:
        ref = ref.astimezone(timezone.utc).replace(tzinfo=None)
    return ref < thresh_naive


def reap_stale_live_sessions(db: Session, *, stale_seconds: int, batch: int = 50) -> int:
    """
    Auto-end active sessions with no recent heartbeat. Returns number of sessions closed.
    """
    thresh_naive = (datetime.now(timezone.utc) - timedelta(seconds=max(1, stale_seconds))).replace(
        tzinfo=None
    )
    cap = float(max(0, settings.gemini_live_max_session_duration_seconds))

    candidates = (
        db.query(LiveSession)
        .filter(
            LiveSession.ended_at.is_(None),
            LiveSession.status == STATUS_ACTIVE,
            or_(
                LiveSession.last_seen_at < thresh_naive,
                and_(LiveSession.last_seen_at.is_(None), LiveSession.started_at < thresh_naive),
            ),
        )
        .limit(batch)
        .all()
    )

    closed = 0
    for row in candidates:
        res = end_live_session_for_user(db, session_id=row.id, user_id=row.user_id, source="auto")
        if res is not None and not res[0]:
            closed += 1
    return closed
