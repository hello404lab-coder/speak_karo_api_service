"""Gemini Live control plane: config and session tracking (no audio proxy)."""
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.live_prompts import build_live_system_instruction
from app.database import get_db
from app.dependencies.live_access import require_gemini_live_plan
from app.models.live_session import LiveSession
from app.models.usage import Conversation
from app.models.user import User
from app.schemas.live import (
    LiveConfigResponse,
    LiveSessionActiveResponse,
    LiveSessionEndRequest,
    LiveSessionEndResponse,
    LiveSessionHeartbeatRequest,
    LiveSessionHeartbeatResponse,
    LiveSessionStartRequest,
    LiveSessionStartResponse,
    LiveTokenRequest,
    LiveTokenResponse,
)
from app.services.live_gemini import mint_live_ephemeral_auth_token
from app.services.live_session_service import (
    end_live_session_for_user,
    get_active_live_session,
    is_session_token_stale,
    touch_live_session_heartbeat,
)
from app.services.redis_live import LiveTokenRateLimited, check_live_token_rate_limit
from app.services.subscription_service import (
    check_usage_limit,
    get_usage_today,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/live", tags=["Gemini Live"])

ONBOARDING_REQUIRED_MESSAGE = "User onboarding not completed"


def _require_onboarding(user: User) -> None:
    if not getattr(user, "onboarding_completed", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=ONBOARDING_REQUIRED_MESSAGE)


def _resolve_long_term_context(
    db: Session,
    user_id: str,
    conversation_id: str | None,
) -> str | None:
    if not conversation_id:
        return None
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user_id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    ctx = (conv.long_term_context or "").strip()
    return ctx or None


def _build_instruction_for_user(
    user: User,
    db: Session,
    conversation_id: str | None,
) -> str:
    long_term = _resolve_long_term_context(db, user.id, conversation_id)
    return build_live_system_instruction(user, long_term_context=long_term)


def _iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


async def _enforce_token_rate_limit(request: Request, user_id: str) -> None:
    redis = getattr(request.app.state, "live_redis", None)
    if redis is None:
        if settings.is_prod and settings.gemini_live_redis_required_for_token:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Live token rate limiting requires Redis",
            )
        return
    try:
        await check_live_token_rate_limit(redis, user_id)
    except LiveTokenRateLimited:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "type": "RATE_LIMITED",
                "message": "Too many token requests, try again later.",
            },
        ) from None


@router.get("/config", response_model=LiveConfigResponse)
async def get_live_config(
    conversation_id: str | None = Query(None, description="Optional conversation for learner context"),
    user: User = Depends(require_gemini_live_plan),
    db: Session = Depends(get_db),
):
    """Return safe Live client configuration (no API keys)."""
    _require_onboarding(user)
    system_instruction = _build_instruction_for_user(user, db, conversation_id)
    modalities = ["AUDIO"]
    return LiveConfigResponse(
        model=settings.gemini_live_model,
        system_instruction=system_instruction,
        prompt_version=settings.gemini_live_prompt_version,
        voice=settings.gemini_live_voice,
        language_code=settings.gemini_live_language_code,
        temperature=float(settings.gemini_live_temperature),
        response_modalities=modalities,
    )


@router.get("/session/active", response_model=LiveSessionActiveResponse)
async def get_active_live_session_endpoint(
    user: User = Depends(require_gemini_live_plan),
    db: Session = Depends(get_db),
):
    """Return whether the user has an open Live session (single-session model)."""
    _require_onboarding(user)
    row = get_active_live_session(db, user.id)
    if not row:
        return LiveSessionActiveResponse(active=False)
    return LiveSessionActiveResponse(
        active=True,
        session_id=row.id,
        started_at=_iso_z(row.started_at),
        last_seen_at=_iso_z(row.last_seen_at) if row.last_seen_at else _iso_z(row.started_at),
        conversation_id=row.conversation_id,
    )


@router.post("/session/start", response_model=LiveSessionStartResponse)
async def start_live_session(
    body: LiveSessionStartRequest,
    user: User = Depends(require_gemini_live_plan),
    db: Session = Depends(get_db),
):
    """Start or reuse the single active Live session for this user."""
    _require_onboarding(user)
    usage_today = get_usage_today(user.id, db)
    check_usage_limit(user, usage_today)

    if body.conversation_id:
        _resolve_long_term_context(db, user.id, body.conversation_id)

    existing = get_active_live_session(db, user.id)
    if existing:
        logger.info("live_session_reused user_id=%s session_id=%s", user.id, existing.id)
        return LiveSessionStartResponse(
            session_id=existing.id,
            started_at=_iso_z(existing.started_at),
            reused_existing=True,
        )

    now = datetime.now(timezone.utc)
    now_naive = now.replace(tzinfo=None)
    row = LiveSession(
        user_id=user.id,
        conversation_id=body.conversation_id,
        started_at=now_naive,
        last_seen_at=now_naive,
        status="active",
        prompt_version=settings.gemini_live_prompt_version,
        client_platform=(body.client_platform or None),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    logger.info("live_session_started user_id=%s session_id=%s", user.id, row.id)
    return LiveSessionStartResponse(
        session_id=row.id,
        started_at=now.isoformat().replace("+00:00", "Z"),
        reused_existing=False,
    )


@router.post("/session/heartbeat", response_model=LiveSessionHeartbeatResponse)
async def live_session_heartbeat(
    body: LiveSessionHeartbeatRequest,
    user: User = Depends(require_gemini_live_plan),
    db: Session = Depends(get_db),
):
    """Keep the session alive for the reaper (call every 20–30s while connected)."""
    _require_onboarding(user)
    row = touch_live_session_heartbeat(db, body.session_id, user.id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    ls = row.last_seen_at or row.started_at
    return LiveSessionHeartbeatResponse(session_id=row.id, last_seen_at=_iso_z(ls))


@router.post("/session/end", response_model=LiveSessionEndResponse)
async def end_live_session(
    body: LiveSessionEndRequest,
    user: User = Depends(require_gemini_live_plan),
    db: Session = Depends(get_db),
):
    """Close a Live session; idempotent if already ended (no double usage)."""
    _require_onboarding(user)
    out = end_live_session_for_user(db, session_id=body.session_id, user_id=user.id, source="client")
    if out is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    _idem, sid, duration_seconds, ended_iso = out
    return LiveSessionEndResponse(
        session_id=sid,
        duration_seconds=duration_seconds,
        ended_at=ended_iso,
    )


@router.post("/token", response_model=LiveTokenResponse)
async def create_live_token(
    request: Request,
    body: LiveTokenRequest,
    user: User = Depends(require_gemini_live_plan),
    db: Session = Depends(get_db),
):
    """
    Mint an ephemeral Gemini Live auth token for direct WebSocket connections.
    Requires GEMINI_API_KEY on the server. Rate-limited per user via Redis when available.
    """
    _require_onboarding(user)
    await _enforce_token_rate_limit(request, user.id)

    usage_today = get_usage_today(user.id, db)
    check_usage_limit(user, usage_today)

    conversation_id = None
    if body.session_id:
        sess = (
            db.query(LiveSession)
            .filter(LiveSession.id == body.session_id, LiveSession.user_id == user.id)
            .first()
        )
        if not sess:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        if sess.ended_at is not None:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail={
                    "type": "SESSION_ENDED",
                    "message": "This Live session is no longer active.",
                },
            )
        if is_session_token_stale(sess, settings.gemini_live_heartbeat_stale_seconds):
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail={
                    "type": "SESSION_STALE",
                    "message": "Live session heartbeat is stale; start a new session.",
                },
            )
        conversation_id = sess.conversation_id

    system_instruction = _build_instruction_for_user(user, db, conversation_id)
    try:
        token_name = await mint_live_ephemeral_auth_token(system_instruction)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.exception("Live auth token mint failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not create Live auth token",
        ) from e

    return LiveTokenResponse(
        auth_token=token_name,
        new_session_expire_seconds=int(settings.gemini_live_token_new_session_seconds),
    )
