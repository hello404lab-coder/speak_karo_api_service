"""Social voice matchmaking: WebSocket queue + Agora tokens; REST end-session."""
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, status
from sqlalchemy import or_
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocketDisconnect

from app.core.config import settings
from app.core.security import ACCESS_TOKEN_TYPE, verify_token
from app.database import SessionLocal, get_db
from app.dependencies.subscription import require_active_plan
from app.models.social_session import SocialSession
from app.models.user import User
from app.services.agora import AgoraConfigError, generate_agora_token
from app.services.matchmaking import (
    agora_uid_for_user,
    clear_social_user_redis,
    enqueue_for_match,
    notify_match,
    remove_from_queue,
    try_match,
)
from app.services.subscription_service import check_usage_limit, get_or_create_today_usage, get_usage_today
from pydantic import BaseModel, Field

from app.websocket.social_ws import manager

logger = logging.getLogger(__name__)

router = APIRouter()

ONBOARDING_REQUIRED_MESSAGE = "User onboarding not completed"


class EndSocialSessionRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


def _load_user_for_ws(user_id: str) -> Optional[User]:
    db = SessionLocal()
    try:
        return db.query(User).filter(User.id == user_id).first()
    finally:
        db.close()


def _subscription_gate(user: User) -> None:
    db = SessionLocal()
    try:
        usage = get_usage_today(user.id, db)
        check_usage_limit(user, usage)
    finally:
        db.close()


@router.get("/active-session")
def get_active_session(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_active_plan),
) -> dict:
    session = (
        db.query(SocialSession)
        .filter(
            SocialSession.status == "active",
            or_(SocialSession.user1_id == current_user.id, SocialSession.user2_id == current_user.id),
        )
        .first()
    )
    if not session:
        return {"active": False}

    uid = agora_uid_for_user(current_user.id, session.id)
    try:
        token = generate_agora_token(session.agora_channel, uid)
    except AgoraConfigError as e:
        logger.error("[AGORA] token generation failed for %s", current_user.id)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)) from e

    return {
        "active": True,
        "session_id": session.id,
        "channel": session.agora_channel,
        "token": token,
        "uid": uid,
        "app_id": settings.agora_app_id,
    }


@router.post("/end-session")
def end_session(
    body: EndSocialSessionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_active_plan),
) -> dict[str, bool]:
    session = db.query(SocialSession).filter(SocialSession.id == body.session_id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if current_user.id not in (session.user1_id, session.user2_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a participant in this session")
    if session.status != "ended":
        session.status = "ended"
        session.ended_at = datetime.utcnow()
        if session.started_at:
            duration_seconds = (session.ended_at - session.started_at).total_seconds()
            duration_minutes = duration_seconds / 60.0
            usage = get_or_create_today_usage(current_user.id, db)
            usage.minutes_used = float(usage.minutes_used or 0) + duration_minutes
            usage.voice_count = int(getattr(usage, "voice_count", 0) or 0) + 1
        db.commit()
    return {"success": True}


@router.websocket("/ws")
async def social_ws(websocket: WebSocket) -> None:
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008, reason="missing_token")
        return

    user_id = verify_token(token, ACCESS_TOKEN_TYPE)
    if not user_id:
        await websocket.close(code=1008, reason="invalid_token")
        return

    user = await run_in_threadpool(_load_user_for_ws, user_id)
    if not user:
        await websocket.close(code=1008, reason="user_not_found")
        return

    await websocket.accept()

    if not user.onboarding_completed:
        await websocket.send_json(
            {
                "type": "ERROR",
                "code": "onboarding_required",
                "message": ONBOARDING_REQUIRED_MESSAGE,
            }
        )
        await websocket.close(code=1008)
        return

    try:
        await run_in_threadpool(_subscription_gate, user)
    except HTTPException as e:
        if e.status_code == 402 and isinstance(e.detail, dict):
            await websocket.send_json(
                {
                    "type": "ERROR",
                    "code": e.detail.get("error", "subscription_required"),
                    "message": e.detail.get("message", "Subscription required"),
                }
            )
        else:
            await websocket.send_json(
                {
                    "type": "ERROR",
                    "code": "forbidden",
                    "message": str(e.detail) if e.detail else "Forbidden",
                }
            )
        await websocket.close(code=1008)
        return

    redis = getattr(websocket.app.state, "social_redis", None)
    if redis is None:
        await websocket.send_json(
            {
                "type": "ERROR",
                "code": "redis_unavailable",
                "message": "Matchmaking service unavailable (Redis not connected)",
            }
        )
        await websocket.close(code=1011)
        return

    manager.register(user.id, websocket)
    logger.info("social_ws_connected user_id=%s", user.id)

    try:
        while True:
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                raise
            except Exception:
                await websocket.send_json(
                    {"type": "ERROR", "code": "invalid_message", "message": "Expected JSON object"}
                )
                continue

            if not isinstance(data, dict):
                await websocket.send_json(
                    {"type": "ERROR", "code": "invalid_message", "message": "Expected JSON object"}
                )
                continue

            mtype = data.get("type")
            if mtype == "PING":
                await websocket.send_json({"type": "PONG"})
                continue

            if mtype == "FIND_MATCH":
                try:
                    out = await enqueue_for_match(redis, user.id)
                    if out is not None:
                        await websocket.send_json(out)
                        continue
                    matched = await try_match(redis)
                    if matched:
                        await notify_match(matched, manager, redis)
                    else:
                        await websocket.send_json({"type": "QUEUED"})
                except Exception:
                    logger.exception("social_find_match_failed user_id=%s", user.id)
                    await websocket.send_json(
                        {
                            "type": "ERROR",
                            "code": "matchmaking_error",
                            "message": "Matchmaking failed; try again later",
                        }
                    )
                continue

            await websocket.send_json(
                {
                    "type": "ERROR",
                    "code": "unknown_type",
                    "message": f"Unknown message type: {mtype!r}",
                }
            )
    except WebSocketDisconnect:
        logger.info("social_ws_disconnect user_id=%s", user.id)
    finally:
        manager.disconnect(user.id)
        try:
            await clear_social_user_redis(redis, user.id)
            await remove_from_queue(redis, user.id)
        except Exception:
            logger.exception("social_queue_cleanup_failed user_id=%s", user.id)
