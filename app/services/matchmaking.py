"""Redis-backed FIFO social matchmaking with atomic pair dequeue."""
from __future__ import annotations

import logging
from typing import Any, Optional, TypedDict

from redis.asyncio import Redis
from sqlalchemy import or_
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.database import SessionLocal
from app.models.social_session import SocialSession
from app.services.agora import AgoraConfigError, generate_agora_token

logger = logging.getLogger(__name__)

SOCIAL_QUEUE_KEY = "social:queue"

# Atomically dequeue two distinct user ids (FIFO: RPOP = oldest). If duplicate ids, put back and retry.
_SOCIAL_PAIR_LUA = """
local key = KEYS[1]
local max_iter = 100
for i = 1, max_iter do
  local len = redis.call('LLEN', key)
  if len < 2 then
    return nil
  end
  local a = redis.call('RPOP', key)
  local b = redis.call('RPOP', key)
  if not a or not b then
    if a then redis.call('RPUSH', key, a) end
    if b then redis.call('RPUSH', key, b) end
    return nil
  end
  if a == b then
    redis.call('RPUSH', key, b)
    redis.call('RPUSH', key, a)
  else
    return {a, b}
  end
end
return nil
"""


class MatchedSessionInfo(TypedDict):
    id: str
    user1_id: str
    user2_id: str
    agora_channel: str


def agora_uid_for_user(user_id: str, session_id: str) -> int:
    uid = abs(hash(f"{user_id}:{session_id}")) % (2**31)
    if uid == 0:
        uid = 1
    return uid


def _persist_match_sync(user1_id: str, user2_id: str) -> Optional[MatchedSessionInfo]:
    import uuid
    from datetime import datetime

    session_id = str(uuid.uuid4())
    channel_name = f"session_{session_id}"
    db = SessionLocal()
    try:
        row = SocialSession(
            id=session_id,
            user1_id=user1_id,
            user2_id=user2_id,
            status="active",
            agora_channel=channel_name,
            started_at=datetime.utcnow(),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        logger.info(
            "social_match_persisted session_id=%s channel=%s user1=%s user2=%s",
            row.id,
            row.agora_channel,
            user1_id,
            user2_id,
        )
        return MatchedSessionInfo(
            id=row.id,
            user1_id=row.user1_id,
            user2_id=row.user2_id,
            agora_channel=row.agora_channel,
        )
    except Exception:
        logger.exception("social_match_persist_failed user1=%s user2=%s", user1_id, user2_id)
        db.rollback()
        return None
    finally:
        db.close()


def _delete_social_session_sync(session_id: str) -> None:
    db = SessionLocal()
    try:
        row = db.query(SocialSession).filter(SocialSession.id == session_id).first()
        if row:
            db.delete(row)
            db.commit()
    except Exception:
        logger.exception("social_session_delete_failed session_id=%s", session_id)
        db.rollback()
    finally:
        db.close()


def _active_session_info_sync(user_id: str) -> Optional[dict[str, str]]:
    db = SessionLocal()
    try:
        row = (
            db.query(SocialSession)
            .filter(
                SocialSession.status == "active",
                or_(SocialSession.user1_id == user_id, SocialSession.user2_id == user_id),
            )
            .first()
        )
        if not row:
            return None
        return {"session_id": row.id, "channel": row.agora_channel}
    finally:
        db.close()


async def clear_social_user_redis(redis: Redis, user_id: str) -> None:
    await redis.delete(f"social:lock:{user_id}", f"social:in_queue:{user_id}")


async def enqueue_for_match(redis: Redis, user_id: str) -> Optional[dict[str, Any]]:
    """
    Acquire lock, rate-limit, check DB for active session, then enqueue.
    Returns a client-bound dict to send over WebSocket, or None to proceed to try_match.
    """
    lock_key = f"social:lock:{user_id}"
    is_locked = await redis.set(lock_key, "1", ex=10, nx=True)
    if not is_locked:
        return {"status": "already_searching"}

    rate_key = f"social:rate:{user_id}"
    count = await redis.incr(rate_key)
    if count == 1:
        await redis.expire(rate_key, 600)
    if count > 5:
        await redis.delete(lock_key)
        return {"type": "RATE_LIMITED"}

    existing = await run_in_threadpool(_active_session_info_sync, user_id)
    if existing:
        await redis.delete(lock_key)
        return {
            "type": "ALREADY_IN_SESSION",
            "session_id": existing["session_id"],
            "channel": existing["channel"],
        }

    await redis.lrem(SOCIAL_QUEUE_KEY, 0, user_id)
    await redis.lpush(SOCIAL_QUEUE_KEY, user_id)
    await redis.set(f"social:in_queue:{user_id}", "1", ex=30)
    logger.info("social_queue_enqueue user_id=%s", user_id)
    return None


async def try_match(redis: Redis) -> Optional[MatchedSessionInfo]:
    """
    Atomically pop a distinct pair; drop stale users (no in_queue TTL key); persist session.
    On success clears lock + in_queue for both users.
    """
    raw: Any = await redis.eval(_SOCIAL_PAIR_LUA, 1, SOCIAL_QUEUE_KEY)
    if not raw:
        return None
    user1_id, user2_id = raw[0], raw[1]

    q1 = await redis.get(f"social:in_queue:{user1_id}")
    q2 = await redis.get(f"social:in_queue:{user2_id}")
    if not q1:
        logger.warning("[QUEUE] stale user skipped %s", user1_id)
    if not q2:
        logger.warning("[QUEUE] stale user skipped %s", user2_id)

    if not q1 and not q2:
        return None
    if not q1 and q2:
        await redis.lpush(SOCIAL_QUEUE_KEY, user2_id)
        return None
    if q1 and not q2:
        await redis.lpush(SOCIAL_QUEUE_KEY, user1_id)
        return None

    info = await run_in_threadpool(_persist_match_sync, user1_id, user2_id)
    if info is None:
        await redis.rpush(SOCIAL_QUEUE_KEY, user1_id, user2_id)
        return None

    await redis.delete(
        f"social:lock:{user1_id}",
        f"social:lock:{user2_id}",
        f"social:in_queue:{user1_id}",
        f"social:in_queue:{user2_id}",
    )
    return info


async def remove_from_queue(redis: Redis, user_id: str) -> None:
    removed = await redis.lrem(SOCIAL_QUEUE_KEY, 0, user_id)
    if removed:
        logger.info("social_queue_removed user_id=%s count=%s", user_id, removed)


async def notify_match(session: MatchedSessionInfo, manager: Any, redis: Redis) -> None:
    """Push MATCH_FOUND to both users; on failure delete session and requeue the other user."""
    u1 = session["user1_id"]
    u2 = session["user2_id"]
    sid = session["id"]
    channel = session["agora_channel"]

    uid1 = agora_uid_for_user(u1, sid)
    uid2 = agora_uid_for_user(u2, sid)

    try:
        token1 = generate_agora_token(channel, uid1)
    except AgoraConfigError:
        logger.error("[AGORA] token generation failed for %s", u1)
        await run_in_threadpool(_delete_social_session_sync, sid)
        await redis.lpush(SOCIAL_QUEUE_KEY, u2)
        await redis.lpush(SOCIAL_QUEUE_KEY, u1)
        return

    try:
        token2 = generate_agora_token(channel, uid2)
    except AgoraConfigError:
        logger.error("[AGORA] token generation failed for %s", u2)
        await run_in_threadpool(_delete_social_session_sync, sid)
        await redis.lpush(SOCIAL_QUEUE_KEY, u2)
        await redis.lpush(SOCIAL_QUEUE_KEY, u1)
        return

    msg1 = {
        "type": "MATCH_FOUND",
        "session_id": sid,
        "channel": channel,
        "token": token1,
        "uid": uid1,
        "peer_user_id": u2,
        "app_id": settings.agora_app_id,
    }
    msg2 = {
        "type": "MATCH_FOUND",
        "session_id": sid,
        "channel": channel,
        "token": token2,
        "uid": uid2,
        "peer_user_id": u1,
        "app_id": settings.agora_app_id,
    }

    ok1 = await manager.send_to_user(u1, msg1)
    if not ok1:
        await run_in_threadpool(_delete_social_session_sync, sid)
        await redis.lpush(SOCIAL_QUEUE_KEY, u2)
        return

    ok2 = await manager.send_to_user(u2, msg2)
    if not ok2:
        await run_in_threadpool(_delete_social_session_sync, sid)
        await redis.lpush(SOCIAL_QUEUE_KEY, u1)
        return

    logger.info("[MATCH] %s vs %s session=%s", u1, u2, sid)
