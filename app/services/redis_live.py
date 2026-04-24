"""Async Redis helpers for Gemini Live (token rate limiting)."""
from redis.asyncio import Redis

from app.core.config import settings


class LiveTokenRateLimited(Exception):
    """Raised when per-user /live/token budget is exceeded."""


def create_live_redis() -> Redis:
    """Dedicated async Redis client (same URL as social/cache; distinct connection pool)."""
    return Redis.from_url(settings.redis_url, decode_responses=True)


TOKEN_RATE_KEY_PREFIX = "live:token_rate:"


async def check_live_token_rate_limit(redis: Redis, user_id: str) -> None:
    """
    Increment per-user token mint counter. Raises LiveTokenRateLimited if over budget.

    Key: live:token_rate:{user_id}, TTL = window seconds, max = settings.gemini_live_token_requests_per_minute.
    """
    window = max(1, int(settings.gemini_live_token_rate_window_seconds or 60))
    max_req = max(1, int(settings.gemini_live_token_requests_per_minute or 3))
    key = f"{TOKEN_RATE_KEY_PREFIX}{user_id}"
    n = await redis.incr(key)
    if n == 1:
        await redis.expire(key, window)
    if n > max_req:
        raise LiveTokenRateLimited()
