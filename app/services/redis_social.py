"""Async Redis client factory for social matchmaking (separate from sync cache client)."""
from redis.asyncio import Redis

from app.core.config import settings


def create_social_redis() -> Redis:
    return Redis.from_url(settings.redis_url, decode_responses=True)
