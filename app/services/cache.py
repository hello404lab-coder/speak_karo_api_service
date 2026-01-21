"""Redis caching service."""
import json
import logging
from typing import Optional
import redis
from app.core.config import settings

logger = logging.getLogger(__name__)

# Redis client (with graceful degradation)
try:
    redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    redis_client.ping()  # Test connection
    redis_available = True
except Exception as e:
    logger.warning(f"Redis not available: {e}. Continuing without cache.")
    redis_client = None
    redis_available = False


def get(key: str) -> Optional[str]:
    """
    Get value from cache.
    
    Args:
        key: Cache key
    
    Returns:
        Cached value or None
    """
    if not redis_available or not redis_client:
        return None
    
    try:
        return redis_client.get(key)
    except Exception as e:
        logger.error(f"Cache get error: {e}")
        return None


def set(key: str, value: str, ttl: int) -> None:
    """
    Set value in cache with TTL.
    
    Args:
        key: Cache key
        value: Value to cache
        ttl: Time to live in seconds
    """
    if not redis_available or not redis_client:
        return
    
    try:
        redis_client.setex(key, ttl, value)
    except Exception as e:
        logger.error(f"Cache set error: {e}")


def get_json(key: str) -> Optional[dict]:
    """Get JSON value from cache."""
    value = get(key)
    if value:
        try:
            return json.loads(value)
        except:
            return None
    return None


def set_json(key: str, value: dict, ttl: int) -> None:
    """Set JSON value in cache."""
    set(key, json.dumps(value), ttl)
