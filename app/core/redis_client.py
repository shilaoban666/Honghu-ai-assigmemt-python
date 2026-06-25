"""Redis client factory."""

from redis.asyncio import Redis

from app.core.config import settings

_redis: Redis | None = None


async def get_redis() -> Redis:
    """FastAPI dependency: yields an async Redis client."""
    global _redis
    if _redis is None:
        _redis = Redis(
            host=settings.redis.host,
            port=settings.redis.port,
            db=settings.redis.database,
            password=settings.redis.password or None,
            decode_responses=True,
        )
    return _redis


async def close_redis():
    global _redis
    if _redis:
        await _redis.close()
        _redis = None
