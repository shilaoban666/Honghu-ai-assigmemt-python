"""Redis-based short-term chat memory — mirrors ChatMemoryService.java.

Three-layer memory architecture:
1. Redis token sliding window (short-term)
2. Summary compression (mid-term session summary + user profile)
3. PostgreSQL full history (long-term)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.orm import ChatMessage

# Lua script for atomic append-if-absent (mirrors APPEND_IF_ABSENT_SCRIPT)
APPEND_IF_ABSENT_SCRIPT = """
if redis.call('SADD', KEYS[1], ARGV[1]) == 1 then
    redis.call('RPUSH', KEYS[2], ARGV[2]);
    if tonumber(ARGV[3]) > 0 then
        redis.call('EXPIRE', KEYS[1], ARGV[3]);
        redis.call('EXPIRE', KEYS[2], ARGV[3]);
    end;
    return 1;
else
    if tonumber(ARGV[3]) > 0 then
        redis.call('EXPIRE', KEYS[1], ARGV[3]);
        redis.call('EXPIRE', KEYS[2], ARGV[3]);
    end;
    return 0;
end
"""

RELEASE_LOCK_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1]);
else return 0; end
"""


class ChatMemoryService:
    """Manages short-term chat memory in Redis."""

    def __init__(self, redis: Redis):
        self.redis = redis

    def _build_key(self, session_id: str) -> str:
        return f"{settings.memory.redis_key_prefix}{session_id}"

    def _build_dedup_key(self, session_id: str) -> str:
        return f"{self._build_key(session_id)}:dedup"

    def _build_rebuild_lock_key(self, session_id: str) -> str:
        return f"{self._build_key(session_id)}:rebuild:lock"

    def _ttl_seconds(self) -> int:
        return max(0, settings.memory.redis_ttl_minutes * 60)

    async def append_message(self, message: ChatMessage) -> None:
        """Append a single message to Redis short-term memory."""
        if not message or not message.session_id or not message.content:
            return

        key = self._build_key(message.session_id)
        dedup_key = self._build_dedup_key(message.session_id)

        item = {
            "chatId": message.chat_id,
            "sessionId": message.session_id,
            "chatRole": message.chat_role,
            "contentType": message.content_type,
            "status": message.status,
            "content": message.content,
            "createdAt": message.created_at.isoformat() if message.created_at else datetime.utcnow().isoformat(),
        }
        payload = json.dumps(item, ensure_ascii=False)
        identity = str(message.chat_id) if message.chat_id else f"{message.session_id}:{message.chat_role}:{message.created_at}"

        try:
            await self.redis.eval(
                APPEND_IF_ABSENT_SCRIPT,
                2,
                dedup_key,
                key,
                identity,
                payload,
                str(self._ttl_seconds()),
            )
        except Exception:
            pass  # Redis is best-effort for short-term memory

    async def get_messages(self, session_id: str) -> list[dict]:
        """Read all messages for a session from Redis."""
        if not session_id:
            return []

        key = self._build_key(session_id)
        try:
            payloads = await self.redis.lrange(key, 0, -1)
            if not payloads:
                return []

            # Refresh TTL on access (sliding window)
            await self._refresh_expiration(session_id)

            history = []
            for p in payloads:
                if not p:
                    continue
                try:
                    item = json.loads(p)
                    history.append(item)
                except json.JSONDecodeError:
                    continue
            history.sort(key=lambda m: m.get("createdAt", ""))
            return history
        except Exception:
            return []

    async def rebuild_session_memory(
        self,
        session_id: str,
        db: AsyncSession,
    ) -> None:
        """Rebuild Redis memory from DB history after expiry/restart."""
        if not session_id:
            return

        lock_key = self._build_rebuild_lock_key(session_id)
        lock_value = str(uuid.uuid4())

        # Acquire rebuild lock
        acquired = await self.redis.set(lock_key, lock_value, nx=True, ex=settings.memory.rebuild_lock_seconds)
        if not acquired:
            return  # Another node is rebuilding

        try:
            key = self._build_key(session_id)
            existing = await self.redis.llen(key)
            if existing and existing > 0:
                await self._refresh_expiration(session_id)
                return

            # Rebuild from database
            result = await db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
            )
            messages = result.scalars().all()

            await self.redis.delete(key)
            dedup_key = self._build_dedup_key(session_id)
            await self.redis.delete(dedup_key)

            for msg in messages:
                await self.append_message(msg)
        finally:
            await self.redis.eval(RELEASE_LOCK_SCRIPT, 1, lock_key, lock_value)

    async def clear_session_memory(self, session_id: str) -> None:
        """Clear all Redis memory for a session."""
        if not session_id:
            return
        key = self._build_key(session_id)
        dedup_key = self._build_dedup_key(session_id)
        try:
            await self.redis.delete(key, dedup_key)
        except Exception:
            pass

    async def _refresh_expiration(self, session_id: str) -> None:
        ttl = self._ttl_seconds()
        if ttl <= 0:
            return
        key = self._build_key(session_id)
        dedup_key = self._build_dedup_key(session_id)
        await self.redis.expire(key, ttl)
        await self.redis.expire(dedup_key, ttl)
