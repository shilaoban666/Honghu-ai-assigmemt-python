"""Chat service — mirrors ChatService.java.

Handles simple chat, structured chat, SSE streaming with persistence,
three-layer memory integration, RAG context injection, task keyword routing,
and model selection/fallback.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

import tiktoken
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.orm import (
    AiModelDefinition,
    ChatMessage,
    ChatSession,
    ChatSessionProfile,
    RagDocument,
    User,
)
from app.schemas.dto import (
    AiCallContext,
    ChatMessageWithAttachmentsResponse,
    AttachmentDto,
    ChatRequest,
    ChatResponse,
    TokenUsage,
)
from app.services.gateway import ai_gateway
from app.services.memory import ChatMemoryService

logger = logging.getLogger(__name__)

# Tokenizer for approximate counting (CL100K_BASE as in Java)
_enc = tiktoken.get_encoding("cl100k_base")
MAX_RETRIES = 3
RETRY_DELAY_MS = 2.0  # seconds


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text or ""))


class ChatService:
    """Full chat service with memory, RAG integration, and model routing."""

    def __init__(self, memory: ChatMemoryService):
        self.memory = memory

    # ── Simple Chat ────────────────────────────────────────────────

    async def simple_chat(
        self,
        db: AsyncSession,
        message: str,
    ) -> ChatResponse:
        """Mirrors ChatService.simpleChat()."""
        return await self._simple_chat_with_retry(db, message, MAX_RETRIES)

    async def _simple_chat_with_retry(
        self,
        db: AsyncSession,
        message: str,
        remaining: int,
    ) -> ChatResponse:
        try:
            logger.info(f"Simple chat: {message[:50]}... (retries left: {remaining})")
            task_match = await self._resolve_task_keyword(db, message)
            system_prompt = self._resolve_system_prompt(None, task_match)
            model = await self._resolve_model(db, None, None, message, task_match)

            return await ai_gateway.chat(
                model,
                [SystemMessage(content=system_prompt), HumanMessage(content=message)],
                context=AiCallContext.anonymous("simple-chat"),
            )
        except Exception as e:
            if remaining > 0 and self._is_connection_error(e):
                await asyncio.sleep(RETRY_DELAY_MS)
                return await self._simple_chat_with_retry(db, message, remaining - 1)
            logger.error(f"Simple chat failed: {e}")
            return ChatResponse(
                content=f"Chat service temporarily unavailable: {e}",
                success=False,
                error_message=str(e),
                timestamp=int(time.time() * 1000),
            )

    # ── Stream Chat (SSE) ──────────────────────────────────────────

    async def stream_chat(
        self,
        db: AsyncSession,
        message: str,
    ) -> AsyncIterator[str]:
        """Simple streaming — yields content strings."""
        task_match = await self._resolve_task_keyword(db, message)
        system_prompt = self._resolve_system_prompt(None, task_match)
        model = await self._resolve_model(db, None, None, message, task_match)

        async for chunk in ai_gateway.stream_chat(
            model,
            [SystemMessage(content=system_prompt), HumanMessage(content=message)],
            context=AiCallContext.anonymous("simple-stream"),
        ):
            if chunk.content:
                yield chunk.content

    # ── Structured Stream with Persistence ─────────────────────────

    async def structured_stream_chat_with_persistence(
        self,
        db: AsyncSession,
        redis,
        request: ChatRequest,
    ) -> AsyncIterator[ChatResponse]:
        """Full streaming chat with persistence — mirrors structuredStreamChatWithPersistence()."""
        # Step 0: Model routing
        message_content = request.message or ""
        task_match = await self._resolve_task_keyword(db, message_content)

        user_result = await db.execute(select(User).where(User.user_id == request.user_id))
        chat_user = user_result.scalar_one_or_none()
        if not chat_user:
            raise ValueError("User not found")

        selected_model = await self._resolve_model(db, chat_user, request.model, message_content, task_match)
        request.model = selected_model.model_code

        # Step 1: Ensure session exists
        session_id = request.session_id or str(uuid.uuid4())
        request.session_id = session_id

        session_result = await db.execute(select(ChatSession).where(ChatSession.session_id == session_id))
        session = session_result.scalar_one_or_none()
        if not session:
            session = ChatSession(
                session_id=session_id,
                user_id=chat_user.user_id,
                user_name=chat_user.username,
                system_role=request.system_message,
                session_name=message_content[:64],
                session_status="active",
            )
            db.add(session)
            await db.commit()

        # Step 2: Save user message
        user_msg = ChatMessage(
            session_id=session_id,
            chat_role="user",
            content_type="text",
            content=message_content,
            status="active",
        )
        db.add(user_msg)
        await db.commit()
        await db.refresh(user_msg)
        await self.memory.append_message(user_msg)

        # Step 3: Load history + memory
        history_items = await self.memory.get_messages(session_id)
        if not history_items:
            # Fallback to DB
            db_result = await db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
            )
            db_history = db_result.scalars().all()
            memory_svc = ChatMemoryService(redis)
            for msg in db_history:
                await memory_svc.append_message(msg)
            history_items = [self._msg_to_dict(m) for m in db_history]

        system_prompt = self._resolve_system_prompt(request.system_message, task_match)
        memory_prompts: list[str] = []  # TODO: implement summary/profile in Phase 1b

        # RAG context
        rag_context = ""  # TODO: integrate RAG retrieval in Phase 2

        # Step 4: Build messages and stream
        messages = [SystemMessage(content=system_prompt)]
        for mem in memory_prompts:
            if mem:
                messages.append(SystemMessage(content=mem))
        if rag_context:
            messages.append(SystemMessage(content=rag_context))

        # Add history messages
        for item in self._select_history_by_token_limit(history_items, system_prompt, memory_prompts, rag_context):
            role = item.get("chatRole", "user")
            content = item.get("content", "") or ""
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(SystemMessage(content=content))  # as context

        # Step 5: Stream response
        full_content_buf = ""

        async for chunk in ai_gateway.stream_chat(
            selected_model,
            messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            context=AiCallContext(
                user_id=request.user_id,
                workspace_id=request.workspace_id,
                session_id=session_id,
                chat_id=user_msg.chat_id,
                source="structured-stream",
                query=message_content,
            ),
        ):
            if chunk.content:
                full_content_buf += chunk.content
            yield chunk

        # Step 6: Save assistant message
        assistant_msg = ChatMessage(
            session_id=session_id,
            chat_role="assistant",
            content_type="text",
            content=full_content_buf,
            status="completed",
        )
        db.add(assistant_msg)
        await db.commit()
        await self.memory.append_message(assistant_msg)

        # Step 7: Trigger async summary (Phase 1b)
        # chat_summary_service.trigger_refresh_session_summary_async(session_id, user_id)

    # ── Structured Chat (non-streaming) ────────────────────────────

    async def structured_chat(
        self,
        db: AsyncSession,
        request: ChatRequest,
    ) -> ChatResponse:
        """Mirrors ChatService.structuredChat()."""
        return await self._structured_chat_with_retry(db, request, MAX_RETRIES)

    async def _structured_chat_with_retry(
        self,
        db: AsyncSession,
        request: ChatRequest,
        remaining: int,
    ) -> ChatResponse:
        try:
            chat_user = None
            if request.user_id:
                result = await db.execute(select(User).where(User.user_id == request.user_id))
                chat_user = result.scalar_one_or_none()

            task_match = await self._resolve_task_keyword(db, request.message)
            model_def = await self._resolve_model(db, chat_user, request.model, request.message, task_match)
            request.model = model_def.model_code

            system_prompt = self._resolve_system_prompt(request.system_message, task_match)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=request.message),
            ]

            return await ai_gateway.chat(
                model_def,
                messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                context=AiCallContext(
                    user_id=request.user_id,
                    workspace_id=request.workspace_id,
                    session_id=request.session_id,
                    source="structured-chat",
                    query=request.message,
                ),
            )
        except Exception as e:
            if remaining > 0 and self._is_connection_error(e):
                await asyncio.sleep(RETRY_DELAY_MS)
                return await self._structured_chat_with_retry(db, request, remaining - 1)
            logger.error(f"Structured chat failed: {e}")
            return ChatResponse(
                content=f"Chat service temporarily unavailable: {e}",
                success=False,
                error_message=str(e),
                timestamp=int(time.time() * 1000),
            )

    # ── Chat History ───────────────────────────────────────────────

    async def get_chat_history(
        self,
        db: AsyncSession,
        user_id: str | None,
        session_id: str,
    ) -> list[ChatMessageWithAttachmentsResponse]:
        """Get chat history with attachments."""
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
        )
        messages = result.scalars().all()

        chat_ids = [m.chat_id for m in messages if m.chat_id]
        attachments_by_chat: dict[int, list[RagDocument]] = {}
        if chat_ids:
            doc_result = await db.execute(
                select(RagDocument).where(RagDocument.chat_id.in_(chat_ids))
            )
            for doc in doc_result.scalars().all():
                if doc.chat_id:
                    attachments_by_chat.setdefault(doc.chat_id, []).append(doc)

        return [
            ChatMessageWithAttachmentsResponse(
                chatId=m.chat_id,
                sessionId=m.session_id,
                chatRole=m.chat_role,
                contentType=m.content_type,
                content=m.content,
                status=m.status,
                createdAt=m.created_at,
                attachments=[
                    AttachmentDto(
                        fileId=doc.file_id,
                        fileName=doc.file_name,
                        fileType=doc.file_type,
                        fileSize=doc.file_size,
                        status=doc.status.value if doc.status else None,
                    )
                    for doc in attachments_by_chat.get(m.chat_id, [])
                ],
            )
            for m in messages
        ]

    # ── Helpers ─────────────────────────────────────────────────────

    def _resolve_system_prompt(
        self,
        explicit_prompt: str | None,
        task_match: dict | None,
    ) -> str:
        """Resolve system prompt. Mirrors resolveSystemPrompt()."""
        if explicit_prompt:
            return explicit_prompt
        if task_match and task_match.get("task_type"):
            task_type = task_match["task_type"]
            prompt_locs = settings.chat.prompt.task_type_prompt_locations
            path = prompt_locs.get(task_type)
            if path:
                try:
                    import os
                    full_path = os.path.join(os.path.dirname(__file__), "..", "..", path)
                    with open(full_path, "r", encoding="utf-8") as f:
                        return f.read()
                except Exception:
                    pass
        # Default
        try:
            import os
            default_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "prompts", "default-system-prompt.txt"
            )
            with open(default_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return "You are a helpful AI assistant."

    async def _resolve_task_keyword(
        self,
        db: AsyncSession,
        message: str,
    ) -> dict | None:
        """Match message against task keywords. Mirrors resolveTaskKeywordMatch()."""
        if not message:
            return None
        from app.models.orm import AiTaskKeyword
        result = await db.execute(
            select(AiTaskKeyword).where(AiTaskKeyword.enabled == True)
        )
        keywords = result.scalars().all()

        message_lower = message.lower()
        best: AiTaskKeyword | None = None
        for kw in keywords:
            if kw.keyword and kw.keyword.lower() in message_lower:
                if best is None or (kw.priority or 0) > (best.priority or 0):
                    best = kw

        if best:
            return {"task_type": best.task_type, "keyword": best.keyword, "priority": best.priority}
        return None

    async def _resolve_model(
        self,
        db: AsyncSession,
        user: User | None,
        explicit_model: str | None,
        message: str | None,
        task_match: dict | None,
    ) -> AiModelDefinition:
        """Resolve which model to use. Mirrors resolveChatModelDefinition()."""
        model_code = explicit_model

        if not model_code:
            routing = settings.ai.routing
            msg_len = len(message or "")
            if msg_len <= routing.simple_query_length_threshold:
                model_code = routing.simple_default_model
            else:
                model_code = routing.complex_default_model

        result = await db.execute(
            select(AiModelDefinition).where(
                AiModelDefinition.model_code == model_code,
                AiModelDefinition.enabled == True,
            )
        )
        model = result.scalar_one_or_none()
        if not model:
            model_code = settings.ai.default_model
            result = await db.execute(
                select(AiModelDefinition).where(
                    AiModelDefinition.model_code == model_code,
                    AiModelDefinition.enabled == True,
                )
            )
            model = result.scalar_one_or_none()

        if not model:
            raise ValueError(f"No enabled model found. Requested: {explicit_model}")

        return model

    def _select_history_by_token_limit(
        self,
        history_items: list[dict],
        system_prompt: str,
        memory_prompts: list[str],
        rag_context: str,
    ) -> list[dict]:
        """Select recent history within token budget. Mirrors selectHistoryMessagesByTokenLimit()."""
        system_tokens = _count_tokens(system_prompt)
        memory_tokens = sum(_count_tokens(m) for m in memory_prompts)
        rag_tokens = _count_tokens(rag_context)
        reserved = system_tokens + memory_tokens + rag_tokens + 200

        limit = max(settings.memory.minimum_history_tokens, settings.memory.prompt_token_limit - reserved)
        if limit <= 0:
            return []

        selected = []
        used = 0
        for item in reversed(history_items):
            content = item.get("content", "") or ""
            tokens = _count_tokens(content) + 4
            if used + tokens > limit:
                break
            selected.append(item)
            used += tokens

        return list(reversed(selected))

    def _resolve_history_token_limit(
        self,
        system_prompt: str,
        memory_prompts: list[str],
        rag_context: str,
    ) -> int:
        """Calculate remaining token budget for history."""
        reserved = _count_tokens(system_prompt) + sum(_count_tokens(m) for m in memory_prompts) + _count_tokens(rag_context) + 200
        return max(settings.memory.minimum_history_tokens, settings.memory.prompt_token_limit - reserved)

    def _is_connection_error(self, e: Exception) -> bool:
        msg = str(e).lower()
        return any(kw in msg for kw in [
            "connection refused", "connect timed out", "connection reset",
            "broken pipe", "closedchannel", "timeout",
        ])

    def _msg_to_dict(self, msg: ChatMessage) -> dict:
        return {
            "chatId": msg.chat_id,
            "sessionId": msg.session_id,
            "chatRole": msg.chat_role,
            "contentType": msg.content_type,
            "status": msg.status,
            "content": msg.content,
            "createdAt": msg.created_at.isoformat() if msg.created_at else None,
        }
