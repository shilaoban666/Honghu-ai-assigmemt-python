"""Chat API endpoints — mirrors ChatController.java."""

import json
import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.core.database import get_db
from app.core.redis_client import get_redis
from app.schemas.dto import (
    ChatMessageWithAttachmentsResponse,
    ChatRequest,
    ChatResponse,
)
from app.security.jwt import get_current_user, JwtPrincipal
from app.services.chat import ChatService
from app.services.memory import ChatMemoryService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


def _get_chat_service(redis: Redis = Depends(get_redis)) -> ChatService:
    return ChatService(ChatMemoryService(redis))


@router.get("/simple")
async def simple_chat(
    message: str = Query(..., description="User message"),
    db: AsyncSession = Depends(get_db),
    chat_svc: ChatService = Depends(_get_chat_service),
):
    """Simple chat — send a message, get AI reply."""
    response = await chat_svc.simple_chat(db, message)
    return response


@router.post("/structured")
async def structured_chat(
    request: ChatRequest,
    user_id: str | None = Header(None, alias="X-User-Id"),
    workspace_id: str | None = Header(None, alias="X-Workspace-Id"),
    db: AsyncSession = Depends(get_db),
    chat_svc: ChatService = Depends(_get_chat_service),
):
    """Structured chat with full request body."""
    _bind_context(request, user_id, workspace_id)
    response = await chat_svc.structured_chat(db, request)
    return response


@router.get("/stream")
async def stream_chat(
    message: str = Query(..., description="User message"),
    db: AsyncSession = Depends(get_db),
    chat_svc: ChatService = Depends(_get_chat_service),
):
    """Streaming chat — SSE text stream."""
    async def event_stream():
        async for content in chat_svc.stream_chat(db, message):
            yield f"data: {json.dumps({'content': content})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/structured/stream")
async def structured_stream_chat(
    request: ChatRequest,
    user_id: str | None = Header(None, alias="X-User-Id"),
    workspace_id: str | None = Header(None, alias="X-Workspace-Id"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    """Structured streaming chat — SSE with full request body."""
    _bind_context(request, user_id, workspace_id)
    chat_svc = ChatService(ChatMemoryService(redis))

    async def event_stream():
        async for chunk in chat_svc.structured_stream_chat_with_persistence(db, redis, request):
            yield f"data: {chunk.model_dump_json(by_alias=True)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/structured/stream/persistent")
async def structured_stream_chat_persistent(
    request: ChatRequest,
    user_id: str | None = Header(None, alias="X-User-Id"),
    workspace_id: str | None = Header(None, alias="X-Workspace-Id"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    """Persistent structured streaming chat — SSE with DB persistence + memory."""
    _bind_context(request, user_id, workspace_id)
    chat_svc = ChatService(ChatMemoryService(redis))

    async def event_stream():
        async for chunk in chat_svc.structured_stream_chat_with_persistence(db, redis, request):
            yield f"data: {chunk.model_dump_json(by_alias=True)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    user_id: str | None = Header(None, alias="X-User-Id"),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
) -> list[ChatMessageWithAttachmentsResponse]:
    """Get chat history for a session."""
    chat_svc = ChatService(ChatMemoryService(redis))
    return await chat_svc.get_chat_history(db, user_id, session_id)


def _bind_context(request: ChatRequest, user_id: str | None, workspace_id: str | None):
    """Bind user context from headers to request."""
    if user_id:
        request.user_id = user_id
    if workspace_id:
        request.workspace_id = workspace_id
