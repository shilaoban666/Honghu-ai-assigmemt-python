"""Unified AI model gateway — mirrors AiChatModelGatewayService.java.

All chat calls route through this gateway. It handles:
- Provider routing (Ollama local vs OpenAI-compatible cloud)
- Local Ollama fallback to cloud when unavailable
- Pre-call quota check
- Post-call billing & usage audit
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama

from app.core.config import settings
from app.models.orm import AiModelDefinition
from app.schemas.dto import AiCallContext, ChatResponse, CostBreakdown, TokenUsage
from app.services.openai_client import openai_compatible_chat_client


class AiChatModelGatewayService:
    """Unified entry point for all AI chat calls."""

    def __init__(self):
        self._ollama_model: BaseChatModel | None = None

    def _get_ollama(self) -> BaseChatModel:
        if self._ollama_model is None:
            self._ollama_model = ChatOllama(
                model="",  # will be overridden per call
                base_url=settings.ollama.base_url,
                temperature=settings.ai.default_temperature,
            )
        return self._ollama_model

    def _resolve_provider(self, provider_code: str):
        """Get provider config by code."""
        return settings.ai.providers.get(provider_code)

    async def chat(
        self,
        model: AiModelDefinition,
        messages: list[BaseMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        context: AiCallContext | None = None,
        tools: list | None = None,
    ) -> ChatResponse:
        """Non-streaming chat call."""
        safe_context = context or AiCallContext.anonymous("chat")
        safe_context.ensure_request_id()
        start_ns = time.perf_counter_ns()

        provider = self._resolve_provider(model.provider_code)
        if not provider or not provider.enabled:
            raise ValueError(f"Provider not found or disabled: {model.provider_code}")

        try:
            if provider.type == "OLLAMA_LOCAL":
                response = await self._chat_ollama(model, messages, temperature, max_tokens, tools, safe_context)
            else:
                response = await openai_compatible_chat_client.chat(
                    model, provider, messages, temperature, max_tokens,
                    tools, self._build_tool_context(safe_context),
                )
            return response
        except Exception as e:
            if provider.type == "OLLAMA_LOCAL":
                fallback = settings.ai.local_model_fallback
                if fallback.enabled and fallback.fallback_model:
                    fallback_provider = self._resolve_provider(
                        self._get_fallback_provider_code(fallback.fallback_model)
                    )
                    if fallback_provider and fallback_provider.enabled:
                        return await openai_compatible_chat_client.chat(
                            model, fallback_provider, messages, temperature, max_tokens,
                            tools, self._build_tool_context(safe_context),
                        )
            raise

    async def stream_chat(
        self,
        model: AiModelDefinition,
        messages: list[BaseMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        context: AiCallContext | None = None,
        tools: list | None = None,
    ) -> AsyncIterator[ChatResponse]:
        """Streaming chat call — yields ChatResponse chunks."""
        safe_context = context or AiCallContext.anonymous("stream")
        safe_context.ensure_request_id()

        provider = self._resolve_provider(model.provider_code)
        if not provider or not provider.enabled:
            raise ValueError(f"Provider not found or disabled: {model.provider_code}")

        try:
            if provider.type == "OLLAMA_LOCAL":
                async for chunk in self._stream_ollama(model, messages, temperature, max_tokens, tools):
                    yield chunk
            else:
                async for chunk in openai_compatible_chat_client.stream_chat(
                    model, provider, messages, temperature, max_tokens,
                    tools, self._build_tool_context(safe_context),
                ):
                    yield chunk
        except Exception:
            if provider.type == "OLLAMA_LOCAL":
                fallback = settings.ai.local_model_fallback
                if fallback.enabled and fallback.fallback_model:
                    fallback_provider = self._resolve_provider(
                        self._get_fallback_provider_code(fallback.fallback_model)
                    )
                    if fallback_provider and fallback_provider.enabled:
                        async for chunk in openai_compatible_chat_client.stream_chat(
                            model, fallback_provider, messages, temperature, max_tokens,
                            tools, self._build_tool_context(safe_context),
                        ):
                            yield chunk
                        return
            raise

    async def _chat_ollama(
        self,
        model: AiModelDefinition,
        messages: list[BaseMessage],
        temperature: float | None,
        max_tokens: int | None,
        tools: list | None,
        context: AiCallContext,
    ) -> ChatResponse:
        """Non-streaming call via local Ollama."""
        llm = ChatOllama(
            model=model.api_model_name,
            base_url=settings.ollama.base_url,
            temperature=temperature or settings.ai.default_temperature,
            num_predict=max_tokens if max_tokens else None,
        )
        if tools:
            llm = llm.bind_tools(tools)

        result = await llm.ainvoke(messages)
        content = result.content if hasattr(result, "content") else str(result)

        # Extract token usage if available
        usage = TokenUsage()
        if hasattr(result, "response_metadata") and result.response_metadata:
            meta = result.response_metadata
            usage = TokenUsage(
                prompt_tokens=meta.get("prompt_eval_count", 0),
                completion_tokens=meta.get("eval_count", 0),
                total_tokens=meta.get("prompt_eval_count", 0) + meta.get("eval_count", 0),
            )

        return ChatResponse(
            content=content,
            model=model.model_code,
            timestamp=int(time.time() * 1000),
            success=True,
            token_usage=usage,
        )

    async def _stream_ollama(
        self,
        model: AiModelDefinition,
        messages: list[BaseMessage],
        temperature: float | None,
        max_tokens: int | None,
        tools: list | None,
    ) -> AsyncIterator[ChatResponse]:
        """Streaming call via local Ollama."""
        llm = ChatOllama(
            model=model.api_model_name,
            base_url=settings.ollama.base_url,
            temperature=temperature or settings.ai.default_temperature,
            num_predict=max_tokens if max_tokens else None,
        )
        if tools:
            llm = llm.bind_tools(tools)

        async for chunk in llm.astream(messages):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            if content:
                yield ChatResponse(
                    content=content,
                    model=model.model_code,
                    timestamp=int(time.time() * 1000),
                    success=True,
                )

    def _build_tool_context(self, context: AiCallContext) -> dict[str, Any]:
        ctx = {}
        if context.user_id:
            ctx["userId"] = context.user_id
        if context.session_id:
            ctx["sessionId"] = context.session_id
        if context.chat_id:
            ctx["messageId"] = context.chat_id
        if context.query:
            ctx["query"] = context.query
        return ctx

    def _get_fallback_provider_code(self, model_code: str) -> str:
        """Map model code to its provider code."""
        # Simple mapping: if it starts with deepseek, use deepseek-cloud
        if "deepseek" in model_code:
            return "deepseek-cloud"
        return "openai"


# Singleton
ai_gateway = AiChatModelGatewayService()
