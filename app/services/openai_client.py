"""OpenAI-compatible HTTP chat client — mirrors OpenAiCompatibleChatClient.java.

Supports both non-streaming and streaming (SSE) chat, with function calling (tools).
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.core.config import ProviderSettings, settings
from app.models.orm import AiModelDefinition
from app.schemas.dto import ChatResponse, TokenUsage


class OpenAiCompatibleChatClient:
    """HTTP client for OpenAI-compatible providers (DeepSeek, Gemini, Qwen, etc.)."""

    MAX_TOOL_ROUNDS = 5

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=settings.ai.connection.connect_timeout / 1000.0,
                    read=settings.ai.connection.read_timeout / 1000.0,
                    write=60.0,
                    pool=settings.ai.connection.pool_size,
                ),
                limits=httpx.Limits(max_keepalive_connections=settings.ai.connection.pool_size),
            )
        return self._client

    def _build_url(self, provider: ProviderSettings) -> str:
        base = provider.base_url.rstrip("/")
        path = provider.chat_completions_path
        if not path.startswith("/"):
            path = "/" + path
        return base + path

    def _build_headers(self, provider: ProviderSettings) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if provider.use_api_key and provider.api_key:
            prefix = provider.api_key_prefix or ""
            headers[provider.api_key_header] = prefix + provider.api_key
        return headers

    def _to_openai_messages(self, messages: list[BaseMessage]) -> list[dict]:
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                entry: dict = {"role": "assistant", "content": msg.content}
                # Include tool_calls if present
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.get("id", f"call_{i}"),
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("args", {})) if isinstance(tc.get("args"), dict) else tc.get("args", "{}"),
                            },
                        }
                        for i, tc in enumerate(msg.tool_calls)
                    ]
                result.append(entry)
            elif isinstance(msg, ToolMessage):
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
        return result

    def _to_tool_specs(self, tools: list[BaseTool]) -> list[dict]:
        specs = []
        for tool in tools:
            spec = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.args_schema.schema() if tool.args_schema else {"type": "object", "properties": {}},
                },
            }
            specs.append(spec)
        return specs

    async def chat(
        self,
        model: AiModelDefinition,
        provider: ProviderSettings,
        messages: list[BaseMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[BaseTool] | None = None,
        tool_context: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Non-streaming chat with optional function calling."""
        safe_tools = tools or []
        conversation = self._to_openai_messages(messages)
        usage_acc = {"prompt": 0, "completion": 0, "total": 0, "cached": 0}

        for round_num in range(self.MAX_TOOL_ROUNDS + 1):
            round_tools = safe_tools if round_num < self.MAX_TOOL_ROUNDS else []

            payload = {
                "model": model.api_model_name,
                "messages": conversation,
                "stream": False,
            }
            if temperature is not None:
                payload["temperature"] = temperature
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            if round_tools:
                payload["tools"] = self._to_tool_specs(round_tools)
                payload["tool_choice"] = "auto"

            resp = await self.client.post(
                self._build_url(provider),
                headers=self._build_headers(provider),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

            # Track usage
            if "usage" in data:
                u = data["usage"]
                usage_acc["prompt"] += u.get("prompt_tokens", 0)
                usage_acc["completion"] += u.get("completion_tokens", 0)
                usage_acc["total"] += u.get("total_tokens", 0)

            choice = data["choices"][0] if data.get("choices") else {}
            msg = choice.get("message", {})

            # Check for tool calls
            tool_calls = msg.get("tool_calls", [])
            if not tool_calls:
                return ChatResponse(
                    content=msg.get("content", ""),
                    model=model.model_code,
                    timestamp=int(time.time() * 1000),
                    success=True,
                    token_usage=TokenUsage(
                        prompt_tokens=usage_acc["prompt"],
                        completion_tokens=usage_acc["completion"],
                        cached_prompt_tokens=usage_acc["cached"],
                        total_tokens=usage_acc["total"],
                    ),
                )

            # Execute tools
            conversation.append(msg)
            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                tool_args_str = tc["function"].get("arguments", "{}")
                try:
                    tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                except json.JSONDecodeError:
                    tool_args = {}

                result = await self._invoke_tool(safe_tools, tool_name, tool_args, tool_context)
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

        return ChatResponse(content="", model=model.model_code, timestamp=int(time.time() * 1000), success=True)

    async def stream_chat(
        self,
        model: AiModelDefinition,
        provider: ProviderSettings,
        messages: list[BaseMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[BaseTool] | None = None,
        tool_context: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatResponse]:
        """Streaming chat with SSE parsing, including function calling support."""
        safe_tools = tools or []

        if not safe_tools:
            async for chunk in self._stream_simple(model, provider, messages, temperature, max_tokens):
                yield chunk
            return

        # With tools: multi-round streaming
        conversation = self._to_openai_messages(messages)
        for round_num in range(self.MAX_TOOL_ROUNDS + 1):
            round_tools = safe_tools if round_num < self.MAX_TOOL_ROUNDS else []
            content_buf = ""
            tool_call_acc: dict[int, dict] = {}

            async for chunk in self._stream_raw(model, provider, conversation, temperature, max_tokens, round_tools):
                if chunk.get("type") == "usage":
                    yield ChatResponse(
                        model=model.model_code,
                        timestamp=int(time.time() * 1000),
                        success=True,
                        token_usage=chunk["usage"],
                    )
                elif chunk.get("type") == "content":
                    content_buf += chunk["content"]
                    yield ChatResponse(
                        content=chunk["content"],
                        model=model.model_code,
                        timestamp=int(time.time() * 1000),
                        success=True,
                    )
                elif chunk.get("type") == "tool_call_delta":
                    idx = chunk["index"]
                    if idx not in tool_call_acc:
                        tool_call_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if chunk.get("id"):
                        tool_call_acc[idx]["id"] = chunk["id"]
                    if chunk.get("name"):
                        tool_call_acc[idx]["name"] = chunk["name"]
                    if chunk.get("arguments"):
                        tool_call_acc[idx]["arguments"] += chunk["arguments"]

            tool_calls = [
                {"id": v["id"] or f"call_{k}", "function": {"name": v["name"], "arguments": v["arguments"]}}
                for k, v in tool_call_acc.items() if v["name"]
            ]

            if not tool_calls:
                return  # final text already streamed

            # Append assistant + tool results, continue
            conversation.append({
                "role": "assistant",
                "content": content_buf or None,
                "tool_calls": [
                    {"id": tc["id"], "type": "function", "function": tc["function"]}
                    for tc in tool_calls
                ],
            })
            for tc in tool_calls:
                tool_args = {}
                try:
                    tool_args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    pass
                result = await self._invoke_tool(safe_tools, tc["function"]["name"], tool_args, tool_context)
                conversation.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    async def _stream_simple(
        self,
        model: AiModelDefinition,
        provider: ProviderSettings,
        messages: list[BaseMessage],
        temperature: float | None,
        max_tokens: int | None,
    ) -> AsyncIterator[ChatResponse]:
        """Stream without tools."""
        conversation = self._to_openai_messages(messages)
        async for chunk in self._stream_raw(model, provider, conversation, temperature, max_tokens, []):
            if chunk.get("type") == "content":
                yield ChatResponse(
                    content=chunk["content"],
                    model=model.model_code,
                    timestamp=int(time.time() * 1000),
                    success=True,
                )
            elif chunk.get("type") == "usage":
                yield ChatResponse(
                    model=model.model_code,
                    timestamp=int(time.time() * 1000),
                    success=True,
                    token_usage=chunk["usage"],
                )

    async def _stream_raw(
        self,
        model: AiModelDefinition,
        provider: ProviderSettings,
        conversation: list[dict],
        temperature: float | None,
        max_tokens: int | None,
        tools: list[BaseTool],
    ) -> AsyncIterator[dict]:
        """Raw SSE stream parser, yields {type, content/usage/tool_call_delta}."""
        payload = {
            "model": model.api_model_name,
            "messages": conversation,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = self._to_tool_specs(tools)
            payload["tool_choice"] = "auto"

        async with self.client.stream(
            "POST",
            self._build_url(provider),
            headers=self._build_headers(provider),
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Usage chunk
                if "usage" in data and not data.get("choices"):
                    u = data["usage"]
                    yield {
                        "type": "usage",
                        "usage": TokenUsage(
                            prompt_tokens=u.get("prompt_tokens", 0),
                            completion_tokens=u.get("completion_tokens", 0),
                            cached_prompt_tokens=0,
                            total_tokens=u.get("total_tokens", 0),
                        ),
                    }
                    continue

                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                # Content delta
                if "content" in delta and delta["content"]:
                    yield {"type": "content", "content": delta["content"]}

                # Tool call deltas
                for tc in delta.get("tool_calls", []):
                    func = tc.get("function", {})
                    yield {
                        "type": "tool_call_delta",
                        "index": tc.get("index", 0),
                        "id": tc.get("id"),
                        "name": func.get("name"),
                        "arguments": func.get("arguments"),
                    }

    async def _invoke_tool(
        self,
        tools: list[BaseTool],
        name: str,
        args: dict,
        tool_context: dict[str, Any] | None,
    ) -> str:
        """Execute a single tool and return result string."""
        for tool in tools:
            if tool.name == name:
                try:
                    result = await tool.ainvoke(args)
                    return str(result)
                except Exception as e:
                    return json.dumps({"error": f"Tool execution failed: {str(e)}"})
        return json.dumps({"error": f"Unknown tool: {name}"})


# Singleton
openai_compatible_chat_client = OpenAiCompatibleChatClient()
