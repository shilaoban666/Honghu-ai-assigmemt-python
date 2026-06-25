"""MCP Client — mirrors the Java MCP skill provider.

Implements JSON-RPC 2.0 over HTTP for communicating with MCP servers.
Supports: tools/list, tools/call, resources/list, prompts/list.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class McpClient:
    """JSON-RPC 2.0 client for MCP (Model Context Protocol) servers."""

    def __init__(self, server_url: str, timeout: float = 30.0):
        self.server_url = server_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _rpc_call(self, method: str, params: dict | None = None) -> dict:
        """Make a JSON-RPC 2.0 call."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self._next_id(),
        }
        try:
            resp = await self._client.post(
                f"{self.server_url}/jsonrpc",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            result = resp.json()
            if "error" in result:
                raise McpError(result["error"].get("message", "MCP error"))
            return result.get("result", {})
        except httpx.HTTPError as e:
            logger.error(f"MCP call failed: {method} -> {e}")
            raise McpError(f"MCP HTTP error: {e}")

    # ── Tools ──────────────────────────────────────────────────

    async def list_tools(self) -> list[dict]:
        """List available tools from the MCP server."""
        result = await self._rpc_call("tools/list")
        return result.get("tools", [])

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a specific tool on the MCP server."""
        result = await self._rpc_call("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        content = result.get("content", [])
        if isinstance(content, list) and content:
            return str(content[0].get("text", str(content[0])))
        return str(content)

    # ── Resources ──────────────────────────────────────────────

    async def list_resources(self) -> list[dict]:
        """List available resources from the MCP server."""
        result = await self._rpc_call("resources/list")
        return result.get("resources", [])

    async def read_resource(self, uri: str) -> str:
        """Read a resource from the MCP server."""
        result = await self._rpc_call("resources/read", {"uri": uri})
        contents = result.get("contents", [])
        if isinstance(contents, list) and contents:
            return str(contents[0].get("text", str(contents[0])))
        return str(contents)

    # ── Prompts ────────────────────────────────────────────────

    async def list_prompts(self) -> list[dict]:
        """List available prompt templates."""
        result = await self._rpc_call("prompts/list")
        return result.get("prompts", [])

    async def get_prompt(self, name: str, arguments: dict | None = None) -> str:
        """Get a prompt template with arguments filled."""
        result = await self._rpc_call("prompts/get", {
            "name": name,
            "arguments": arguments or {},
        })
        messages = result.get("messages", [])
        return "\n".join(m.get("content", {}).get("text", "") for m in messages)

    async def close(self):
        await self._client.aclose()


class McpError(Exception):
    """MCP protocol error."""
    pass


# ── MCP Tool Wrapper for LangChain ────────────────────────────────

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class McpToolInput(BaseModel):
    """Generic input for MCP tools — accepts arbitrary JSON arguments."""
    arguments: str = Field(default="{}", description="JSON string of tool arguments")


class McpToolWrapper(BaseTool):
    """Wraps an MCP tool as a LangChain Tool."""

    name: str = ""
    description: str = ""
    mcp_client: McpClient | None = None
    args_schema: type[BaseModel] = McpToolInput

    def _run(self, arguments: str = "{}") -> str:
        """Synchronous run — not supported for MCP, use async."""
        raise NotImplementedError("MCP tools require async invocation")

    async def _arun(self, arguments: str = "{}") -> str:
        """Async run — calls the MCP server."""
        if not self.mcp_client:
            return json.dumps({"error": "MCP client not initialized"})
        try:
            args_dict = json.loads(arguments) if isinstance(arguments, str) else arguments
            return await self.mcp_client.call_tool(self.name, args_dict)
        except Exception as e:
            return json.dumps({"error": str(e)})
