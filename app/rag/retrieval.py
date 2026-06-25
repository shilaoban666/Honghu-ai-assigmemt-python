"""RAG Retrieval Service — orchestrates the full retrieval pipeline.

Mirrors RagRetrievalService.java. Uses RagPipeline for scope → query rewrite →
multi-mode retrieval → RRF fusion → fallback → format.
"""

from __future__ import annotations

from app.rag.pipeline import RagPipeline, RagRequest


class RagRetrievalService:
    """RAG retrieval service — main entry point for retrieval operations."""

    RAG_BLOCK_BEGIN = "<<<RAG_DOC_BEGIN>>>"
    RAG_BLOCK_END = "<<<RAG_DOC_END>>>"

    def __init__(self, db_session_factory):
        self.pipeline = RagPipeline(db_session_factory)

    async def retrieve_by_session(
        self, user_id: str, session_id: str, query: str
    ) -> list[dict]:
        """Retrieve relevant chunks for a session. Returns list of snippet dicts."""
        request = RagRequest(
            user_id=user_id,
            session_id=session_id,
            query=query,
        )
        snippets = await self.pipeline._retrieve(request, query, "session_only")
        return [
            {
                "content": s.content,
                "score": s.score,
                "metadata": s.metadata,
            }
            for s in snippets
        ]

    async def build_context_block(self, request: RagRequest) -> str:
        """Build formatted RAG context block for LLM prompt."""
        if not request or not request.query:
            return ""
        return await self.pipeline.execute(request)

    async def build_context_block_simple(
        self, user_id: str, session_id: str, query: str
    ) -> str:
        """Simple version — build context from userId + sessionId + query."""
        if not query:
            return ""
        request = RagRequest(
            user_id=user_id,
            session_id=session_id,
            query=query,
        )
        return await self.pipeline.execute(request)
