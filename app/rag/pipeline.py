"""RAG Retrieval Pipeline — mirrors the Java retrieval pipeline.

Orchestrates: scope resolution → query analysis → multi-mode retrieval →
RRF fusion → scope fallback → rerank → augment → format.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.core.config import settings
from app.rag.embedding import embedding_service
from app.rag.vectorstore import milvus_store

logger = logging.getLogger(__name__)


class RagSnippet:
    """A single retrieval hit — mirrors RagSnippet.java."""
    def __init__(self, content: str, score: float, metadata: dict):
        self.content = content
        self.score = score
        self.metadata = metadata


class RagPipeline:
    """Full RAG retrieval pipeline."""

    RAG_BLOCK_BEGIN = "<<<RAG_DOC_BEGIN>>>"
    RAG_BLOCK_END = "<<<RAG_DOC_END>>>"

    def __init__(self, db_session_factory):
        self.db_factory = db_session_factory

    async def execute(self, request: "RagRequest") -> str:
        """Execute the full pipeline and return formatted context block."""
        # 1. Resolve scope
        scope = self._resolve_scope(request)

        # 2. Optional query rewrite
        query = await self._maybe_rewrite_query(request.query, request.recent_history)

        # 3. Multi-mode retrieval
        snippets = await self._retrieve(request, query, scope)

        # 4. RRF fusion (if multiple retrievers)
        rag_cfg = settings.rag
        if rag_cfg.retrieval.fusion.enabled:
            snippets = self._rrf_fusion([snippets])

        # 5. Scope fallback
        snippets = await self._scope_fallback(request, query, snippets, scope)

        # 6. Rerank (optional)
        if rag_cfg.retrieval.rerank.enabled:
            snippets = await self._rerank(query, snippets)

        # 7. Augment (dedup, compress, citation, budget)
        snippets = self._augment(snippets)

        # 8. Format
        return self._format(snippets)

    # ── Scope Resolution ──────────────────────────────────────

    def _resolve_scope(self, request: "RagRequest") -> str:
        """Determine retrieval scope: attachment_chat_first, session_only, etc."""
        scope = settings.rag.retrieval.scope.default_scope
        if request.attachment_file_ids:
            return "attachment_chat_first"
        return scope

    # ── Query Rewrite ─────────────────────────────────────────

    async def _maybe_rewrite_query(self, query: str, history: list | None) -> str:
        """Optionally rewrite query for better retrieval."""
        if not settings.rag.query_rewrite.enabled:
            return query
        # Simple pronoun resolution for now
        # Full implementation would use an LLM to rewrite
        return query

    # ── Retrieval ─────────────────────────────────────────────

    async def _retrieve(
        self, request: "RagRequest", query: str, scope: str
    ) -> list[RagSnippet]:
        """Execute retrieval based on configured mode (keyword/vector)."""
        mode = settings.rag.retrieval.mode
        top_k = settings.rag.retrieval.top_k
        threshold = settings.rag.retrieval.similarity_threshold

        if mode == "vector":
            return await self._vector_retrieve(query, top_k, threshold, request)
        else:
            return await self._keyword_retrieve(query, top_k, threshold, request)

    async def _vector_retrieve(
        self, query: str, top_k: int, threshold: float, request: "RagRequest"
    ) -> list[RagSnippet]:
        """Vector similarity search via Milvus."""
        embedding = await embedding_service.embed_query(query)

        # Build filter expression based on scope
        filter_expr = self._build_filter_expr(request)

        results = await milvus_store.search(
            query_embedding=embedding,
            top_k=top_k,
            similarity_threshold=threshold,
            filter_expr=filter_expr,
        )
        return [
            RagSnippet(
                content=r["content"],
                score=r["score"],
                metadata=r["metadata"],
            )
            for r in results
        ]

    async def _keyword_retrieve(
        self, query: str, top_k: int, threshold: float, request: "RagRequest"
    ) -> list[RagSnippet]:
        """Keyword-based retrieval from PostgreSQL chunk table."""
        # Extract keywords from query
        keywords = self._extract_keywords(query)

        async with self.db_factory() as db:
            from sqlalchemy import select, or_, func
            from app.models.orm import RagDocumentChunk, RagDocument, RagDocumentStatus

            conditions = []
            for kw in keywords:
                conditions.append(RagDocumentChunk.content.ilike(f"%{kw}%"))

            stmt = (
                select(RagDocumentChunk, RagDocument)
                .join(RagDocument, RagDocumentChunk.document_id == RagDocument.document_id)
                .where(
                    RagDocument.status == RagDocumentStatus.INDEXED,
                    or_(*conditions) if conditions else True,
                )
                .limit(top_k * 3)  # fetch more for scoring
            )

            # Scope filtering
            if request.session_id:
                stmt = stmt.where(RagDocument.session_id == request.session_id)

            result = await db.execute(stmt)
            rows = result.all()

            # Score by keyword match count
            snippets = []
            for chunk, doc in rows:
                score = sum(1 for kw in keywords if kw in (chunk.content or "").lower()) / max(1, len(keywords))
                if score >= threshold:
                    snippets.append(RagSnippet(
                        content=chunk.content,
                        score=score,
                        metadata={
                            "document_id": doc.document_id,
                            "chunk_index": chunk.chunk_index,
                            "file_name": doc.file_name,
                            "file_type": doc.file_type,
                            "session_id": doc.session_id,
                            "chat_id": doc.chat_id,
                            "file_id": doc.file_id,
                        },
                    ))

            snippets.sort(key=lambda s: s.score, reverse=True)
            return snippets[:top_k]

    def _build_filter_expr(self, request: "RagRequest") -> str | None:
        """Build Milvus filter expression for scope."""
        parts = []
        if request.session_id:
            parts.append(f'session_id == "{request.session_id}"')
        if request.chat_id:
            parts.append(f"chat_id == {request.chat_id}")
        if request.attachment_file_ids:
            ids = ", ".join(f'"{fid}"' for fid in request.attachment_file_ids)
            parts.append(f"file_id in [{ids}]")
        return " and ".join(parts) if parts else None

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract meaningful keywords from query."""
        # Simple implementation: split and filter short words
        import re
        words = re.findall(r"[\w\u4e00-\u9fff]+", query.lower())
        min_len = settings.rag.retrieval.min_keyword_length
        return [w for w in words if len(w) >= min_len]

    # ── RRF Fusion ────────────────────────────────────────────

    def _rrf_fusion(self, result_sets: list[list[RagSnippet]]) -> list[RagSnippet]:
        """Reciprocal Rank Fusion across multiple result sets."""
        k = settings.rag.retrieval.fusion.rrf_k
        scores: dict[str, tuple[float, RagSnippet]] = {}

        for results in result_sets:
            for rank, snippet in enumerate(results, start=1):
                key = snippet.content[:100]  # dedup by content prefix
                rrf_score = 1.0 / (k + rank)
                if key in scores:
                    prev_score, prev_snippet = scores[key]
                    scores[key] = (prev_score + rrf_score, prev_snippet)
                else:
                    scores[key] = (rrf_score, snippet)

        merged = sorted(scores.values(), key=lambda x: x[0], reverse=True)
        limit = settings.rag.retrieval.fusion.candidate_limit_after_fusion
        return [s for _, s in merged[:limit]]

    # ── Scope Fallback ────────────────────────────────────────

    async def _scope_fallback(
        self, request: "RagRequest", query: str, snippets: list[RagSnippet], scope: str
    ) -> list[RagSnippet]:
        """If not enough hits, expand scope progressively."""
        fallback_cfg = settings.rag.retrieval.scope
        min_hits = fallback_cfg.fallback_min_hits
        min_score = fallback_cfg.fallback_min_score

        good_hits = [s for s in snippets if s.score >= min_score]
        if len(good_hits) >= min_hits:
            return snippets

        # Fallback: expand to session scope
        if "attachment" in scope or "chat" in scope:
            logger.info(f"RAG scope fallback: {scope} → session (hits={len(good_hits)})")
            fallback_request = RagRequest(
                user_id=request.user_id,
                session_id=request.session_id,
                query=query,
            )
            fallback_snippets = await self._retrieve(fallback_request, query, "session_only")
            # Penalize fallback results
            penalty = fallback_cfg.session_penalty_factor
            for s in fallback_snippets:
                s.score *= penalty
            return snippets + fallback_snippets

        return snippets

    # ── Rerank ────────────────────────────────────────────────

    async def _rerank(self, query: str, snippets: list[RagSnippet]) -> list[RagSnippet]:
        """Rerank results using a cross-encoder (placeholder)."""
        # TODO: Implement cross-encoder reranking (e.g., bge-reranker)
        # For now, return as-is with a noop
        return snippets

    # ── Augment ───────────────────────────────────────────────

    def _augment(self, snippets: list[RagSnippet]) -> list[RagSnippet]:
        """Apply augmentation filters: dedup, compress, citation, budget."""
        cfg = settings.rag.augmentor
        result = snippets

        if cfg.dedup_enabled:
            result = self._dedup(result)
        if cfg.compress_enabled:
            result = self._compress(result)
        if cfg.budget_enabled:
            result = self._apply_budget(result, cfg.token_budget)

        return result

    def _dedup(self, snippets: list[RagSnippet]) -> list[RagSnippet]:
        """Remove duplicate/near-duplicate snippets."""
        seen = set()
        unique = []
        for s in snippets:
            key = s.content[:200]
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique

    def _compress(self, snippets: list[RagSnippet]) -> list[RagSnippet]:
        """Compress snippets by trimming to most relevant portions."""
        # Simple: cap each snippet to ~500 chars
        for s in snippets:
            if len(s.content) > 500:
                s.content = s.content[:500] + "..."
        return snippets

    def _apply_budget(self, snippets: list[RagSnippet], token_budget: int) -> list[RagSnippet]:
        """Enforce token budget on context."""
        total = 0
        result = []
        for s in snippets:
            est_tokens = len(s.content) // 2
            if total + est_tokens > token_budget:
                break
            result.append(s)
            total += est_tokens
        return result

    # ── Format ────────────────────────────────────────────────

    def _format(self, snippets: list[RagSnippet]) -> str:
        """Format snippets into LLM context block with safety markers."""
        if not snippets:
            return ""

        parts = [self.RAG_BLOCK_BEGIN]
        parts.append("以下是从知识库中检索到的相关资料，请基于这些资料回答问题。")
        parts.append("注意：资料可能不完整，请结合你的知识综合判断。\n")

        for i, s in enumerate(snippets, 1):
            source = s.metadata.get("file_name", "unknown")
            parts.append(f"[资料 {i}] (来源: {source}, 相关度: {s.score:.2f})")
            parts.append(s.content)
            parts.append("")

        parts.append(self.RAG_BLOCK_END)
        return "\n".join(parts)


class RagRequest:
    """RAG retrieval request — mirrors RagRequest.java."""
    def __init__(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        chat_id: int | None = None,
        attachment_file_ids: list[str] | None = None,
        query: str = "",
        recent_history: list | None = None,
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.chat_id = chat_id
        self.attachment_file_ids = attachment_file_ids or []
        self.query = query
        self.recent_history = recent_history or []
