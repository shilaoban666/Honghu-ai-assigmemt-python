"""RAG Embedding Service — OpenAI-compatible embedding API.

Mirrors the embedding configuration in the Java version.
Uses LangChain's OpenAIEmbeddings for compatibility with any OpenAI-compatible endpoint.
"""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


class EmbeddingService:
    """Generates embeddings using OpenAI-compatible API."""

    def __init__(self):
        # Use Aliyun DashScope as default embedding provider (same as Java)
        provider = settings.ai.providers.get("aliyun")
        base_url = provider.base_url if provider else "https://dashscope.aliyuncs.com/compatible-mode"
        api_key = provider.api_key if provider else ""

        self._embeddings = OpenAIEmbeddings(
            model="text-embedding-v3",
            dimensions=settings.milvus.embedding_dimension,
            openai_api_base=base_url,
            openai_api_key=api_key,
        )

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        # LangChain's embed_documents is synchronous; wrap in executor for async
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embeddings.embed_documents, texts)

    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embeddings.embed_query, query)


# Singleton
embedding_service = EmbeddingService()
