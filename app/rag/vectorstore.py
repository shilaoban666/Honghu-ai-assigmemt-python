"""Milvus Vector Store — mirrors the Java Milvus integration.

Handles: collection management, insert, search (vector + hybrid).
Uses pymilvus directly for full control over schema and search params.
"""

from __future__ import annotations

import logging
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


class MilvusVectorStore:
    """Milvus vector database client for RAG chunk storage and retrieval."""

    COLLECTION_NAME = settings.milvus.collection
    DIM = settings.milvus.embedding_dimension

    def __init__(self):
        self._connected = False
        self._collection: Collection | None = None

    def _connect(self):
        if self._connected:
            return
        connections.connect(
            alias="default",
            host=settings.milvus.host,
            port=settings.milvus.port,
            user=settings.milvus.username,
            password=settings.milvus.password,
            db_name=settings.milvus.database,
        )
        self._connected = True

    def _ensure_collection(self):
        """Ensure the collection exists with the correct schema."""
        self._connect()

        if utility.has_collection(self.COLLECTION_NAME):
            self._collection = Collection(self.COLLECTION_NAME)
            self._collection.load()
            return

        # Create collection with schema matching Java version
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="document_id", dtype=DataType.INT64),
            FieldSchema(name="chunk_index", dtype=DataType.INT32),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.DIM),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="file_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="chat_id", dtype=DataType.INT64),
            FieldSchema(name="file_id", dtype=DataType.VARCHAR, max_length=64),
        ]

        schema = CollectionSchema(fields, description="Honghu AI RAG Chunks")
        self._collection = Collection(self.COLLECTION_NAME, schema)

        # Create index
        index_params = {
            "metric_type": settings.milvus.metric_type,
            "index_type": settings.milvus.index_type,
            "params": {"nlist": 128},
        }
        self._collection.create_index("embedding", index_params)
        self._collection.load()
        logger.info(f"Created Milvus collection: {self.COLLECTION_NAME}")

    async def insert_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        document_id: int,
        file_name: str = "",
        file_type: str = "",
        session_id: str = "",
        chat_id: int | None = None,
        file_id: str = "",
    ) -> int:
        """Insert chunks with embeddings into Milvus. Returns number inserted."""
        self._ensure_collection()

        if not chunks or not embeddings:
            return 0

        entities = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            entities.append({
                "document_id": document_id,
                "chunk_index": chunk.get("chunk_index", i),
                "content": chunk.get("content", ""),
                "embedding": emb,
                "file_name": file_name,
                "file_type": file_type,
                "session_id": session_id,
                "chat_id": chat_id or 0,
                "file_id": file_id,
            })

        # Delete existing chunks for this document first (idempotent)
        try:
            self._collection.delete(f'document_id == {document_id}')
        except Exception:
            pass

        result = self._collection.insert(entities)
        self._collection.flush()
        return result.insert_count

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
        similarity_threshold: float = 0.55,
        filter_expr: str | None = None,
    ) -> list[dict]:
        """Vector similarity search."""
        self._ensure_collection()

        search_params = {
            "metric_type": settings.milvus.metric_type,
            "params": {"nprobe": 16},
        }

        results = self._collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=[
                "document_id", "chunk_index", "content",
                "file_name", "file_type", "session_id", "chat_id", "file_id",
            ],
        )

        hits = []
        for hits_list in results:
            for hit in hits_list:
                if hit.distance >= similarity_threshold:
                    hits.append({
                        "content": hit.entity.get("content", ""),
                        "score": hit.distance,
                        "metadata": {
                            "document_id": hit.entity.get("document_id"),
                            "chunk_index": hit.entity.get("chunk_index"),
                            "file_name": hit.entity.get("file_name"),
                            "file_type": hit.entity.get("file_type"),
                            "session_id": hit.entity.get("session_id"),
                            "chat_id": hit.entity.get("chat_id"),
                            "file_id": hit.entity.get("file_id"),
                        },
                    })
        return hits

    async def delete_by_document(self, document_id: int):
        """Delete all chunks for a document."""
        self._ensure_collection()
        try:
            self._collection.delete(f'document_id == {document_id}')
        except Exception as e:
            logger.warning(f"Failed to delete chunks for document {document_id}: {e}")


# Singleton
milvus_store = MilvusVectorStore()
