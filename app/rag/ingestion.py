"""RAG Document Ingestion Service — full ingestion pipeline.

Mirrors DocumentIngestionService.java.
Pipeline: S3 download → parse → clean → split → embed → Milvus insert → update status.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.orm import RagDocument, RagDocumentChunk, RagDocumentStatus, RagIngestionEvent, RagFileStatus, RagIngestionRagStatus
from app.rag.parser import document_parser
from app.rag.splitter import TextSplitter
from app.rag.embedding import embedding_service
from app.rag.vectorstore import milvus_store

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    """Full document ingestion pipeline for RAG."""

    SUPPORTED_EXTENSIONS = set(settings.rag.ingestion.supported_extensions)

    def __init__(self, db_session_factory):
        self.db_factory = db_session_factory

    async def ingest_from_s3_event(
        self,
        bucket_name: str,
        object_key: str,
        file_name: str,
        file_type: str,
        file_size: int,
        session_id: str = "",
        chat_id: int | None = None,
        file_id: str = "",
    ) -> dict:
        """Full ingestion triggered by S3 upload event."""
        async with self.db_factory() as db:
            # Create or update document record
            doc = await self._ensure_document(
                db, bucket_name, object_key, file_name, file_type, file_size,
                session_id, chat_id, file_id,
            )

            # Create ingestion event
            event = RagIngestionEvent(
                deduplication_key=f"{bucket_name}/{object_key}",
                bucket_name=bucket_name,
                object_key=object_key,
                event_name="s3:ObjectCreated:Put",
                file_status=RagFileStatus.RECEIVED,
                rag_status=RagIngestionRagStatus.RECEIVED,
            )
            db.add(event)
            await db.commit()

            try:
                # Step 1: Download from S3 (mock for now)
                doc.status = RagDocumentStatus.PROCESSING
                event.rag_status = RagIngestionRagStatus.DOWNLOADING
                await db.commit()

                file_bytes = await self._download_from_s3(bucket_name, object_key)

                # Step 2: Extract text
                event.rag_status = RagIngestionRagStatus.EXTRACTING
                await db.commit()
                text = await document_parser.parse(file_bytes, file_name, file_type)

                if not text or not text.strip():
                    doc.status = RagDocumentStatus.SKIPPED
                    event.rag_status = RagIngestionRagStatus.SKIPPED
                    event.file_status = RagFileStatus.SKIPPED
                    await db.commit()
                    return {"status": "skipped", "reason": "No extractable text"}

                max_chars = settings.rag.ingestion.max_extracted_characters
                if len(text) > max_chars:
                    text = text[:max_chars]
                doc.extracted_character_count = len(text)

                # Step 3: Split into chunks
                event.rag_status = RagIngestionRagStatus.CHUNKING
                await db.commit()
                splitter = TextSplitter(
                    chunk_size=settings.rag.ingestion.chunk_size,
                    chunk_overlap=settings.rag.ingestion.chunk_overlap,
                    min_chunk_length=settings.rag.ingestion.min_chunk_length,
                    max_chunks=settings.rag.ingestion.max_chunks_per_document,
                )
                chunks = splitter.split(text, strategy="recursive", metadata={
                    "file_name": file_name,
                    "file_type": file_type,
                    "document_id": doc.document_id,
                })

                if not chunks:
                    doc.status = RagDocumentStatus.SKIPPED
                    event.rag_status = RagIngestionRagStatus.SKIPPED
                    event.file_status = RagFileStatus.SKIPPED
                    await db.commit()
                    return {"status": "skipped", "reason": "No valid chunks"}

                # Step 4: Save chunks to database
                for chunk in chunks:
                    chunk_entity = RagDocumentChunk(
                        document_id=doc.document_id,
                        chunk_index=chunk["chunk_index"],
                        content=chunk["content"],
                        char_count=chunk["char_count"],
                        token_estimate=chunk["token_estimate"],
                        metadata_=chunk["metadata"],
                    )
                    db.add(chunk_entity)
                doc.chunk_count = len(chunks)

                # Step 5: Generate embeddings
                event.rag_status = RagIngestionRagStatus.EMBEDDING
                await db.commit()
                chunk_texts = [c["content"] for c in chunks]
                embeddings = await embedding_service.embed_texts(chunk_texts)

                # Step 6: Insert into Milvus
                event.rag_status = RagIngestionRagStatus.INDEXING
                await db.commit()
                inserted = await milvus_store.insert_chunks(
                    chunks=chunks,
                    embeddings=embeddings,
                    document_id=doc.document_id,
                    file_name=file_name,
                    file_type=file_type,
                    session_id=session_id,
                    chat_id=chat_id,
                    file_id=file_id,
                )

                # Step 7: Mark as indexed
                doc.status = RagDocumentStatus.INDEXED
                doc.last_indexed_at = datetime.now(timezone.utc)
                event.rag_status = RagIngestionRagStatus.SUCCESS
                event.file_status = RagFileStatus.SUCCESS
                event.processed_at = datetime.now(timezone.utc)
                await db.commit()

                logger.info(f"Document {doc.document_id} ingested: {len(chunks)} chunks, {inserted} vectors")
                return {
                    "status": "indexed",
                    "document_id": doc.document_id,
                    "chunks": len(chunks),
                    "vectors": inserted,
                }

            except Exception as e:
                logger.error(f"Ingestion failed for {bucket_name}/{object_key}: {e}")
                doc.status = RagDocumentStatus.FAILED
                doc.error_message = str(e)
                event.rag_status = RagIngestionRagStatus.FAILED
                event.file_status = RagFileStatus.FAILED
                event.error_message = str(e)
                await db.commit()
                return {"status": "failed", "error": str(e)}

    async def _ensure_document(
        self,
        db: AsyncSession,
        bucket_name: str,
        object_key: str,
        file_name: str,
        file_type: str,
        file_size: int,
        session_id: str,
        chat_id: int | None,
        file_id: str,
    ) -> RagDocument:
        """Get or create document record."""
        result = await db.execute(
            select(RagDocument).where(
                RagDocument.bucket_name == bucket_name,
                RagDocument.object_key == object_key,
            )
        )
        doc = result.scalar_one_or_none()

        if doc:
            doc.status = RagDocumentStatus.RECEIVED
            doc.file_name = file_name
            doc.file_size = file_size
        else:
            doc = RagDocument(
                bucket_name=bucket_name,
                object_key=object_key,
                file_name=file_name,
                file_type=file_type,
                file_size=file_size,
                session_id=session_id,
                chat_id=chat_id,
                file_id=file_id,
                status=RagDocumentStatus.RECEIVED,
            )
            db.add(doc)

        await db.commit()
        await db.refresh(doc)
        return doc

    async def _download_from_s3(self, bucket_name: str, object_key: str) -> bytes:
        """Download file bytes from S3."""
        import boto3
        from app.core.config import settings as cfg

        if not cfg.aws.s3.enabled:
            # Mock for local development
            raise NotImplementedError("S3 download requires AWS S3 to be enabled")

        s3 = boto3.client(
            "s3",
            region_name=cfg.aws.region,
        )
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        return response["Body"].read()

    async def generate_presigned_url(
        self, bucket_name: str, object_key: str, expiration: int = 1800
    ) -> str:
        """Generate a presigned S3 download URL."""
        import boto3
        from app.core.config import settings as cfg

        s3 = boto3.client("s3", region_name=cfg.aws.region)
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_key},
            ExpiresIn=expiration,
        )

    async def generate_presigned_upload_url(
        self, file_name: str, file_type: str, expiration: int = 1800
    ) -> dict:
        """Generate a presigned S3 upload URL."""
        import boto3
        from app.core.config import settings as cfg

        file_id = str(uuid.uuid4())
        object_key = f"uploads/{file_id}/{file_name}"

        s3 = boto3.client("s3", region_name=cfg.aws.region)
        url = s3.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": cfg.aws.s3.uploaded_bucket,
                "Key": object_key,
                "ContentType": file_type,
            },
            ExpiresIn=expiration,
        )

        return {
            "upload_url": url,
            "file_id": file_id,
            "object_key": object_key,
            "bucket_name": cfg.aws.s3.uploaded_bucket,
        }
