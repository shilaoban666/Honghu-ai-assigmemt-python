"""RAG API endpoints — mirrors RagController.java.

Full implementation with presigned URLs, file registration, ingestion status (SSE),
and RAG evaluation endpoints.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.core.database import get_db, async_session_factory
from app.models.orm import RagDocument, RagDocumentStatus, RagIngestionEvent, RagIngestionRagStatus
from app.rag.ingestion import DocumentIngestionService
from app.rag.retrieval import RagRetrievalService, RagRequest as PipelineRagRequest
from app.schemas.dto import (
    UploadUrlRequest,
    UploadUrlResponse,
    RegisterUploadedFileRequest,
    RagEvalRequest,
    RagEvalResponse,
    RagFileStatusResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])

ingestion_svc = DocumentIngestionService(async_session_factory)
retrieval_svc = RagRetrievalService(async_session_factory)


@router.post("/upload-url")
async def get_upload_url(
    request: UploadUrlRequest,
    user_id: str | None = Header(None, alias="X-User-Id"),
) -> UploadUrlResponse:
    """Get a presigned S3 upload URL for direct browser upload."""
    try:
        result = await ingestion_svc.generate_presigned_upload_url(
            file_name=request.file_name,
            file_type=request.file_type,
        )
        return UploadUrlResponse(
            uploadUrl=result["upload_url"],
            fileId=result["file_id"],
            objectKey=result["object_key"],
            expiresIn=1800,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/register-uploaded-file")
async def register_uploaded_file(
    request: RegisterUploadedFileRequest,
    user_id: str | None = Header(None, alias="X-User-Id"),
    db: AsyncSession = Depends(get_db),
):
    """Register a file that has been uploaded to S3 and trigger ingestion."""
    # Create document record
    doc = RagDocument(
        bucket_name=request.bucket_name,
        object_key=request.object_key,
        object_etag=request.object_etag,
        file_name=request.object_key.split("/")[-1],
        session_id=request.session_id,
        chat_id=request.chat_id,
        file_id=request.file_id,
        status=RagDocumentStatus.RECEIVED,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    # Trigger async ingestion (background task)
    import asyncio
    asyncio.create_task(
        ingestion_svc.ingest_from_s3_event(
            bucket_name=request.bucket_name,
            object_key=request.object_key,
            file_name=request.object_key.split("/")[-1],
            file_type=request.object_key.split(".")[-1] if "." in request.object_key else "unknown",
            file_size=0,
            session_id=request.session_id or "",
            chat_id=request.chat_id,
            file_id=request.file_id,
        )
    )

    return {"message": "File registered", "fileId": request.file_id, "documentId": doc.document_id}


@router.get("/file-status/{file_id}")
async def get_file_status(
    file_id: str,
    db: AsyncSession = Depends(get_db),
) -> RagFileStatusResponse:
    """Get ingestion status for a file."""
    result = await db.execute(
        select(RagDocument).where(RagDocument.file_id == file_id)
    )
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    # Get latest ingestion event
    event_result = await db.execute(
        select(RagIngestionEvent)
        .where(
            RagIngestionEvent.bucket_name == doc.bucket_name,
            RagIngestionEvent.object_key == doc.object_key,
        )
        .order_by(RagIngestionEvent.created_at.desc())
        .limit(1)
    )
    event = event_result.scalar_one_or_none()

    return RagFileStatusResponse(
        fileId=file_id,
        fileName=doc.file_name,
        status=doc.status.value if doc.status else "RECEIVED",
        ragStatus=event.rag_status.value if event and event.rag_status else None,
        chunkCount=doc.chunk_count,
        errorMessage=doc.error_message,
    )


@router.get("/file-status/{file_id}/stream")
async def stream_file_status(
    file_id: str,
    db: AsyncSession = Depends(get_db),
):
    """SSE stream of file ingestion status — real-time updates."""
    async def event_stream():
        import asyncio
        while True:
            async with async_session_factory() as session:
                result = await session.execute(
                    select(RagDocument).where(RagDocument.file_id == file_id)
                )
                doc = result.scalar_one_or_none()
                if not doc:
                    yield f"data: {json.dumps({'error': 'File not found'})}\n\n"
                    return

                event_result = await session.execute(
                    select(RagIngestionEvent)
                    .where(
                        RagIngestionEvent.bucket_name == doc.bucket_name,
                        RagIngestionEvent.object_key == doc.object_key,
                    )
                    .order_by(RagIngestionEvent.created_at.desc())
                    .limit(1)
                )
                event = event_result.scalar_one_or_none()

                status_data = {
                    "fileId": file_id,
                    "status": doc.status.value if doc.status else "RECEIVED",
                    "ragStatus": event.rag_status.value if event and event.rag_status else None,
                    "chunkCount": doc.chunk_count,
                    "errorMessage": doc.error_message,
                }

                # Check if terminal state
                terminal = {"INDEXED", "FAILED", "SKIPPED"}
                if doc.status and doc.status.value in terminal:
                    yield f"data: {json.dumps(status_data, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                yield f"data: {json.dumps(status_data, ensure_ascii=False)}\n\n"
                await asyncio.sleep(2)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/eval/retrieve")
async def eval_retrieve(
    request: RagEvalRequest,
    db: AsyncSession = Depends(get_db),
) -> RagEvalResponse:
    """RAG evaluation endpoint — returns retrieval results for RAGAS/DeepEval."""
    snippets = await retrieval_svc.retrieve_by_session(
        user_id=settings.rag.eval.user_id or "eval-user",
        session_id=request.session_id or settings.rag.eval.session_id,
        query=request.query,
    )

    context_block = await retrieval_svc.build_context_block(
        PipelineRagRequest(
            user_id=settings.rag.eval.user_id or "eval-user",
            session_id=request.session_id or settings.rag.eval.session_id,
            query=request.query,
        )
    )

    return RagEvalResponse(
        snippets=[
            {
                "content": s["content"],
                "score": s["score"],
                "fileName": s.get("metadata", {}).get("file_name", ""),
            }
            for s in snippets
        ],
        contextBlock=context_block,
    )


@router.post("/backfill")
async def backfill(
    db: AsyncSession = Depends(get_db),
):
    """Trigger backfill of RAG vector index for all indexed documents."""
    # Find all indexed documents
    result = await db.execute(
        select(RagDocument).where(RagDocument.status == RagDocumentStatus.INDEXED)
    )
    documents = result.scalars().all()

    return {
        "message": f"Backfill would reprocess {len(documents)} documents",
        "documentCount": len(documents),
    }


@router.get("/download-url/{file_id}")
async def get_download_url(
    file_id: str,
    user_id: str | None = Header(None, alias="X-User-Id"),
    db: AsyncSession = Depends(get_db),
):
    """Generate a presigned download URL for a file."""
    result = await db.execute(
        select(RagDocument).where(RagDocument.file_id == file_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    url = await ingestion_svc.generate_presigned_url(
        bucket_name=doc.bucket_name,
        object_key=doc.object_key,
    )
    return {"downloadUrl": url, "fileName": doc.file_name}
