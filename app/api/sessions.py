"""Chat session API endpoints — mirrors ChatSessionController.java."""

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.orm import ChatSession, ChatMessage

router = APIRouter(prefix="/api/v1/sessions", tags=["Sessions"])


@router.get("")
async def list_sessions(
    user_id: str | None = Header(None, alias="X-User-Id"),
    db: AsyncSession = Depends(get_db),
):
    """List sessions for a user."""
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-User-Id header")

    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatSession.created_at.desc())
    )
    sessions = result.scalars().all()
    return [
        {
            "sessionId": s.session_id,
            "sessionName": s.session_name,
            "title": s.title,
            "sessionStatus": s.session_status,
            "createdAt": s.created_at.isoformat() if s.created_at else None,
        }
        for s in sessions
    ]


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get session details."""
    result = await db.execute(select(ChatSession).where(ChatSession.session_id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    return {
        "sessionId": session.session_id,
        "userId": session.user_id,
        "userName": session.user_name,
        "systemRole": session.system_role,
        "sessionName": session.session_name,
        "sessionStatus": session.session_status,
        "title": session.title,
        "createdAt": session.created_at.isoformat() if session.created_at else None,
    }


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a session and its messages."""
    await db.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
    await db.execute(delete(ChatSession).where(ChatSession.session_id == session_id))
    await db.commit()
    return {"message": "Session deleted"}
