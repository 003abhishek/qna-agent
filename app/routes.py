"""API Routes."""

import uuid
from fastapi import APIRouter, HTTPException

from app import database as db
from app import rag
from app import llm
from app.schemas import (
    SessionCreate,
    SessionResponse,
    SessionList,
    MessageCreate,
    MessageResponse,
    ChatResponse,
    MessageHistory,
    IndexResponse,
    HealthResponse
)

router = APIRouter()


# ============ Health ============

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", version="0.1.0")


# ============ Sessions ============

@router.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(data: SessionCreate):
    """Create a new chat session."""
    session_id = str(uuid.uuid4())
    session = await db.create_session(session_id, data.title)
    return SessionResponse(**session)


@router.get("/sessions", response_model=SessionList)
async def list_sessions(limit: int = 50, offset: int = 0):
    """List all chat sessions."""
    sessions = await db.list_sessions(limit, offset)
    return SessionList(
        sessions=[SessionResponse(**s) for s in sessions],
        total=len(sessions)
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get a specific session."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(**session)


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    """Delete a session."""
    deleted = await db.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return None


# ============ Messages ============

@router.get("/sessions/{session_id}/messages", response_model=MessageHistory)
async def get_messages(session_id: str):
    """Get message history for a session."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = await db.get_messages(session_id)
    return MessageHistory(
        session_id=session_id,
        messages=[MessageResponse(**m) for m in messages]
    )


@router.post("/sessions/{session_id}/messages", response_model=ChatResponse, status_code=201)
async def send_message(session_id: str, data: MessageCreate):
    """Send a message and get AI response."""
    # Check session exists
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get conversation history
    history = await db.get_messages(session_id)
    
    # Save user message
    user_msg_id = str(uuid.uuid4())
    user_msg = await db.create_message(
        msg_id=user_msg_id,
        session_id=session_id,
        role="user",
        content=data.content
    )
    
    # Generate AI response
    response_content = await llm.generate_response(
        user_message=data.content,
        conversation_history=history
    )
    
    # Save assistant message
    assistant_msg_id = str(uuid.uuid4())
    assistant_msg = await db.create_message(
        msg_id=assistant_msg_id,
        session_id=session_id,
        role="assistant",
        content=response_content
    )
    
    return ChatResponse(
        user_message=MessageResponse(**user_msg),
        assistant_message=MessageResponse(**assistant_msg)
    )


# ============ Knowledge Base ============

@router.post("/index", response_model=IndexResponse)
async def index_knowledge_base():
    """Index all documents in knowledge base."""
    result = await rag.index_knowledge_base()
    return IndexResponse(**result)


@router.get("/index/status", response_model=IndexResponse)
async def get_index_status():
    """Get current index status."""
    count = await db.get_embedding_count()
    return IndexResponse(
        indexed_files=0,
        total_chunks=count,
        status="ready" if count > 0 else "empty"
    )