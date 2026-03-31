"""Pydantic schemas for API."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ============ Sessions ============

class SessionCreate(BaseModel):
    """Create session request."""
    title: Optional[str] = None


class SessionResponse(BaseModel):
    """Session response."""
    id: str
    title: Optional[str]
    created_at: str
    updated_at: str


class SessionList(BaseModel):
    """List sessions response."""
    sessions: list[SessionResponse]
    total: int


# ============ Messages ============

class MessageCreate(BaseModel):
    """Create message request."""
    content: str = Field(..., min_length=1)


class MessageResponse(BaseModel):
    """Message response."""
    id: str
    session_id: str
    role: str
    content: str
    created_at: str


class ChatResponse(BaseModel):
    """Chat response with user and assistant messages."""
    user_message: MessageResponse
    assistant_message: MessageResponse


class MessageHistory(BaseModel):
    """Message history response."""
    session_id: str
    messages: list[MessageResponse]


# ============ Knowledge Base ============

class IndexResponse(BaseModel):
    """Knowledge base index response."""
    indexed_files: int
    total_chunks: int
    status: str


# ============ Health ============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str