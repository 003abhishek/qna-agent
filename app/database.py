import aiosqlite
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import get_settings

settings = get_settings()

# SQL Schema
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(file_name, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_file ON embeddings(file_name);
"""


async def init_db() -> None:
    """Initialize database with schema."""
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    
    async with aiosqlite.connect(settings.database_path) as db:
        await db.executescript(SCHEMA)
        await db.commit()


async def get_db() -> aiosqlite.Connection:
    """Get database connection."""
    db = await aiosqlite.connect(settings.database_path)
    db.row_factory = aiosqlite.Row
    return db


# ============ Sessions ============

async def create_session(session_id: str, title: Optional[str] = None) -> dict:
    """Create new chat session."""
    now = datetime.utcnow().isoformat()
    
    async with aiosqlite.connect(settings.database_path) as db:
        await db.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title, now, now)
        )
        await db.commit()
    
    return {"id": session_id, "title": title, "created_at": now, "updated_at": now}


async def get_session(session_id: str) -> Optional[dict]:
    """Get session by ID."""
    async with aiosqlite.connect(settings.database_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def list_sessions(limit: int = 50, offset: int = 0) -> list[dict]:
    """List all sessions."""
    async with aiosqlite.connect(settings.database_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def delete_session(session_id: str) -> bool:
    """Delete session and its messages."""
    async with aiosqlite.connect(settings.database_path) as db:
        await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor = await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await db.commit()
        return cursor.rowcount > 0


# ============ Messages ============

async def create_message(msg_id: str, session_id: str, role: str, content: str) -> dict:
    """Create new message."""
    now = datetime.utcnow().isoformat()
    
    async with aiosqlite.connect(settings.database_path) as db:
        await db.execute(
            "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (msg_id, session_id, role, content, now)
        )
        await db.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now, session_id)
        )
        await db.commit()
    
    return {
        "id": msg_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "created_at": now
    }


async def get_messages(session_id: str) -> list[dict]:
    """Get all messages for a session."""
    async with aiosqlite.connect(settings.database_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


# ============ Embeddings ============

async def save_embedding(
    emb_id: str,
    file_name: str,
    chunk_index: int,
    content: str,
    embedding: list[float]
) -> None:
    """Save document embedding."""
    now = datetime.utcnow().isoformat()
    embedding_json = json.dumps(embedding)
    
    async with aiosqlite.connect(settings.database_path) as db:
        await db.execute(
            """INSERT OR REPLACE INTO embeddings 
               (id, file_name, chunk_index, content, embedding, created_at) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (emb_id, file_name, chunk_index, content, embedding_json, now)
        )
        await db.commit()


async def get_all_embeddings() -> list[dict]:
    """Get all embeddings for similarity search."""
    async with aiosqlite.connect(settings.database_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT id, file_name, chunk_index, content, embedding FROM embeddings"
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["embedding"] = json.loads(d["embedding"])
            results.append(d)
        return results


async def get_embedding_count() -> int:
    """Count embeddings."""
    async with aiosqlite.connect(settings.database_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM embeddings")
        row = await cursor.fetchone()
        return row[0]


async def clear_embeddings() -> None:
    """Clear all embeddings."""
    async with aiosqlite.connect(settings.database_path) as db:
        await db.execute("DELETE FROM embeddings")
        await db.commit()