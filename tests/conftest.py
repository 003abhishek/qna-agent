"""Test fixtures."""

import os
import pytest
from pathlib import Path
from httpx import AsyncClient, ASGITransport

# -------------------------
# Force Ollama for tests
# -------------------------
os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["LLM_MODEL"] = "llama3.2"
os.environ["EMBEDDING_MODEL"] = "nomic-embed-text"

# Use a separate test database
os.environ["DATABASE_PATH"] = "./data/test_qna.db"
os.environ["KNOWLEDGE_PATH"] = "./knowledge"

from app.main import app
from app.database import init_db


@pytest.fixture(scope="session", autouse=True)
async def setup_database():
    """Initialize and cleanup test database."""
    await init_db()
    yield
    db_path = Path("./data/test_qna.db")
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
async def client():
    """Async HTTP client for API tests."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac