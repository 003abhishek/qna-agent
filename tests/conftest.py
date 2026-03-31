"""Test fixtures."""

import os
import pytest
import asyncio
from pathlib import Path
from httpx import AsyncClient, ASGITransport

# Set test environment before importing app
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "test-key")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
os.environ["DATABASE_PATH"] = "./data/test_qna.db"
os.environ["KNOWLEDGE_PATH"] = "./knowledge"

from app.main import app
from app.database import init_db


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
async def setup_database():
    """Initialize test database."""
    await init_db()
    yield
    # Cleanup: remove test database
    db_path = Path("./data/test_qna.db")
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
async def client():
    """Async HTTP client for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac