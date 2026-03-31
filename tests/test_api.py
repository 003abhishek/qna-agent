"""API tests."""

import pytest
import os


class TestHealth:
    """Health endpoint tests."""

    async def test_health_check(self, client):
        """Test health endpoint."""
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"

    async def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "QnA Agent API"


class TestSessions:
    """Session endpoint tests."""

    async def test_create_session(self, client):
        """Test creating a session."""
        response = await client.post(
            "/api/v1/sessions",
            json={"title": "Test Session"}
        )
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Test Session"
        assert "id" in data
        assert "created_at" in data

    async def test_create_session_no_title(self, client):
        """Test creating session without title."""
        response = await client.post(
            "/api/v1/sessions",
            json={}
        )
        assert response.status_code == 201
        data = response.json()
        assert data["title"] is None

    async def test_list_sessions(self, client):
        """Test listing sessions."""
        response = await client.get("/api/v1/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "total" in data

    async def test_get_session(self, client):
        """Test getting a specific session."""
        # Create session first
        create_response = await client.post(
            "/api/v1/sessions",
            json={"title": "Get Test"}
        )
        session_id = create_response.json()["id"]

        # Get session
        response = await client.get(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == session_id
        assert data["title"] == "Get Test"

    async def test_get_session_not_found(self, client):
        """Test getting non-existent session."""
        response = await client.get("/api/v1/sessions/non-existent-id")
        assert response.status_code == 404

    async def test_delete_session(self, client):
        """Test deleting a session."""
        # Create session
        create_response = await client.post(
            "/api/v1/sessions",
            json={"title": "Delete Test"}
        )
        session_id = create_response.json()["id"]

        # Delete session
        response = await client.delete(f"/api/v1/sessions/{session_id}")
        assert response.status_code == 204

        # Verify deleted
        get_response = await client.get(f"/api/v1/sessions/{session_id}")
        assert get_response.status_code == 404


class TestMessages:
    """Message endpoint tests."""

    async def test_get_messages_empty(self, client):
        """Test getting messages from empty session."""
        # Create session
        create_response = await client.post(
            "/api/v1/sessions",
            json={"title": "Messages Test"}
        )
        session_id = create_response.json()["id"]

        # Get messages
        response = await client.get(f"/api/v1/sessions/{session_id}/messages")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert data["messages"] == []

    async def test_get_messages_not_found(self, client):
        """Test getting messages from non-existent session."""
        response = await client.get("/api/v1/sessions/non-existent/messages")
        assert response.status_code == 404


class TestIndex:
    """Knowledge base index tests."""

    async def test_get_index_status(self, client):
        """Test getting index status."""
        response = await client.get("/api/v1/index/status")
        assert response.status_code == 200
        data = response.json()
        assert "total_chunks" in data
        assert "status" in data


# Integration tests - require real API key
@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY", "test-key") == "test-key",
    reason="Requires valid OPENAI_API_KEY"
)
class TestIntegration:
    """Integration tests with real LLM calls."""

    async def test_index_knowledge_base(self, client):
        """Test indexing knowledge base."""
        response = await client.post("/api/v1/index")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["indexed_files"] > 0

    async def test_send_message(self, client):
        """Test sending message and getting AI response."""
        # Index first
        await client.post("/api/v1/index")

        # Create session
        session_response = await client.post(
            "/api/v1/sessions",
            json={"title": "Chat Test"}
        )
        session_id = session_response.json()["id"]

        # Send message
        response = await client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"content": "What is the company name?"}
        )
        assert response.status_code == 201
        data = response.json()
        assert "user_message" in data
        assert "assistant_message" in data
        assert data["user_message"]["role"] == "user"
        assert data["assistant_message"]["role"] == "assistant"