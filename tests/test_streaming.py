"""Tests for streaming API endpoints."""

from fastapi.testclient import TestClient

from denialops.main import app


def test_streaming_health_check() -> None:
    """Test streaming health endpoint."""
    with TestClient(app) as client:
        response = client.get("/api/v1/stream/health")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Check that we get SSE events
        content = response.text
        assert "event:" in content
        assert "data:" in content
        assert "streaming" in content


def test_streaming_completion_request_format() -> None:
    """Test streaming completion endpoint accepts proper request format."""
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/stream/completion",
            json={
                "prompt": "Test prompt",
                "max_tokens": 100,
            },
        )

        # Either returns 503 (no API key) or 200 (streaming response)
        # Both are valid depending on environment
        assert response.status_code in (200, 503)


def test_streaming_summary_case_not_found(client: TestClient) -> None:
    """Test streaming summary for nonexistent case."""
    response = client.post("/api/v1/cases/nonexistent-id/stream/summary")
    assert response.status_code == 404
