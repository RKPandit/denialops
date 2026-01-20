"""Tests for API endpoints."""

from fastapi.testclient import TestClient


def test_health_check(client: TestClient) -> None:
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_health_detailed(client: TestClient) -> None:
    """Test detailed health check endpoint."""
    response = client.get("/health/detailed")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "unhealthy"]
    assert "checks" in data
    assert "storage" in data["checks"]


def test_health_live(client: TestClient) -> None:
    """Test liveness probe endpoint."""
    response = client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"
    assert "uptime_seconds" in data


def test_health_ready(client: TestClient) -> None:
    """Test readiness probe endpoint."""
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ready", "not_ready"]


def test_create_case(client: TestClient) -> None:
    """Test case creation."""
    response = client.post("/api/v1/cases", json={"mode": "fast"})
    assert response.status_code == 201
    data = response.json()
    assert "case_id" in data
    assert data["mode"] == "fast"
    assert "created_at" in data


def test_create_case_verified_mode(client: TestClient) -> None:
    """Test case creation in verified mode."""
    response = client.post("/api/v1/cases", json={"mode": "verified"})
    assert response.status_code == 201
    data = response.json()
    assert data["mode"] == "verified"


def test_create_case_with_context(client: TestClient) -> None:
    """Test case creation with user context."""
    response = client.post(
        "/api/v1/cases",
        json={
            "mode": "fast",
            "user_context": {
                "payer_name": "Blue Cross",
                "state": "NC",
            },
        },
    )
    assert response.status_code == 201


def test_get_nonexistent_case(client: TestClient) -> None:
    """Test accessing a case that doesn't exist."""
    response = client.get("/api/v1/cases/nonexistent-id/artifacts")
    assert response.status_code == 404


def test_upload_document_no_case(client: TestClient) -> None:
    """Test uploading to a case that doesn't exist."""
    response = client.post(
        "/api/v1/cases/nonexistent-id/documents",
        files={"file": ("test.txt", b"test content", "text/plain")},
        data={"doc_type": "denial_letter"},
    )
    assert response.status_code == 404
