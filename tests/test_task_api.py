"""Tests for background task API endpoints."""

from fastapi.testclient import TestClient


def test_start_async_pipeline_no_case(client: TestClient) -> None:
    """Test starting async pipeline for nonexistent case."""
    response = client.post("/api/v1/cases/nonexistent-id/run/async")
    assert response.status_code == 404


def test_get_task_status_not_found(client: TestClient) -> None:
    """Test getting status of nonexistent task."""
    response = client.get("/api/v1/tasks/nonexistent-task")
    assert response.status_code == 404


def test_cancel_task_not_found(client: TestClient) -> None:
    """Test cancelling nonexistent task."""
    response = client.post("/api/v1/tasks/nonexistent-task/cancel")
    assert response.status_code == 404


def test_list_tasks(client: TestClient) -> None:
    """Test listing tasks."""
    response = client.get("/api/v1/tasks")
    assert response.status_code == 200
    data = response.json()
    assert "tasks" in data
    assert "count" in data
    assert isinstance(data["tasks"], list)


def test_start_async_pipeline_requires_denial_letter(client: TestClient) -> None:
    """Test that starting async pipeline requires a denial letter."""
    # Create a case
    response = client.post("/api/v1/cases", json={"mode": "fast"})
    assert response.status_code == 201
    case_id = response.json()["case_id"]

    # Try to start pipeline without denial letter
    response = client.post(f"/api/v1/cases/{case_id}/run/async")
    assert response.status_code == 400
    assert "denial letter" in response.json()["detail"].lower()
