"""Tests for background task processing."""

import asyncio

import pytest

from denialops.tasks.manager import TaskManager, TaskResult, TaskStatus


class TestTaskResult:
    """Tests for TaskResult."""

    def test_creates_result(self) -> None:
        """Test task result creation."""
        result = TaskResult(
            task_id="test-123",
            status=TaskStatus.PENDING,
        )

        assert result.task_id == "test-123"
        assert result.status == TaskStatus.PENDING
        assert result.result is None
        assert result.error is None
        assert result.progress == 0.0

    def test_duration_calculation(self) -> None:
        """Test duration calculation."""
        from datetime import datetime, timedelta, timezone

        start = datetime.now(timezone.utc)
        end = start + timedelta(seconds=5)

        result = TaskResult(
            task_id="test-123",
            status=TaskStatus.COMPLETED,
            started_at=start,
            completed_at=end,
        )

        assert result.duration_seconds == pytest.approx(5.0, rel=0.01)

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = TaskResult(
            task_id="test-123",
            status=TaskStatus.RUNNING,
            progress=0.5,
            progress_message="Processing",
        )

        data = result.to_dict()
        assert data["task_id"] == "test-123"
        assert data["status"] == "running"
        assert data["progress"] == 0.5
        assert data["progress_message"] == "Processing"


class TestTaskManager:
    """Tests for TaskManager."""

    @pytest.fixture
    def manager(self) -> TaskManager:
        """Create a fresh task manager for each test."""
        # Reset singleton for testing
        TaskManager._instance = None
        return TaskManager()

    def test_singleton_pattern(self) -> None:
        """Test that TaskManager is a singleton."""
        TaskManager._instance = None
        manager1 = TaskManager()
        manager2 = TaskManager()
        assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_submit_task(self, manager: TaskManager) -> None:
        """Test submitting a task."""

        async def simple_task() -> str:
            return "completed"

        task_id = manager.submit(simple_task())

        assert task_id is not None
        status = manager.get_status(task_id)
        assert status is not None
        assert status.status in (TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.COMPLETED)

    @pytest.mark.asyncio
    async def test_task_completes(self, manager: TaskManager) -> None:
        """Test that a task completes successfully."""

        async def simple_task() -> str:
            await asyncio.sleep(0.1)
            return "done"

        task_id = manager.submit(simple_task())

        # Wait for completion
        await asyncio.sleep(0.2)

        status = manager.get_status(task_id)
        assert status is not None
        assert status.status == TaskStatus.COMPLETED
        assert status.result == "done"

    @pytest.mark.asyncio
    async def test_task_failure(self, manager: TaskManager) -> None:
        """Test that task failures are captured."""

        async def failing_task() -> str:
            await asyncio.sleep(0.1)
            raise ValueError("Task failed!")

        task_id = manager.submit(failing_task())

        # Wait for failure
        await asyncio.sleep(0.2)

        status = manager.get_status(task_id)
        assert status is not None
        assert status.status == TaskStatus.FAILED
        assert "Task failed!" in (status.error or "")

    @pytest.mark.asyncio
    async def test_progress_updates(self, manager: TaskManager) -> None:
        """Test progress updates during task execution."""
        task_id = "progress-test"

        async def task_with_progress() -> str:
            manager.update_progress(task_id, 0.25, "Step 1")
            await asyncio.sleep(0.05)
            manager.update_progress(task_id, 0.5, "Step 2")
            await asyncio.sleep(0.05)
            manager.update_progress(task_id, 0.75, "Step 3")
            return "done"

        manager.submit(task_with_progress(), task_id=task_id)

        # Check intermediate progress
        await asyncio.sleep(0.075)
        status = manager.get_status(task_id)
        assert status is not None
        # Progress should be between 0.25 and 0.75 at this point

        # Wait for completion
        await asyncio.sleep(0.2)
        status = manager.get_status(task_id)
        assert status is not None
        assert status.progress == 1.0

    @pytest.mark.asyncio
    async def test_cancel_task(self, manager: TaskManager) -> None:
        """Test cancelling a running task."""

        async def long_task() -> str:
            await asyncio.sleep(10)
            return "done"

        task_id = manager.submit(long_task())

        # Give it time to start
        await asyncio.sleep(0.1)

        # Cancel
        cancelled = manager.cancel(task_id)
        assert cancelled is True

        # Wait for cancellation to complete
        await asyncio.sleep(0.1)

        status = manager.get_status(task_id)
        assert status is not None
        assert status.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_list_tasks(self, manager: TaskManager) -> None:
        """Test listing tasks."""

        async def quick_task() -> str:
            return "done"

        # Submit multiple tasks
        for _ in range(3):
            manager.submit(quick_task())

        # Wait for completion
        await asyncio.sleep(0.2)

        # List all tasks
        tasks = manager.list_tasks()
        assert len(tasks) >= 3

        # List only completed tasks
        completed = manager.list_tasks(status=TaskStatus.COMPLETED)
        for task in completed:
            assert task.status == TaskStatus.COMPLETED

    def test_get_nonexistent_task(self, manager: TaskManager) -> None:
        """Test getting a nonexistent task."""
        status = manager.get_status("nonexistent-task")
        assert status is None

    def test_cancel_nonexistent_task(self, manager: TaskManager) -> None:
        """Test cancelling a nonexistent task."""
        cancelled = manager.cancel("nonexistent-task")
        assert cancelled is False
