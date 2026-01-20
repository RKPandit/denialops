"""Task manager for background processing."""

import asyncio
import logging
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Status of a background task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a completed task."""

    task_id: str
    status: TaskStatus
    result: Any | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: float = 0.0
    progress_message: str = ""

    @property
    def duration_seconds(self) -> float | None:
        """Calculate task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "progress": self.progress,
            "progress_message": self.progress_message,
        }


class TaskManager:
    """
    Manager for background task execution.

    Provides:
    - Task submission and tracking
    - Progress updates
    - Result retrieval
    - Cancellation support
    """

    _instance: "TaskManager | None" = None
    _lock = Lock()

    def __new__(cls) -> "TaskManager":
        """Singleton pattern for task manager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        """Initialize the task manager."""
        if getattr(self, "_initialized", False):
            return

        self._tasks: dict[str, TaskResult] = {}
        self._running_tasks: dict[str, asyncio.Task[Any]] = {}
        self._task_lock = Lock()
        self._max_tasks = 100  # Maximum tasks to keep in memory
        self._initialized = True

    def submit(
        self,
        coro: Coroutine[Any, Any, Any],
        task_id: str | None = None,
    ) -> str:
        """
        Submit a coroutine for background execution.

        Args:
            coro: The coroutine to execute
            task_id: Optional task ID (generated if not provided)

        Returns:
            The task ID
        """
        task_id = task_id or str(uuid.uuid4())

        with self._task_lock:
            # Clean up old tasks if we have too many
            if len(self._tasks) >= self._max_tasks:
                self._cleanup_old_tasks()

            # Create task result
            result = TaskResult(
                task_id=task_id,
                status=TaskStatus.PENDING,
            )
            self._tasks[task_id] = result

        # Create and store the asyncio task
        async def run_task() -> None:
            try:
                with self._task_lock:
                    self._tasks[task_id].status = TaskStatus.RUNNING
                    self._tasks[task_id].started_at = datetime.now(timezone.utc)

                task_result = await coro

                with self._task_lock:
                    self._tasks[task_id].status = TaskStatus.COMPLETED
                    self._tasks[task_id].result = task_result
                    self._tasks[task_id].completed_at = datetime.now(timezone.utc)
                    self._tasks[task_id].progress = 1.0
                    self._tasks[task_id].progress_message = "Completed"

                logger.info(f"Task {task_id} completed successfully")

            except asyncio.CancelledError:
                with self._task_lock:
                    self._tasks[task_id].status = TaskStatus.CANCELLED
                    self._tasks[task_id].completed_at = datetime.now(timezone.utc)
                    self._tasks[task_id].error = "Task was cancelled"
                logger.info(f"Task {task_id} was cancelled")

            except Exception as e:
                with self._task_lock:
                    self._tasks[task_id].status = TaskStatus.FAILED
                    self._tasks[task_id].error = str(e)
                    self._tasks[task_id].completed_at = datetime.now(timezone.utc)
                logger.error(f"Task {task_id} failed: {e}")

            finally:
                with self._task_lock:
                    self._running_tasks.pop(task_id, None)

        # Schedule the task
        loop = asyncio.get_event_loop()
        asyncio_task = loop.create_task(run_task())

        with self._task_lock:
            self._running_tasks[task_id] = asyncio_task

        logger.info(f"Task {task_id} submitted")
        return task_id

    def get_status(self, task_id: str) -> TaskResult | None:
        """Get the status of a task."""
        with self._task_lock:
            return self._tasks.get(task_id)

    def update_progress(
        self,
        task_id: str,
        progress: float,
        message: str = "",
    ) -> None:
        """
        Update the progress of a running task.

        Args:
            task_id: The task ID
            progress: Progress value between 0.0 and 1.0
            message: Optional progress message
        """
        with self._task_lock:
            if task_id in self._tasks:
                self._tasks[task_id].progress = min(max(progress, 0.0), 1.0)
                self._tasks[task_id].progress_message = message

    def cancel(self, task_id: str) -> bool:
        """
        Cancel a running task.

        Returns:
            True if the task was cancelled, False if it couldn't be cancelled
        """
        with self._task_lock:
            task = self._running_tasks.get(task_id)
            if task and not task.done():
                task.cancel()
                return True
            return False

    def list_tasks(
        self,
        status: TaskStatus | None = None,
        limit: int = 50,
    ) -> list[TaskResult]:
        """
        List tasks, optionally filtered by status.

        Args:
            status: Filter by status (None for all)
            limit: Maximum number of tasks to return

        Returns:
            List of task results
        """
        with self._task_lock:
            tasks = list(self._tasks.values())

            if status:
                tasks = [t for t in tasks if t.status == status]

            # Sort by created_at descending
            tasks.sort(key=lambda t: t.created_at, reverse=True)

            return tasks[:limit]

    def _cleanup_old_tasks(self) -> None:
        """Remove old completed tasks to free memory."""
        # Keep only running/pending tasks and recent completed/failed ones
        completed_tasks = [
            (tid, t)
            for tid, t in self._tasks.items()
            if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
        ]

        # Sort by completion time and remove oldest
        completed_tasks.sort(
            key=lambda x: x[1].completed_at or x[1].created_at,
            reverse=True,
        )

        # Keep the 20 most recent completed tasks
        for tid, _ in completed_tasks[20:]:
            del self._tasks[tid]


# Global task manager instance
def get_task_manager() -> TaskManager:
    """Get the global task manager instance."""
    return TaskManager()
