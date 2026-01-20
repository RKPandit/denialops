"""API endpoints for background task management."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from denialops.api.dependencies import CaseIdDep, SettingsDep, StorageDep
from denialops.tasks import PipelineTask, TaskManager, TaskStatus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tasks"])


class StartTaskResponse(BaseModel):
    """Response from starting a background task."""

    task_id: str = Field(..., description="Task ID for tracking")
    case_id: str = Field(..., description="Case being processed")
    status: str = Field(..., description="Initial task status")


class TaskStatusResponse(BaseModel):
    """Response with task status."""

    task_id: str
    status: str
    progress: float = Field(..., description="Progress from 0.0 to 1.0")
    progress_message: str = Field("", description="Current stage description")
    result: dict[str, Any] | None = Field(None, description="Task result if completed")
    error: str | None = Field(None, description="Error message if failed")
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None


class TaskListResponse(BaseModel):
    """Response listing tasks."""

    tasks: list[TaskStatusResponse]
    count: int


def get_task_manager() -> TaskManager:
    """Get the task manager instance."""
    return TaskManager()


@router.post(
    "/cases/{case_id}/run/async",
    response_model=StartTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_pipeline_async(
    case_id: CaseIdDep,
    storage: StorageDep,
    settings: SettingsDep,
) -> StartTaskResponse:
    """
    Start the processing pipeline as a background task.

    Returns immediately with a task ID that can be used to track progress.
    """
    # Verify case exists
    metadata = storage.get_artifact(case_id, "metadata.json")
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found",
        )

    # Check for denial letter
    documents = storage.list_documents(case_id)
    denial_doc = next(
        (d for d in documents if d.get("doc_type") == "denial_letter"),
        None,
    )
    if not denial_doc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No denial letter uploaded. Upload a denial letter first.",
        )

    # Start the pipeline task
    pipeline_task = PipelineTask(
        case_id=case_id,
        storage=storage,
        settings=settings,
    )
    task_id = pipeline_task.start()

    return StartTaskResponse(
        task_id=task_id,
        case_id=case_id,
        status="pending",
    )


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """Get the status of a background task."""
    manager = get_task_manager()
    result = manager.get_status(task_id)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )

    return TaskStatusResponse(
        task_id=result.task_id,
        status=result.status.value,
        progress=result.progress,
        progress_message=result.progress_message,
        result=result.result if isinstance(result.result, dict) else None,
        error=result.error,
        created_at=result.created_at.isoformat(),
        started_at=result.started_at.isoformat() if result.started_at else None,
        completed_at=result.completed_at.isoformat() if result.completed_at else None,
        duration_seconds=result.duration_seconds,
    )


@router.post("/tasks/{task_id}/cancel", response_model=TaskStatusResponse)
async def cancel_task(task_id: str) -> TaskStatusResponse:
    """Cancel a running background task."""
    manager = get_task_manager()
    result = manager.get_status(task_id)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )

    if result.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task cannot be cancelled (status: {result.status.value})",
        )

    cancelled = manager.cancel(task_id)
    if not cancelled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Task could not be cancelled",
        )

    # Get updated status
    result = manager.get_status(task_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )

    return TaskStatusResponse(
        task_id=result.task_id,
        status=result.status.value,
        progress=result.progress,
        progress_message=result.progress_message,
        result=result.result if isinstance(result.result, dict) else None,
        error=result.error,
        created_at=result.created_at.isoformat(),
        started_at=result.started_at.isoformat() if result.started_at else None,
        completed_at=result.completed_at.isoformat() if result.completed_at else None,
        duration_seconds=result.duration_seconds,
    )


@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks(
    status_filter: TaskStatus | None = None,
    limit: int = 50,
) -> TaskListResponse:
    """List all background tasks."""
    manager = get_task_manager()
    tasks = manager.list_tasks(status=status_filter, limit=limit)

    return TaskListResponse(
        tasks=[
            TaskStatusResponse(
                task_id=t.task_id,
                status=t.status.value,
                progress=t.progress,
                progress_message=t.progress_message,
                result=t.result if isinstance(t.result, dict) else None,
                error=t.error,
                created_at=t.created_at.isoformat(),
                started_at=t.started_at.isoformat() if t.started_at else None,
                completed_at=t.completed_at.isoformat() if t.completed_at else None,
                duration_seconds=t.duration_seconds,
            )
            for t in tasks
        ],
        count=len(tasks),
    )
