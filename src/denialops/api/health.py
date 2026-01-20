"""Health check endpoints for DenialOps."""

import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from denialops.config import get_settings

router = APIRouter(tags=["health"])

# Track startup time
_startup_time = time.time()


class HealthStatus(BaseModel):
    """Basic health check response."""

    status: str = Field(..., description="Health status: 'healthy' or 'unhealthy'")
    timestamp: datetime = Field(..., description="Current server time")
    version: str = Field(..., description="Application version")


class ReadinessStatus(BaseModel):
    """Readiness check response with component status."""

    status: str = Field(..., description="Overall status: 'ready' or 'not_ready'")
    timestamp: datetime = Field(..., description="Current server time")
    checks: dict[str, Any] = Field(..., description="Individual component checks")


class LivenessStatus(BaseModel):
    """Liveness check response."""

    status: str = Field(..., description="Status: 'alive'")
    uptime_seconds: float = Field(..., description="Seconds since startup")


class DetailedHealthStatus(BaseModel):
    """Detailed health status with all system information."""

    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Current server time")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Environment (dev/prod)")
    uptime_seconds: float = Field(..., description="Seconds since startup")
    checks: dict[str, Any] = Field(..., description="Component health checks")
    config: dict[str, Any] = Field(..., description="Safe configuration values")


def _check_storage() -> dict[str, Any]:
    """Check storage availability."""
    settings = get_settings()
    artifacts_path = settings.artifacts_path

    try:
        # Check if path exists and is writable
        if not artifacts_path.exists():
            artifacts_path.mkdir(parents=True, exist_ok=True)

        # Test write
        test_file = artifacts_path / ".health_check"
        test_file.write_text("ok")
        test_file.unlink()

        return {
            "status": "healthy",
            "path": str(artifacts_path),
            "writable": True,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "path": str(artifacts_path),
            "error": str(e),
        }


def _check_llm() -> dict[str, Any]:
    """Check LLM configuration."""
    settings = get_settings()

    return {
        "status": "healthy" if settings.has_llm_key else "degraded",
        "provider": settings.llm_provider.value,
        "model": settings.llm_model,
        "api_key_configured": settings.has_llm_key,
        "note": "LLM unavailable, using heuristics" if not settings.has_llm_key else None,
    }


def _check_cache() -> dict[str, Any]:
    """Check cache configuration."""
    settings = get_settings()

    return {
        "status": "healthy",
        "backend": settings.cache_backend.value,
        "ttl_seconds": settings.cache_ttl,
        "redis_url": settings.redis_url if settings.cache_backend.value == "redis" else None,
    }


@router.get(
    "/health",
    response_model=HealthStatus,
    summary="Basic health check",
    description="Quick health check for load balancers",
)
async def health_check() -> HealthStatus:
    """Basic health check endpoint."""
    return HealthStatus(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version="0.1.0",
    )


@router.get(
    "/health/live",
    response_model=LivenessStatus,
    summary="Liveness probe",
    description="Kubernetes liveness probe - checks if process is running",
)
async def liveness_check() -> LivenessStatus:
    """Liveness probe for Kubernetes."""
    return LivenessStatus(
        status="alive",
        uptime_seconds=round(time.time() - _startup_time, 2),
    )


@router.get(
    "/health/ready",
    response_model=ReadinessStatus,
    summary="Readiness probe",
    description="Kubernetes readiness probe - checks if ready to serve traffic",
)
async def readiness_check() -> ReadinessStatus:
    """Readiness probe for Kubernetes."""
    storage_check = _check_storage()
    llm_check = _check_llm()

    # Determine overall status
    all_healthy = storage_check["status"] == "healthy"
    overall_status = "ready" if all_healthy else "not_ready"

    return ReadinessStatus(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        checks={
            "storage": storage_check,
            "llm": llm_check,
        },
    )


@router.get(
    "/health/detailed",
    response_model=DetailedHealthStatus,
    summary="Detailed health check",
    description="Comprehensive health status with all system information",
)
async def detailed_health_check() -> DetailedHealthStatus:
    """Detailed health check with all component status."""
    settings = get_settings()

    storage_check = _check_storage()
    llm_check = _check_llm()
    cache_check = _check_cache()

    # Determine overall status
    all_healthy = storage_check["status"] == "healthy"
    overall_status = "healthy" if all_healthy else "unhealthy"

    # Safe config values (no secrets)
    safe_config = {
        "api_host": settings.api_host,
        "api_port": settings.api_port,
        "artifacts_path": str(settings.artifacts_path),
        "max_upload_size": settings.max_upload_size,
        "artifact_retention_days": settings.artifact_retention_days,
        "log_level": settings.log_level,
    }

    return DetailedHealthStatus(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        version="0.1.0",
        environment=settings.environment,
        uptime_seconds=round(time.time() - _startup_time, 2),
        checks={
            "storage": storage_check,
            "llm": llm_check,
            "cache": cache_check,
        },
        config=safe_config,
    )


@router.get(
    "/",
    response_model=dict[str, str],
    summary="API root",
    description="Basic API information",
    status_code=status.HTTP_200_OK,
)
async def root() -> dict[str, str]:
    """API root endpoint."""
    return {
        "name": "DenialOps API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
