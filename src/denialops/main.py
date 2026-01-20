"""FastAPI application entry point for DenialOps."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from denialops import __version__
from denialops.api.health import router as health_router
from denialops.api.routes import router
from denialops.api.streaming import router as streaming_router
from denialops.api.tasks import router as tasks_router
from denialops.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    settings = get_settings()

    # Startup: ensure artifacts directory exists
    settings.artifacts_path.mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown: cleanup if needed
    pass


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="DenialOps",
        description="AI-powered insurance claim denial understanding and action copilot",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS middleware (dev only)
    if settings.is_dev:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Include routes
    app.include_router(health_router)
    app.include_router(router, prefix="/api/v1", tags=["cases"])
    app.include_router(streaming_router, prefix="/api/v1", tags=["streaming"])
    app.include_router(tasks_router, prefix="/api/v1", tags=["tasks"])

    return app


# Create app instance for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "denialops.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_dev,
    )
