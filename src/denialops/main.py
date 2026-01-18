"""FastAPI application entry point for DenialOps."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from denialops import __version__
from denialops.api.routes import router
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
    app.include_router(router, tags=["cases"])

    # Health check
    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "version": __version__}

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
