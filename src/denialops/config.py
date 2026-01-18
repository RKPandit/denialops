"""Configuration management for DenialOps."""

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Environment
    environment: str = "dev"
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Storage
    artifacts_path: Path = Path("./artifacts")

    # LLM
    llm_provider: LLMProvider = LLMProvider.OPENAI
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_model: str = "gpt-4o"

    # Limits
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    artifact_retention_days: int = 7

    @property
    def is_dev(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "dev"

    @property
    def llm_api_key(self) -> str:
        """Get the API key for the configured LLM provider."""
        if self.llm_provider == LLMProvider.OPENAI:
            return self.openai_api_key
        return self.anthropic_api_key

    @property
    def has_llm_key(self) -> bool:
        """Check if an LLM API key is configured."""
        return bool(self.llm_api_key)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
