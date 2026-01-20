"""Base cache interface for DenialOps."""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheKey:
    """Structured cache key for LLM responses."""

    provider: str
    model: str
    prompt_hash: str
    system_hash: str | None = None

    @classmethod
    def from_request(
        cls,
        provider: str,
        model: str,
        prompt: str,
        system: str | None = None,
    ) -> "CacheKey":
        """Create cache key from LLM request parameters."""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        system_hash = (
            hashlib.sha256(system.encode()).hexdigest()[:16] if system else None
        )
        return cls(
            provider=provider,
            model=model,
            prompt_hash=prompt_hash,
            system_hash=system_hash,
        )

    def to_string(self) -> str:
        """Convert to string key for storage."""
        parts = [self.provider, self.model, self.prompt_hash]
        if self.system_hash:
            parts.append(self.system_hash)
        return ":".join(parts)

    @classmethod
    def from_string(cls, key_str: str) -> "CacheKey":
        """Parse string key back to CacheKey."""
        parts = key_str.split(":")
        if len(parts) == 3:
            return cls(provider=parts[0], model=parts[1], prompt_hash=parts[2])
        elif len(parts) == 4:
            return cls(
                provider=parts[0],
                model=parts[1],
                prompt_hash=parts[2],
                system_hash=parts[3],
            )
        else:
            raise ValueError(f"Invalid cache key format: {key_str}")


@dataclass
class CachedResponse:
    """Cached LLM response with metadata."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    cached_at: float  # Unix timestamp
    ttl: int  # Seconds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "content": self.content,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model": self.model,
            "cached_at": self.cached_at,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CachedResponse":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            prompt_tokens=data["prompt_tokens"],
            completion_tokens=data["completion_tokens"],
            model=data["model"],
            cached_at=data["cached_at"],
            ttl=data["ttl"],
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "CachedResponse":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: CacheKey) -> CachedResponse | None:
        """
        Get cached response by key.

        Returns None if key not found or expired.
        """
        pass

    @abstractmethod
    def set(self, key: CacheKey, response: CachedResponse) -> None:
        """
        Store response in cache.

        TTL should be respected by the implementation.
        """
        pass

    @abstractmethod
    def delete(self, key: CacheKey) -> bool:
        """
        Delete cached response.

        Returns True if key existed and was deleted.
        """
        pass

    @abstractmethod
    def clear(self) -> int:
        """
        Clear all cached responses.

        Returns number of entries cleared.
        """
        pass

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns dict with hits, misses, size, etc.
        """
        pass
