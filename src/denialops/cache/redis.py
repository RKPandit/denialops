"""Redis cache implementation."""

import logging
import time
from typing import Any

from denialops.cache.base import CacheBackend, CachedResponse, CacheKey

logger = logging.getLogger(__name__)


class RedisCache(CacheBackend):
    """
    Redis-backed cache for distributed deployments.

    Requires redis package to be installed.
    Falls back gracefully if Redis is unavailable.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "denialops:llm:",
        default_ttl: int = 3600,
    ):
        """
        Initialize Redis cache.

        Args:
            url: Redis connection URL
            prefix: Key prefix for namespacing
            default_ttl: Default TTL in seconds (1 hour)
        """
        self._url = url
        self._prefix = prefix
        self._default_ttl = default_ttl
        self._client = None
        self._connected = False
        self._hits = 0
        self._misses = 0
        self._errors = 0

        self._connect()

    def _connect(self) -> None:
        """Establish Redis connection."""
        try:
            import redis

            self._client = redis.from_url(self._url, decode_responses=True)
            # Test connection
            self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self._url}")
        except ImportError:
            logger.warning("redis package not installed, cache disabled")
            self._connected = False
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}, cache disabled")
            self._connected = False

    def _full_key(self, key: CacheKey) -> str:
        """Get full Redis key with prefix."""
        return f"{self._prefix}{key.to_string()}"

    def get(self, key: CacheKey) -> CachedResponse | None:
        """Get cached response from Redis."""
        if not self._connected:
            self._misses += 1
            return None

        try:
            full_key = self._full_key(key)
            data = self._client.get(full_key)

            if data is None:
                self._misses += 1
                return None

            self._hits += 1
            return CachedResponse.from_json(data)

        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            self._errors += 1
            self._misses += 1
            return None

    def set(self, key: CacheKey, response: CachedResponse) -> None:
        """Store response in Redis with TTL."""
        if not self._connected:
            return

        try:
            # Set cached_at if not set
            if response.cached_at == 0:
                response.cached_at = time.time()

            # Use default TTL if not set
            ttl = response.ttl if response.ttl > 0 else self._default_ttl
            response.ttl = ttl

            full_key = self._full_key(key)
            self._client.setex(full_key, ttl, response.to_json())

        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            self._errors += 1

    def delete(self, key: CacheKey) -> bool:
        """Delete cached response from Redis."""
        if not self._connected:
            return False

        try:
            full_key = self._full_key(key)
            return self._client.delete(full_key) > 0

        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            self._errors += 1
            return False

    def clear(self) -> int:
        """Clear all cached responses with our prefix."""
        if not self._connected:
            return 0

        try:
            # Find all keys with our prefix
            pattern = f"{self._prefix}*"
            keys = list(self._client.scan_iter(pattern))

            if keys:
                return self._client.delete(*keys)
            return 0

        except Exception as e:
            logger.warning(f"Redis clear error: {e}")
            self._errors += 1
            return 0

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        stats = {
            "backend": "redis",
            "connected": self._connected,
            "url": self._url,
            "prefix": self._prefix,
            "hits": self._hits,
            "misses": self._misses,
            "errors": self._errors,
            "hit_rate": round(hit_rate, 4),
            "default_ttl": self._default_ttl,
        }

        # Add Redis info if connected
        if self._connected:
            try:
                info = self._client.info("memory")
                stats["redis_memory_used"] = info.get("used_memory_human", "unknown")

                # Count our keys
                pattern = f"{self._prefix}*"
                stats["size"] = sum(1 for _ in self._client.scan_iter(pattern))
            except Exception:
                pass

        return stats

    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._connected

    def reconnect(self) -> bool:
        """Attempt to reconnect to Redis."""
        self._connect()
        return self._connected
