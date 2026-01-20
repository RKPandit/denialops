"""In-memory cache implementation."""

import time
from threading import Lock
from typing import Any

from denialops.cache.base import CacheBackend, CachedResponse, CacheKey


class MemoryCache(CacheBackend):
    """
    Thread-safe in-memory cache with TTL support.

    Suitable for development and single-process deployments.
    For multi-process or distributed deployments, use RedisCache.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of entries before LRU eviction
            default_ttl: Default TTL in seconds (1 hour)
        """
        self._cache: dict[str, CachedResponse] = {}
        self._access_order: list[str] = []  # For LRU tracking
        self._lock = Lock()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def get(self, key: CacheKey) -> CachedResponse | None:
        """Get cached response, respecting TTL."""
        key_str = key.to_string()

        with self._lock:
            if key_str not in self._cache:
                self._misses += 1
                return None

            response = self._cache[key_str]

            # Check if expired
            if time.time() > response.cached_at + response.ttl:
                del self._cache[key_str]
                self._access_order.remove(key_str)
                self._misses += 1
                return None

            # Update access order for LRU
            self._access_order.remove(key_str)
            self._access_order.append(key_str)

            self._hits += 1
            return response

    def set(self, key: CacheKey, response: CachedResponse) -> None:
        """Store response with TTL."""
        key_str = key.to_string()

        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self._max_size:
                self._evict_lru()

            # Update cached_at if not set
            if response.cached_at == 0:
                response.cached_at = time.time()

            # Use default TTL if not set
            if response.ttl == 0:
                response.ttl = self._default_ttl

            # Remove from access order if updating existing key
            if key_str in self._access_order:
                self._access_order.remove(key_str)

            self._cache[key_str] = response
            self._access_order.append(key_str)

    def delete(self, key: CacheKey) -> bool:
        """Delete cached response."""
        key_str = key.to_string()

        with self._lock:
            if key_str in self._cache:
                del self._cache[key_str]
                self._access_order.remove(key_str)
                return True
            return False

    def clear(self) -> int:
        """Clear all cached responses."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_order.clear()
            return count

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "backend": "memory",
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
                "default_ttl": self._default_ttl,
            }

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._access_order:
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns number of entries removed.
        Useful for periodic cleanup in long-running processes.
        """
        now = time.time()
        expired_keys: list[str] = []

        with self._lock:
            for key_str, response in self._cache.items():
                if now > response.cached_at + response.ttl:
                    expired_keys.append(key_str)

            for key_str in expired_keys:
                del self._cache[key_str]
                self._access_order.remove(key_str)

            return len(expired_keys)
