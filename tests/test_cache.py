"""Tests for caching functionality."""

import time

import pytest

from denialops.cache import CachedResponse, CacheKey, MemoryCache


def _make_response(
    content: str = "Test",
    prompt_tokens: int = 5,
    completion_tokens: int = 10,
    model: str = "gpt-4o",
    ttl: int = 3600,
) -> CachedResponse:
    """Helper to create CachedResponse with proper arguments."""
    return CachedResponse(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model,
        cached_at=time.time(),
        ttl=ttl,
    )


class TestCacheKey:
    """Tests for CacheKey."""

    def test_creates_key_from_request(self) -> None:
        """Test cache key creation from request."""
        key = CacheKey.from_request(
            provider="openai",
            model="gpt-4o",
            prompt="Test prompt",
            system="Test system",
        )

        assert key.provider == "openai"
        assert key.model == "gpt-4o"
        assert len(key.prompt_hash) == 16  # SHA256 truncated
        assert key.system_hash is not None
        assert len(key.system_hash) == 16

    def test_same_inputs_produce_same_key(self) -> None:
        """Test that identical inputs produce identical keys."""
        key1 = CacheKey.from_request(
            provider="openai",
            model="gpt-4o",
            prompt="Test prompt",
        )
        key2 = CacheKey.from_request(
            provider="openai",
            model="gpt-4o",
            prompt="Test prompt",
        )

        assert key1.to_string() == key2.to_string()

    def test_different_prompts_produce_different_keys(self) -> None:
        """Test that different prompts produce different keys."""
        key1 = CacheKey.from_request(
            provider="openai",
            model="gpt-4o",
            prompt="Test prompt 1",
        )
        key2 = CacheKey.from_request(
            provider="openai",
            model="gpt-4o",
            prompt="Test prompt 2",
        )

        assert key1.to_string() != key2.to_string()


class TestCachedResponse:
    """Tests for CachedResponse."""

    def test_creates_response(self) -> None:
        """Test cached response creation."""
        response = _make_response(content="Test content", prompt_tokens=10, completion_tokens=20)

        assert response.content == "Test content"
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 20

    def test_serialization(self) -> None:
        """Test JSON serialization."""
        response = _make_response(content="Test", prompt_tokens=5, completion_tokens=10)

        # Test to_dict and from_dict
        data = response.to_dict()
        restored = CachedResponse.from_dict(data)
        assert restored.content == response.content
        assert restored.prompt_tokens == response.prompt_tokens

        # Test to_json and from_json
        json_str = response.to_json()
        restored2 = CachedResponse.from_json(json_str)
        assert restored2.content == response.content


class TestMemoryCache:
    """Tests for MemoryCache."""

    def test_set_and_get(self) -> None:
        """Test basic set and get operations."""
        cache = MemoryCache(max_size=100, default_ttl=3600)
        key = CacheKey.from_request("openai", "gpt-4o", "Test")
        response = _make_response(content="Result")

        cache.set(key, response)
        retrieved = cache.get(key)

        assert retrieved is not None
        assert retrieved.content == "Result"

    def test_get_nonexistent(self) -> None:
        """Test getting a nonexistent key."""
        cache = MemoryCache()
        key = CacheKey.from_request("openai", "gpt-4o", "Nonexistent")

        result = cache.get(key)
        assert result is None

    def test_expired_entries_not_returned(self) -> None:
        """Test that expired entries are not returned."""
        cache = MemoryCache(default_ttl=1)
        key = CacheKey.from_request("openai", "gpt-4o", "Test")
        response = _make_response(content="Result", ttl=1)

        cache.set(key, response)
        assert cache.get(key) is not None

        time.sleep(1.1)
        assert cache.get(key) is None

    def test_lru_eviction(self) -> None:
        """Test LRU eviction when cache is full."""
        cache = MemoryCache(max_size=2)

        key1 = CacheKey.from_request("openai", "gpt-4o", "Prompt 1")
        key2 = CacheKey.from_request("openai", "gpt-4o", "Prompt 2")
        key3 = CacheKey.from_request("openai", "gpt-4o", "Prompt 3")

        cache.set(key1, _make_response(content="1"))
        cache.set(key2, _make_response(content="2"))

        # Access key1 to make it more recently used
        cache.get(key1)

        # Add key3, should evict key2 (least recently used)
        cache.set(key3, _make_response(content="3"))

        assert cache.get(key1) is not None
        assert cache.get(key2) is None  # Evicted
        assert cache.get(key3) is not None

    def test_delete(self) -> None:
        """Test deleting a cache entry."""
        cache = MemoryCache()
        key = CacheKey.from_request("openai", "gpt-4o", "Test")
        response = _make_response(content="Result")

        cache.set(key, response)
        assert cache.get(key) is not None

        cache.delete(key)
        assert cache.get(key) is None

    def test_clear(self) -> None:
        """Test clearing all cache entries."""
        cache = MemoryCache()

        for i in range(5):
            key = CacheKey.from_request("openai", "gpt-4o", f"Prompt {i}")
            cache.set(key, _make_response(content=str(i)))

        cache.clear()

        for i in range(5):
            key = CacheKey.from_request("openai", "gpt-4o", f"Prompt {i}")
            assert cache.get(key) is None

    def test_stats(self) -> None:
        """Test cache statistics."""
        cache = MemoryCache(max_size=10)

        # Add some entries
        for i in range(3):
            key = CacheKey.from_request("openai", "gpt-4o", f"Prompt {i}")
            cache.set(key, _make_response(content=str(i)))

        # Get some entries (hits and misses)
        cache.get(CacheKey.from_request("openai", "gpt-4o", "Prompt 0"))  # Hit
        cache.get(CacheKey.from_request("openai", "gpt-4o", "Prompt 1"))  # Hit
        cache.get(CacheKey.from_request("openai", "gpt-4o", "Nonexistent"))  # Miss

        stats = cache.stats()
        assert stats["size"] == 3
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3, rel=0.01)

    def test_cleanup_expired(self) -> None:
        """Test cleanup of expired entries."""
        cache = MemoryCache(default_ttl=1)

        # Add entries that will expire
        for i in range(3):
            key = CacheKey.from_request("openai", "gpt-4o", f"Prompt {i}")
            cache.set(key, _make_response(content=str(i), ttl=1))

        # Add an entry that won't expire
        key_long = CacheKey.from_request("openai", "gpt-4o", "Long TTL")
        cache.set(key_long, _make_response(content="long", ttl=3600))

        assert cache.stats()["size"] == 4

        time.sleep(1.1)
        removed = cache.cleanup_expired()

        assert removed == 3
        assert cache.stats()["size"] == 1
        assert cache.get(key_long) is not None
