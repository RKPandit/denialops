"""Cached LLM client wrapper."""

import logging
import time
from typing import Any

from denialops.cache.base import CacheBackend, CachedResponse, CacheKey
from denialops.llm.client import BaseLLMClient, LLMResponse, TokenUsage, UsageTracker

logger = logging.getLogger(__name__)


class CachedLLMClient(BaseLLMClient):
    """
    LLM client wrapper that adds caching support.

    Wraps any BaseLLMClient and caches responses based on prompt content.
    """

    def __init__(
        self,
        client: BaseLLMClient,
        cache: CacheBackend,
        provider: str,
        model: str,
        cache_ttl: int = 3600,
        cache_temperature_threshold: float = 0.1,
    ):
        """
        Initialize cached client.

        Args:
            client: The underlying LLM client to wrap
            cache: Cache backend to use
            provider: Provider name for cache key
            model: Model name for cache key
            cache_ttl: Default TTL for cached responses
            cache_temperature_threshold: Only cache if temperature <= this value
        """
        super().__init__(client.retry_config)
        self._client = client
        self._cache = cache
        self._provider = provider
        self._model = model
        self._cache_ttl = cache_ttl
        self._cache_temperature_threshold = cache_temperature_threshold
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def usage_tracker(self) -> UsageTracker:
        """Share usage tracker with wrapped client."""
        return self._client.usage_tracker

    def _should_cache(self, temperature: float) -> bool:
        """Determine if response should be cached based on temperature."""
        # Only cache deterministic responses (low temperature)
        return temperature <= self._cache_temperature_threshold

    def _get_cache_key(self, prompt: str, system: str | None) -> CacheKey:
        """Generate cache key for request."""
        return CacheKey.from_request(
            provider=self._provider,
            model=self._model,
            prompt=prompt,
            system=system,
        )

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Get a completion, using cache if available."""
        response = self.complete_with_usage(prompt, system, max_tokens, temperature)
        return response.content

    def complete_with_usage(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Get a completion with usage tracking, using cache if available."""
        # Check if we should use cache
        if not self._should_cache(temperature):
            logger.debug("Skipping cache due to high temperature")
            return self._client.complete_with_usage(prompt, system, max_tokens, temperature)

        # Generate cache key
        cache_key = self._get_cache_key(prompt, system)

        # Try to get from cache
        cached = self._cache.get(cache_key)
        if cached:
            self._cache_hits += 1
            logger.info(
                f"Cache hit for {self._provider}/{self._model} "
                f"(saved ~{cached.prompt_tokens + cached.completion_tokens} tokens)"
            )

            # Create response from cache
            usage = TokenUsage(
                prompt_tokens=cached.prompt_tokens,
                completion_tokens=cached.completion_tokens,
                total_tokens=cached.prompt_tokens + cached.completion_tokens,
                model=cached.model,
                latency_ms=0.0,  # Cached, no latency
            )

            return LLMResponse(
                content=cached.content,
                usage=usage,
                model=cached.model,
                provider=self._provider,
            )

        # Cache miss - make actual request
        self._cache_misses += 1
        logger.debug(f"Cache miss for {self._provider}/{self._model}")

        response = self._client.complete_with_usage(
            prompt, system, max_tokens, temperature
        )

        # Store in cache
        cached_response = CachedResponse(
            content=response.content,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            model=response.model,
            cached_at=time.time(),
            ttl=self._cache_ttl,
        )
        self._cache.set(cache_key, cached_response)

        logger.debug(
            f"Cached response for {self._provider}/{self._model} "
            f"(TTL: {self._cache_ttl}s)"
        )

        return response

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "provider": self._provider,
            "model": self._model,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": round(hit_rate, 4),
            "cache_ttl": self._cache_ttl,
            "cache_backend_stats": self._cache.stats(),
        }

    def clear_cache(self) -> int:
        """Clear the cache. Returns number of entries cleared."""
        return self._cache.clear()


def create_cached_client(
    client: BaseLLMClient,
    cache: CacheBackend,
    provider: str,
    model: str,
    cache_ttl: int = 3600,
) -> CachedLLMClient:
    """
    Create a cached wrapper for an LLM client.

    Args:
        client: The LLM client to wrap
        cache: Cache backend (MemoryCache or RedisCache)
        provider: Provider name ('openai' or 'anthropic')
        model: Model name
        cache_ttl: TTL for cached responses in seconds

    Returns:
        CachedLLMClient wrapping the original client
    """
    return CachedLLMClient(
        client=client,
        cache=cache,
        provider=provider,
        model=model,
        cache_ttl=cache_ttl,
    )
