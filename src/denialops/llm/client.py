"""LLM client for DenialOps - supports OpenAI and Anthropic with production features."""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import anthropic
import openai

from denialops.config import LLMProvider

logger = logging.getLogger(__name__)


# =============================================================================
# Token Usage Tracking
# =============================================================================


@dataclass
class TokenUsage:
    """Track token usage for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0

    @property
    def estimated_cost(self) -> float:
        """Estimate cost based on model pricing (approximate)."""
        # Pricing per 1M tokens (as of 2024)
        pricing = {
            # OpenAI
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            # Anthropic
            "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
            "claude-3-opus": {"input": 15.0, "output": 75.0},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
        }

        # Find matching model (partial match)
        model_pricing = {"input": 5.0, "output": 15.0}  # Default
        for model_key, prices in pricing.items():
            if model_key in self.model.lower():
                model_pricing = prices
                break

        input_cost = (self.prompt_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (self.completion_tokens / 1_000_000) * model_pricing["output"]

        return input_cost + output_cost


@dataclass
class UsageTracker:
    """Aggregate token usage across multiple calls."""

    calls: list[TokenUsage] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(u.total_tokens for u in self.calls)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(u.prompt_tokens for u in self.calls)

    @property
    def total_completion_tokens(self) -> int:
        return sum(u.completion_tokens for u in self.calls)

    @property
    def total_cost(self) -> float:
        return sum(u.estimated_cost for u in self.calls)

    @property
    def total_latency_ms(self) -> float:
        return sum(u.latency_ms for u in self.calls)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def add(self, usage: TokenUsage) -> None:
        self.calls.append(usage)

    def summary(self) -> dict[str, Any]:
        """Return a summary of usage statistics."""
        return {
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "estimated_cost_usd": round(self.total_cost, 4),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "avg_latency_ms": (
                round(self.total_latency_ms / self.call_count, 2) if self.call_count > 0 else 0
            ),
        }


# =============================================================================
# Retry Logic
# =============================================================================


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        retryable_errors: tuple[type[Exception], ...] | None = None,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retryable_errors = retryable_errors or (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
        )

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt using exponential backoff."""
        delay = self.initial_delay * (self.exponential_base**attempt)
        return min(delay, self.max_delay)


def retry_with_backoff(config: RetryConfig):
    """Decorator for retry with exponential backoff."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_errors as e:
                    last_exception = e
                    if attempt < config.max_retries:
                        delay = config.get_delay(attempt)
                        logger.warning(
                            f"LLM call failed (attempt {attempt + 1}/{config.max_retries + 1}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"LLM call failed after {config.max_retries + 1} attempts: {e}"
                        )

            raise last_exception  # type: ignore

        return wrapper

    return decorator


# =============================================================================
# LLM Response
# =============================================================================


@dataclass
class LLMResponse:
    """Response from an LLM call with metadata."""

    content: str
    usage: TokenUsage
    model: str
    provider: str

    def __str__(self) -> str:
        return self.content


# =============================================================================
# Base Client
# =============================================================================


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, retry_config: RetryConfig | None = None):
        self.retry_config = retry_config or RetryConfig()
        self.usage_tracker = UsageTracker()

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Get a completion from the LLM."""
        pass

    @abstractmethod
    def complete_with_usage(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Get a completion with token usage tracking."""
        pass

    def extract_structured(
        self,
        prompt: str,
        system: str | None = None,
    ) -> dict[str, Any]:
        """Get a structured JSON response from the LLM."""
        response = self.complete(prompt, system)

        # Try to extract JSON from response
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            json_str = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            json_str = response[start:end].strip()
        else:
            json_str = response.strip()

        return json.loads(json_str)

    def get_usage_summary(self) -> dict[str, Any]:
        """Get summary of all token usage."""
        return self.usage_tracker.summary()


# =============================================================================
# OpenAI Client
# =============================================================================


class OpenAIClient(BaseLLMClient):
    """OpenAI API client with retry and usage tracking."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize the OpenAI client."""
        super().__init__(retry_config)
        self.api_key = api_key
        self.model = model
        self._client: openai.OpenAI | None = None

    @property
    def client(self) -> openai.OpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def _make_request(
        self,
        prompt: str,
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, TokenUsage]:
        """Make the actual API request."""
        messages: list[dict[str, str]] = []

        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            max_tokens=max_tokens,
            temperature=temperature,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract usage
        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
            model=self.model,
            latency_ms=latency_ms,
        )

        content = ""
        if response.choices and response.choices[0].message.content:
            content = response.choices[0].message.content

        return content, usage

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Get a completion from OpenAI with retry."""

        @retry_with_backoff(self.retry_config)
        def _call():
            content, usage = self._make_request(prompt, system, max_tokens, temperature)
            self.usage_tracker.add(usage)
            logger.debug(
                f"OpenAI call completed: {usage.total_tokens} tokens, "
                f"{usage.latency_ms:.0f}ms, ${usage.estimated_cost:.4f}"
            )
            return content

        return _call()

    def complete_with_usage(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Get a completion with full response metadata."""

        @retry_with_backoff(self.retry_config)
        def _call():
            content, usage = self._make_request(prompt, system, max_tokens, temperature)
            self.usage_tracker.add(usage)
            return LLMResponse(
                content=content,
                usage=usage,
                model=self.model,
                provider="openai",
            )

        return _call()


# =============================================================================
# Anthropic Client
# =============================================================================


class AnthropicClient(BaseLLMClient):
    """Anthropic API client with retry and usage tracking."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize the Anthropic client."""
        super().__init__(retry_config)
        self.api_key = api_key
        self.model = model
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        """Get or create Anthropic client."""
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _make_request(
        self,
        prompt: str,
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, TokenUsage]:
        """Make the actual API request."""
        messages = [{"role": "user", "content": prompt}]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if system:
            kwargs["system"] = system

        if temperature > 0:
            kwargs["temperature"] = temperature

        start_time = time.time()
        response = self.client.messages.create(**kwargs)
        latency_ms = (time.time() - start_time) * 1000

        # Extract usage
        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens if response.usage else 0,
            completion_tokens=response.usage.output_tokens if response.usage else 0,
            total_tokens=(
                (response.usage.input_tokens + response.usage.output_tokens)
                if response.usage
                else 0
            ),
            model=self.model,
            latency_ms=latency_ms,
        )

        content = ""
        if response.content and len(response.content) > 0:
            content = response.content[0].text

        return content, usage

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Get a completion from Anthropic with retry."""

        @retry_with_backoff(self.retry_config)
        def _call():
            content, usage = self._make_request(prompt, system, max_tokens, temperature)
            self.usage_tracker.add(usage)
            logger.debug(
                f"Anthropic call completed: {usage.total_tokens} tokens, "
                f"{usage.latency_ms:.0f}ms, ${usage.estimated_cost:.4f}"
            )
            return content

        return _call()

    def complete_with_usage(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Get a completion with full response metadata."""

        @retry_with_backoff(self.retry_config)
        def _call():
            content, usage = self._make_request(prompt, system, max_tokens, temperature)
            self.usage_tracker.add(usage)
            return LLMResponse(
                content=content,
                usage=usage,
                model=self.model,
                provider="anthropic",
            )

        return _call()


# =============================================================================
# Unified Client
# =============================================================================


class LLMClient:
    """Unified LLM client that supports multiple providers."""

    def __init__(
        self,
        provider: LLMProvider,
        api_key: str,
        model: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """
        Initialize the LLM client.

        Args:
            provider: The LLM provider to use
            api_key: API key for the provider
            model: Model name (uses provider default if not specified)
            retry_config: Configuration for retry behavior
        """
        self.provider = provider
        self.retry_config = retry_config or RetryConfig()

        if provider == LLMProvider.OPENAI:
            default_model = "gpt-4o"
            self._client: BaseLLMClient = OpenAIClient(
                api_key=api_key,
                model=model or default_model,
                retry_config=self.retry_config,
            )
        else:
            default_model = "claude-3-5-sonnet-20241022"
            self._client = AnthropicClient(
                api_key=api_key,
                model=model or default_model,
                retry_config=self.retry_config,
            )

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Get a completion from the configured LLM."""
        return self._client.complete(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def complete_with_usage(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Get a completion with token usage tracking."""
        return self._client.complete_with_usage(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def extract_structured(
        self,
        prompt: str,
        system: str | None = None,
    ) -> dict[str, Any]:
        """Get a structured JSON response from the LLM."""
        return self._client.extract_structured(prompt, system)

    def get_usage_summary(self) -> dict[str, Any]:
        """Get summary of all token usage for this client."""
        return self._client.get_usage_summary()


# =============================================================================
# Factory Function
# =============================================================================


def create_llm_client(
    provider: LLMProvider | str,
    api_key: str,
    model: str | None = None,
    max_retries: int = 3,
) -> LLMClient:
    """
    Factory function to create an LLM client.

    Args:
        provider: "openai" or "anthropic" (or LLMProvider enum)
        api_key: API key for the provider
        model: Optional model override
        max_retries: Maximum number of retries for failed calls

    Returns:
        Configured LLMClient
    """
    if isinstance(provider, str):
        provider = LLMProvider(provider.lower())

    retry_config = RetryConfig(max_retries=max_retries)

    return LLMClient(
        provider=provider,
        api_key=api_key,
        model=model,
        retry_config=retry_config,
    )
