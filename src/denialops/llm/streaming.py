"""Streaming LLM client for real-time response delivery."""

import logging
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

import anthropic
import openai

from denialops.config import LLMProvider
from denialops.llm.client import RetryConfig, TokenUsage

logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """A single chunk from a streaming response."""

    content: str
    is_final: bool = False
    usage: TokenUsage | None = None


@dataclass
class StreamingResponse:
    """Container for streaming response with metadata."""

    chunks: Iterator[StreamChunk]
    model: str
    provider: str


@dataclass
class AsyncStreamingResponse:
    """Container for async streaming response with metadata."""

    chunks: AsyncIterator[StreamChunk]
    model: str
    provider: str


class OpenAIStreamingClient:
    """OpenAI streaming client."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        retry_config: RetryConfig | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.retry_config = retry_config or RetryConfig()
        self._client: openai.OpenAI | None = None
        self._async_client: openai.AsyncOpenAI | None = None

    @property
    def client(self) -> openai.OpenAI:
        """Get or create sync OpenAI client."""
        if self._client is None:
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    @property
    def async_client(self) -> openai.AsyncOpenAI:
        """Get or create async OpenAI client."""
        if self._async_client is None:
            self._async_client = openai.AsyncOpenAI(api_key=self.api_key)
        return self._async_client

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> StreamingResponse:
        """Stream a completion from OpenAI."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        def generate_chunks() -> Iterator[StreamChunk]:
            start_time = time.time()
            total_tokens = 0

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamChunk(
                        content=chunk.choices[0].delta.content,
                        is_final=False,
                    )

                # Check for usage in final chunk
                if chunk.usage:
                    latency_ms = (time.time() - start_time) * 1000
                    total_tokens = chunk.usage.total_tokens
                    yield StreamChunk(
                        content="",
                        is_final=True,
                        usage=TokenUsage(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                            total_tokens=total_tokens,
                            model=self.model,
                            latency_ms=latency_ms,
                        ),
                    )

        return StreamingResponse(
            chunks=generate_chunks(),
            model=self.model,
            provider="openai",
        )

    async def stream_async(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AsyncStreamingResponse:
        """Stream a completion from OpenAI asynchronously."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async def generate_chunks() -> AsyncIterator[StreamChunk]:
            start_time = time.time()

            stream = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamChunk(
                        content=chunk.choices[0].delta.content,
                        is_final=False,
                    )

                if chunk.usage:
                    latency_ms = (time.time() - start_time) * 1000
                    yield StreamChunk(
                        content="",
                        is_final=True,
                        usage=TokenUsage(
                            prompt_tokens=chunk.usage.prompt_tokens,
                            completion_tokens=chunk.usage.completion_tokens,
                            total_tokens=chunk.usage.total_tokens,
                            model=self.model,
                            latency_ms=latency_ms,
                        ),
                    )

        return AsyncStreamingResponse(
            chunks=generate_chunks(),
            model=self.model,
            provider="openai",
        )


class AnthropicStreamingClient:
    """Anthropic streaming client."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        retry_config: RetryConfig | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.retry_config = retry_config or RetryConfig()
        self._client: anthropic.Anthropic | None = None
        self._async_client: anthropic.AsyncAnthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        """Get or create sync Anthropic client."""
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    @property
    def async_client(self) -> anthropic.AsyncAnthropic:
        """Get or create async Anthropic client."""
        if self._async_client is None:
            self._async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._async_client

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> StreamingResponse:
        """Stream a completion from Anthropic."""
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

        def generate_chunks() -> Iterator[StreamChunk]:
            start_time = time.time()
            input_tokens = 0
            output_tokens = 0

            with self.client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield StreamChunk(content=text, is_final=False)

                # Get final message for usage
                message = stream.get_final_message()
                if message and message.usage:
                    latency_ms = (time.time() - start_time) * 1000
                    input_tokens = message.usage.input_tokens
                    output_tokens = message.usage.output_tokens

                    yield StreamChunk(
                        content="",
                        is_final=True,
                        usage=TokenUsage(
                            prompt_tokens=input_tokens,
                            completion_tokens=output_tokens,
                            total_tokens=input_tokens + output_tokens,
                            model=self.model,
                            latency_ms=latency_ms,
                        ),
                    )

        return StreamingResponse(
            chunks=generate_chunks(),
            model=self.model,
            provider="anthropic",
        )

    async def stream_async(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AsyncStreamingResponse:
        """Stream a completion from Anthropic asynchronously."""
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

        async def generate_chunks() -> AsyncIterator[StreamChunk]:
            start_time = time.time()

            async with self.async_client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield StreamChunk(content=text, is_final=False)

                message = await stream.get_final_message()
                if message and message.usage:
                    latency_ms = (time.time() - start_time) * 1000
                    yield StreamChunk(
                        content="",
                        is_final=True,
                        usage=TokenUsage(
                            prompt_tokens=message.usage.input_tokens,
                            completion_tokens=message.usage.output_tokens,
                            total_tokens=message.usage.input_tokens + message.usage.output_tokens,
                            model=self.model,
                            latency_ms=latency_ms,
                        ),
                    )

        return AsyncStreamingResponse(
            chunks=generate_chunks(),
            model=self.model,
            provider="anthropic",
        )


class StreamingLLMClient:
    """Unified streaming LLM client supporting multiple providers."""

    def __init__(
        self,
        provider: LLMProvider,
        api_key: str,
        model: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self.provider = provider
        self.retry_config = retry_config or RetryConfig()

        if provider == LLMProvider.OPENAI:
            default_model = "gpt-4o"
            self._client: OpenAIStreamingClient | AnthropicStreamingClient = OpenAIStreamingClient(
                api_key=api_key,
                model=model or default_model,
                retry_config=self.retry_config,
            )
        else:
            default_model = "claude-3-5-sonnet-20241022"
            self._client = AnthropicStreamingClient(
                api_key=api_key,
                model=model or default_model,
                retry_config=self.retry_config,
            )

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> StreamingResponse:
        """Stream a completion from the configured LLM."""
        return self._client.stream(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def stream_async(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AsyncStreamingResponse:
        """Stream a completion asynchronously."""
        return await self._client.stream_async(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )


def create_streaming_client(
    provider: LLMProvider | str,
    api_key: str,
    model: str | None = None,
    max_retries: int = 3,
) -> StreamingLLMClient:
    """
    Factory function to create a streaming LLM client.

    Args:
        provider: "openai" or "anthropic" (or LLMProvider enum)
        api_key: API key for the provider
        model: Optional model override
        max_retries: Maximum number of retries for failed calls

    Returns:
        Configured StreamingLLMClient
    """
    if isinstance(provider, str):
        provider = LLMProvider(provider.lower())

    retry_config = RetryConfig(max_retries=max_retries)

    return StreamingLLMClient(
        provider=provider,
        api_key=api_key,
        model=model,
        retry_config=retry_config,
    )
