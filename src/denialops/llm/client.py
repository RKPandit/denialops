"""LLM client for DenialOps - supports OpenAI and Anthropic."""

import json
from abc import ABC, abstractmethod
from typing import Any

import anthropic
import openai

from denialops.config import LLMProvider


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

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


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        """Initialize the OpenAI client."""
        self.api_key = api_key
        self.model = model
        self._client: openai.OpenAI | None = None

    @property
    def client(self) -> openai.OpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Get a completion from OpenAI."""
        messages: list[dict[str, str]] = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return ""


class AnthropicClient(BaseLLMClient):
    """Anthropic API client."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022") -> None:
        """Initialize the Anthropic client."""
        self.api_key = api_key
        self.model = model
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        """Get or create Anthropic client."""
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """Get a completion from Anthropic."""
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

        response = self.client.messages.create(**kwargs)

        if response.content and len(response.content) > 0:
            return response.content[0].text
        return ""


class LLMClient:
    """Unified LLM client that supports multiple providers."""

    def __init__(
        self,
        provider: LLMProvider,
        api_key: str,
        model: str | None = None,
    ) -> None:
        """
        Initialize the LLM client.

        Args:
            provider: The LLM provider to use
            api_key: API key for the provider
            model: Model name (uses provider default if not specified)
        """
        self.provider = provider

        if provider == LLMProvider.OPENAI:
            default_model = "gpt-4o"
            self._client: BaseLLMClient = OpenAIClient(
                api_key=api_key,
                model=model or default_model,
            )
        else:
            default_model = "claude-3-5-sonnet-20241022"
            self._client = AnthropicClient(
                api_key=api_key,
                model=model or default_model,
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

    def extract_structured(
        self,
        prompt: str,
        system: str | None = None,
    ) -> dict[str, Any]:
        """Get a structured JSON response from the LLM."""
        return self._client.extract_structured(prompt, system)


def create_llm_client(
    provider: LLMProvider | str,
    api_key: str,
    model: str | None = None,
) -> LLMClient:
    """
    Factory function to create an LLM client.

    Args:
        provider: "openai" or "anthropic" (or LLMProvider enum)
        api_key: API key for the provider
        model: Optional model override

    Returns:
        Configured LLMClient
    """
    if isinstance(provider, str):
        provider = LLMProvider(provider.lower())

    return LLMClient(provider=provider, api_key=api_key, model=model)
