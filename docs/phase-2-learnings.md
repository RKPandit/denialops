# Phase 2 Learnings: Production LLM Patterns

This document explains what we built in Phase 2 and the AI/ML engineering concepts behind each feature.

---

## Table of Contents

1. [Overview](#overview)
2. [Retry Logic with Exponential Backoff](#1-retry-logic-with-exponential-backoff)
3. [Token Usage Tracking](#2-token-usage-tracking)
4. [Prompt Template Organization](#3-prompt-template-organization)
5. [Code Walkthrough](#4-code-walkthrough)
6. [Key Takeaways](#5-key-takeaways)

---

## Overview

Phase 2 focused on making the LLM client **production-ready**. In Phase 1, we had a basic client that could call OpenAI/Anthropic. In Phase 2, we added:

| Feature | Why It Matters |
|---------|----------------|
| Retry with backoff | Handles rate limits and transient failures |
| Token tracking | Monitor costs and optimize prompts |
| Cost estimation | Budget management and alerting |
| Latency tracking | Performance monitoring |
| Organized prompts | Maintainable, testable prompt engineering |

**Files Changed:**
- `src/denialops/llm/client.py` (214 → 599 lines)
- `src/denialops/llm/prompts.py` (95 → 440 lines)

---

## 1. Retry Logic with Exponential Backoff

### The Problem

LLM APIs fail for various reasons:
- **Rate limits**: Too many requests per minute
- **Timeouts**: Network issues or slow responses
- **Server errors**: Temporary API outages

Without retry logic, a single failure crashes your application.

### The Solution: Exponential Backoff

```
Request fails → Wait 1 second → Retry
Fails again  → Wait 2 seconds → Retry
Fails again  → Wait 4 seconds → Retry
Fails again  → Wait 8 seconds → Retry (or give up)
```

The wait time doubles each attempt. This is called **exponential backoff**.

### Why Exponential?

- **Linear backoff** (1s, 2s, 3s, 4s): Too aggressive, still hammers the API
- **Exponential backoff** (1s, 2s, 4s, 8s): Backs off quickly, gives API time to recover
- **With jitter** (random variation): Prevents "thundering herd" when many clients retry simultaneously

### The Code

```python
# src/denialops/llm/client.py

class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,           # How many times to retry
        initial_delay: float = 1.0,      # First retry waits 1 second
        max_delay: float = 60.0,         # Never wait more than 60 seconds
        exponential_base: float = 2.0,   # Double the delay each time
        retryable_errors: tuple = None,  # Which errors to retry
    ):
        self.retryable_errors = retryable_errors or (
            openai.RateLimitError,       # 429 Too Many Requests
            openai.APITimeoutError,      # Request timed out
            openai.APIConnectionError,   # Network error
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
        )

    def get_delay(self, attempt: int) -> float:
        """Calculate delay: 1s, 2s, 4s, 8s... up to max_delay."""
        delay = self.initial_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)
```

### The Decorator Pattern

We use a **decorator** to wrap any function with retry logic:

```python
def retry_with_backoff(config: RetryConfig):
    """Decorator for retry with exponential backoff."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)  # Try the function
                except config.retryable_errors as e:
                    last_exception = e
                    if attempt < config.max_retries:
                        delay = config.get_delay(attempt)
                        logger.warning(f"Retry {attempt + 1}: {e}. Waiting {delay}s...")
                        time.sleep(delay)

            raise last_exception  # All retries failed

        return wrapper
    return decorator

# Usage:
@retry_with_backoff(RetryConfig(max_retries=3))
def make_api_call():
    return client.chat.completions.create(...)
```

### Key Learning

> **Production systems must handle failures gracefully.** Never assume an API call will succeed. Always have a retry strategy.

---

## 2. Token Usage Tracking

### The Problem

LLM APIs charge per token:
- GPT-4o: $5/1M input tokens, $15/1M output tokens
- Claude 3.5 Sonnet: $3/1M input, $15/1M output

Without tracking, you can't:
- Monitor costs
- Optimize expensive prompts
- Set up budget alerts
- Compare model efficiency

### The Solution: Track Every Call

```python
@dataclass
class TokenUsage:
    """Track token usage for a single LLM call."""

    prompt_tokens: int = 0       # Input tokens (your prompt)
    completion_tokens: int = 0   # Output tokens (model response)
    total_tokens: int = 0        # Sum of above
    model: str = ""              # Which model was used
    latency_ms: float = 0.0      # How long the call took
```

### Cost Estimation

```python
@property
def estimated_cost(self) -> float:
    """Estimate cost based on model pricing."""

    # Pricing per 1M tokens
    pricing = {
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
    }

    # Calculate cost
    input_cost = (self.prompt_tokens / 1_000_000) * pricing[model]["input"]
    output_cost = (self.completion_tokens / 1_000_000) * pricing[model]["output"]

    return input_cost + output_cost
```

### Aggregate Tracking

Track usage across multiple calls:

```python
@dataclass
class UsageTracker:
    """Aggregate token usage across multiple calls."""

    calls: list[TokenUsage] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(u.total_tokens for u in self.calls)

    @property
    def total_cost(self) -> float:
        return sum(u.estimated_cost for u in self.calls)

    def summary(self) -> dict:
        return {
            "call_count": len(self.calls),
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.total_cost, 4),
            "avg_latency_ms": self.total_latency_ms / len(self.calls),
        }
```

### Usage Example

```python
# Create client
client = create_llm_client("openai", api_key)

# Make calls (usage is tracked automatically)
response1 = client.complete("Summarize this document...")
response2 = client.complete("Extract the key points...")

# Get usage summary
summary = client.get_usage_summary()
print(summary)
# {
#   "call_count": 2,
#   "total_tokens": 1500,
#   "estimated_cost_usd": 0.0125,
#   "avg_latency_ms": 850.5
# }
```

### Key Learning

> **What you can't measure, you can't optimize.** Always track token usage, latency, and cost for LLM calls.

---

## 3. Prompt Template Organization

### The Problem

Prompts scattered across code are:
- Hard to find and modify
- Difficult to test
- Easy to introduce inconsistencies
- Not reusable

### The Solution: Centralized Prompt Library

All prompts live in `src/denialops/llm/prompts.py`:

```python
# =============================================================================
# Extraction Prompts
# =============================================================================

EXTRACT_FACTS_SYSTEM = """You are an expert at analyzing health insurance
denial letters. Extract structured information from the denial letter text
provided. Always respond with valid JSON."""

EXTRACT_FACTS_USER = """Analyze this insurance denial letter:

<denial_letter>
{text}
</denial_letter>

Extract and return a JSON object with these fields:
{{
  "denial_reason": "...",
  "denial_codes": [...],
  ...
}}

Return ONLY valid JSON, no other text."""
```

### Prompt Engineering Patterns Used

| Pattern | Example | Purpose |
|---------|---------|---------|
| **Role setting** | "You are an expert at..." | Activates relevant knowledge |
| **XML tags** | `<denial_letter>...</denial_letter>` | Clear input boundaries |
| **JSON schema** | Show exact output format | Structured, parseable output |
| **Negative examples** | "not zip codes" | Prevent common errors |
| **Explicit format** | "Return ONLY valid JSON" | Reduce wrapper text |

### Helper Functions

```python
def format_extraction_prompt(text: str) -> tuple[str, str]:
    """Format the extraction prompt with the given text.

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    return EXTRACT_FACTS_SYSTEM, EXTRACT_FACTS_USER.format(text=text)

# Usage:
system, user = format_extraction_prompt(denial_letter_text)
response = client.complete(user, system=system)
```

### The PromptLibrary Class

```python
class PromptLibrary:
    """Central registry of all prompts."""

    # Extraction
    EXTRACT_FACTS_SYSTEM = EXTRACT_FACTS_SYSTEM
    EXTRACT_FACTS_USER = EXTRACT_FACTS_USER

    # Generation
    GENERATE_PLAN_SYSTEM = GENERATE_PLAN_SYSTEM
    GENERATE_APPEAL_LETTER = GENERATE_APPEAL_LETTER

    @classmethod
    def get_all_prompts(cls) -> dict[str, str]:
        """Return all prompts as a dictionary."""
        return {name: value for name, value in vars(cls).items()
                if isinstance(value, str)}

    @classmethod
    def validate_prompt(cls, prompt: str, required_vars: list[str]) -> bool:
        """Validate that a prompt contains required variables."""
        return all(f"{{{var}}}" in prompt for var in required_vars)
```

### Key Learning

> **Treat prompts as code.** Version control them, test them, organize them. Prompts are the "source code" of LLM applications.

---

## 4. Code Walkthrough

### File: `src/denialops/llm/client.py`

```
Lines 1-15:    Imports and logger setup
Lines 17-108:  Token tracking (TokenUsage, UsageTracker)
Lines 110-175: Retry logic (RetryConfig, retry_with_backoff decorator)
Lines 177-193: LLMResponse dataclass
Lines 195-254: BaseLLMClient abstract class
Lines 256-365: OpenAIClient implementation
Lines 367-481: AnthropicClient implementation
Lines 483-566: LLMClient unified wrapper
Lines 568-599: create_llm_client factory function
```

### How a Request Flows

```
User calls client.complete(prompt)
    │
    ▼
LLMClient.complete()
    │
    ▼
OpenAIClient.complete()
    │
    ▼
@retry_with_backoff decorator
    │
    ├─► Try request
    │       │
    │       ▼
    │   _make_request()
    │       │
    │       ├─► Success: Track usage, return content
    │       │
    │       └─► Rate limit error: Wait, retry
    │
    └─► All retries failed: Raise exception
```

### File: `src/denialops/llm/prompts.py`

```
Lines 1-10:    Module docstring with best practices
Lines 12-80:   EXTRACT_FACTS prompts (system + user)
Lines 82-120:  Legacy extraction prompts
Lines 122-190: Action plan generation prompts
Lines 192-272: Document generation prompts
Lines 274-304: Grounding validation prompts
Lines 306-400: Helper functions (format_*_prompt)
Lines 402-450: PromptLibrary class
```

---

## 5. Key Takeaways

### For Senior AI/ML Engineers

1. **LLM APIs are unreliable** - Always implement retry logic with exponential backoff

2. **Cost matters at scale** - Track every token, estimate costs, set up alerts

3. **Prompts are code** - Version control them, test them, organize them centrally

4. **Abstraction enables flexibility** - The provider abstraction lets us switch OpenAI↔Anthropic without changing business logic

5. **Observability is critical** - Log every LLM call with tokens, latency, and cost

### Design Patterns Used

| Pattern | Where | Why |
|---------|-------|-----|
| **Factory** | `create_llm_client()` | Create objects without specifying exact class |
| **Decorator** | `@retry_with_backoff` | Add behavior without modifying function |
| **Abstract Base Class** | `BaseLLMClient` | Define interface for multiple implementations |
| **Dataclass** | `TokenUsage`, `LLMResponse` | Clean data containers with minimal boilerplate |
| **Lazy Initialization** | `@property def client` | Create expensive objects only when needed |

### Production Checklist

Before deploying LLM applications:

- [ ] Retry logic with exponential backoff
- [ ] Token usage tracking
- [ ] Cost estimation and alerts
- [ ] Latency monitoring
- [ ] Error logging with context
- [ ] Rate limiting (client-side)
- [ ] Timeout configuration
- [ ] Fallback to cheaper models when appropriate

---

## Next Steps

The remaining Phase 2 tasks are:

1. **LLM-powered summaries** - Use the LLM to generate personalized situation summaries instead of templates

2. **Grounding validation** - Check that generated content doesn't hallucinate codes, dates, or amounts not in source data

3. **Tests** - Unit tests for retry logic, token tracking, and prompt formatting

---

## Quick Reference

### Create a Client

```python
from denialops.llm import create_llm_client

client = create_llm_client(
    provider="openai",      # or "anthropic"
    api_key="sk-...",
    model="gpt-4o",         # optional
    max_retries=3,          # optional
)
```

### Make a Request with Usage Tracking

```python
response = client.complete_with_usage(
    prompt="Summarize this document...",
    system="You are a helpful assistant.",
    max_tokens=1000,
    temperature=0.0,
)

print(f"Content: {response.content}")
print(f"Tokens: {response.usage.total_tokens}")
print(f"Cost: ${response.usage.estimated_cost:.4f}")
print(f"Latency: {response.usage.latency_ms:.0f}ms")
```

### Get Aggregate Usage

```python
summary = client.get_usage_summary()
# {
#   "call_count": 10,
#   "total_tokens": 15000,
#   "estimated_cost_usd": 0.125,
#   "avg_latency_ms": 920.5
# }
```

### Format a Prompt

```python
from denialops.llm.prompts import format_extraction_prompt

system, user = format_extraction_prompt(denial_letter_text)
response = client.complete(user, system=system)
```
