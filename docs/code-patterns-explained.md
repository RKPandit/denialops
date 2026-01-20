# Code Patterns Explained

Line-by-line explanation of key patterns in DenialOps with "why" for every decision.

---

## 1. Configuration Pattern

### File: `src/denialops/config.py`

```python
from enum import Enum
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
```

**Why these imports?**
- `Enum`: Type-safe constants (better than string literals)
- `lru_cache`: Memoization to avoid re-reading config
- `Path`: Cross-platform file paths (Windows vs Unix)
- `pydantic_settings`: Environment variable binding with validation

```python
class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
```

**Why `str, Enum`?**
```python
# Without str:
LLMProvider.OPENAI        # <LLMProvider.OPENAI: 'openai'>
str(LLMProvider.OPENAI)   # "LLMProvider.OPENAI"  ❌ Not what we want

# With str:
LLMProvider.OPENAI        # <LLMProvider.OPENAI: 'openai'>
str(LLMProvider.OPENAI)   # "openai"  ✅ JSON serializable
```

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # API_PORT == api_port
    )
```

**Why `case_sensitive=False`?**
- Environment variables are traditionally UPPERCASE
- Python attributes are traditionally lowercase
- This lets us write `settings.api_port` but set `API_PORT=8000`

```python
    # Environment
    environment: str = "dev"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
```

**Why `0.0.0.0` as default host?**
- `127.0.0.1`: Only accepts connections from localhost
- `0.0.0.0`: Accepts connections from any interface (needed in Docker)

```python
    @property
    def is_dev(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "dev"
```

**Why a property instead of direct comparison?**
- Single source of truth for "what is dev mode"
- If logic changes (e.g., also include "local"), change one place
- Cleaner: `if settings.is_dev` vs `if settings.environment == "dev"`

```python
@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
```

**Why `@lru_cache`?**
- `Settings()` reads from environment/files (I/O)
- We want the same instance everywhere (singleton-ish)
- `lru_cache` with no arguments = cache forever

**Tradeoff**: Settings won't update if env vars change at runtime. Usually fine.

---

## 2. Dependency Injection Pattern

### File: `src/denialops/api/dependencies.py`

```python
from typing import Annotated
from fastapi import Depends, HTTPException, Path, status
```

**Why `Annotated`?**
```python
# Old way (still works):
def endpoint(storage: CaseStorage = Depends(get_storage)):
    ...

# New way with Annotated (cleaner for reuse):
StorageDep = Annotated[CaseStorage, Depends(get_storage)]

def endpoint(storage: StorageDep):
    ...
```

```python
def get_storage(settings: Annotated[Settings, Depends(get_settings)]) -> CaseStorage:
    """Get storage instance."""
    return CaseStorage(settings.artifacts_path)
```

**Dependency chain:**
```
endpoint
    └── needs StorageDep
            └── calls get_storage()
                    └── needs Settings
                            └── calls get_settings()
```

FastAPI resolves this automatically!

```python
def validate_case_exists(
    case_id: Annotated[str, Path(description="Case ID")],
    storage: Annotated[CaseStorage, Depends(get_storage)],
) -> str:
    """Validate that a case exists, return case_id if valid."""
    if not storage.case_exists(case_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case {case_id} not found",
        )
    return case_id
```

**Why a dependency for validation?**

Without (repeated code):
```python
@router.get("/cases/{case_id}/artifacts")
def get_artifacts(case_id: str, storage: StorageDep):
    if not storage.case_exists(case_id):
        raise HTTPException(404, "Not found")
    ...

@router.post("/cases/{case_id}/run")
def run_pipeline(case_id: str, storage: StorageDep):
    if not storage.case_exists(case_id):  # Repeated!
        raise HTTPException(404, "Not found")
    ...
```

With dependency (DRY):
```python
@router.get("/cases/{case_id}/artifacts")
def get_artifacts(case_id: CaseIdDep, storage: StorageDep):
    # case_id is guaranteed to exist!
    ...
```

---

## 3. Abstract Base Class Pattern

### File: `src/denialops/llm/client.py`

```python
from abc import ABC, abstractmethod

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
```

**Why ABC (Abstract Base Class)?**

```python
# Without ABC - runtime error
class BadClient:
    pass

client = BadClient()
client.complete("hi")  # AttributeError at runtime 💥

# With ABC - error at instantiation
class BadClient(BaseLLMClient):
    pass  # Forgot to implement complete()

client = BadClient()  # TypeError: Can't instantiate abstract class 💥
```

**Why `@abstractmethod`?**
- Forces subclasses to implement this method
- Documents "this is the interface you must follow"
- IDE shows error if you forget

```python
class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self._client: openai.OpenAI | None = None  # Lazy initialization
```

**Why lazy initialization?**
```python
# Eager (creates client immediately):
def __init__(self, api_key: str):
    self._client = openai.OpenAI(api_key=api_key)

# Lazy (creates client on first use):
@property
def client(self) -> openai.OpenAI:
    if self._client is None:
        self._client = openai.OpenAI(api_key=self.api_key)
    return self._client
```

**Benefits of lazy:**
- Faster startup (don't create what you don't use)
- Can create instance without valid API key (for testing)
- Connection established only when needed

---

## 4. Decorator Pattern (Wrapper)

### File: `src/denialops/llm/cached_client.py`

```python
class CachedLLMClient(BaseLLMClient):
    """LLM client wrapper that adds caching."""

    def __init__(
        self,
        client: BaseLLMClient,  # Wraps another client
        cache: CacheBackend,
        ...
    ):
        self._client = client
        self._cache = cache
```

**Visual representation:**
```
┌─────────────────────────────────────────┐
│           CachedLLMClient               │
│  ┌─────────────────────────────────┐   │
│  │         OpenAIClient            │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │         MemoryCache             │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

```python
    def complete(self, prompt: str, ...) -> str:
        # 1. Check if we should cache this request
        if not self._should_cache(temperature):
            return self._client.complete(prompt, ...)

        # 2. Try to get from cache
        cache_key = self._get_cache_key(prompt, system)
        cached = self._cache.get(cache_key)
        if cached:
            self._cache_hits += 1
            return cached.content

        # 3. Call underlying client
        result = self._client.complete(prompt, ...)

        # 4. Store in cache
        self._cache.set(cache_key, CachedResponse(...))

        return result
```

**Why decorator pattern over inheritance?**

```python
# Inheritance approach (inflexible):
class CachedOpenAIClient(OpenAIClient):
    ...
class CachedAnthropicClient(AnthropicClient):
    ...
class RetryingCachedOpenAIClient(CachedOpenAIClient):
    ...  # Combinatorial explosion!

# Decorator approach (composable):
client = CachedLLMClient(
    client=RetryingLLMClient(
        client=OpenAIClient(api_key)
    ),
    cache=MemoryCache()
)
```

**Key insight**: Decorators add behavior, inheritance adds identity.

---

## 5. Factory Pattern

### File: `src/denialops/llm/client.py`

```python
def create_llm_client(
    provider: LLMProvider | str,
    api_key: str,
    model: str | None = None,
    max_retries: int = 3,
) -> LLMClient:
    """Factory function to create an LLM client."""
    if isinstance(provider, str):
        provider = LLMProvider(provider.lower())

    retry_config = RetryConfig(max_retries=max_retries)

    return LLMClient(
        provider=provider,
        api_key=api_key,
        model=model,
        retry_config=retry_config,
    )
```

**Why a factory function?**

Without factory (knowledge spread everywhere):
```python
# In routes.py
client = LLMClient(
    provider=LLMProvider(settings.llm_provider),
    api_key=settings.openai_api_key,
    model=settings.llm_model,
    retry_config=RetryConfig(max_retries=3),
)

# In pipeline.py (duplicated!)
client = LLMClient(
    provider=LLMProvider(settings.llm_provider),
    ...
)
```

With factory (centralized):
```python
# In routes.py
client = create_llm_client(settings.llm_provider, settings.llm_api_key)

# In pipeline.py
client = create_llm_client(settings.llm_provider, settings.llm_api_key)
```

**Benefits:**
- Single place to change defaults
- Hide complex construction
- Can return different types based on input

---

## 6. Data Class Patterns

### File: `src/denialops/models/case.py`

```python
from pydantic import BaseModel, Field

class CaseFacts(BaseModel):
    """Extracted facts from a denial case."""

    case_id: str = Field(..., description="Unique case identifier")
    denial_reason: DenialReason | None = Field(None, description="Why claim was denied")
    dates: DenialDates | None = None
    amounts: DenialAmounts | None = None
```

**Why Pydantic over dataclass?**

```python
# dataclass - basic
@dataclass
class Person:
    name: str
    age: int

Person(name="Alice", age="not a number")  # No error! 💥

# Pydantic - validated
class Person(BaseModel):
    name: str
    age: int

Person(name="Alice", age="not a number")  # ValidationError ✅
```

**What `Field(...)` does:**
```python
# Required field (no default)
case_id: str = Field(...)  # ... means "required"

# Optional with default
denial_reason: DenialReason | None = Field(None)

# With metadata
case_id: str = Field(..., description="Unique ID", min_length=1)
```

**Why descriptions?**
- Auto-generated API docs (OpenAPI/Swagger)
- Self-documenting code
- IDE hints

---

## 7. Error Handling Pattern

### File: `src/denialops/api/routes.py`

```python
@router.post("/cases/{case_id}/run")
async def run_pipeline(case_id: CaseIdDep, storage: StorageDep, settings: SettingsDep):
    try:
        # ... pipeline logic ...
        return RunPipelineResponse(status="completed", ...)

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is

    except Exception as e:
        return RunPipelineResponse(
            status="failed",
            error=str(e),
            artifacts=[],
        )
```

**Why this pattern?**

```python
# Option 1: Let exceptions bubble up
@router.post("/run")
async def run_pipeline(...):
    result = do_work()  # If this throws, client sees 500 Internal Server Error
    return result

# Option 2: Catch and return error response (current)
@router.post("/run")
async def run_pipeline(...):
    try:
        result = do_work()
        return SuccessResponse(result=result)
    except Exception as e:
        return ErrorResponse(error=str(e))  # Client sees structured error
```

**Tradeoff:**

| Approach | Pros | Cons |
|----------|------|------|
| Bubble up | Simple, stack trace in logs | Client sees ugly 500 |
| Catch all | Client sees nice error | May hide bugs |
| Selective catch | Best of both | More code |

**Best practice:**
```python
try:
    ...
except KnownBusinessError as e:
    return ErrorResponse(error=str(e))  # Expected errors
except Exception as e:
    logger.exception("Unexpected error")  # Log full stack trace
    raise  # Let it become 500 (alerts us to fix it)
```

---

## 8. Async Patterns

### File: `src/denialops/tasks/manager.py`

```python
def submit(self, coro: Coroutine[Any, Any, Any], task_id: str | None = None) -> str:
    task_id = task_id or str(uuid.uuid4())

    async def run_task() -> None:
        try:
            self._tasks[task_id].status = TaskStatus.RUNNING
            task_result = await coro  # Actually runs the coroutine
            self._tasks[task_id].status = TaskStatus.COMPLETED
            self._tasks[task_id].result = task_result
        except asyncio.CancelledError:
            self._tasks[task_id].status = TaskStatus.CANCELLED
        except Exception as e:
            self._tasks[task_id].status = TaskStatus.FAILED
            self._tasks[task_id].error = str(e)

    loop = asyncio.get_event_loop()
    asyncio_task = loop.create_task(run_task())
    self._running_tasks[task_id] = asyncio_task

    return task_id
```

**Key concepts:**

1. **Coroutine vs Task:**
```python
async def my_func():
    return "hello"

coro = my_func()      # Coroutine object (not running yet!)
task = asyncio.create_task(coro)  # Now it's scheduled to run
```

2. **Why wrap in `run_task`?**
```python
# Direct approach - can't track status
asyncio_task = loop.create_task(coro)

# Wrapper approach - full control
async def run_task():
    update_status("running")
    try:
        result = await coro
        update_status("completed")
    except:
        update_status("failed")
```

3. **Why `asyncio.CancelledError`?**
```python
# When someone calls task.cancel():
await asyncio.sleep(100)  # This will raise CancelledError
```
Catching it lets us update status properly.

---

## 9. Thread Safety Pattern

### File: `src/denialops/cache/memory.py`

```python
from threading import Lock

class MemoryCache(CacheBackend):
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self._cache: dict[str, CachedResponse] = {}
        self._access_order: list[str] = []
        self._lock = Lock()  # Thread safety
        self._hits = 0
        self._misses = 0
```

**Why do we need locks?**

```python
# Without lock - race condition:
# Thread 1: reads _cache["key"] = None
# Thread 2: sets _cache["key"] = "value"
# Thread 1: sets _cache["key"] = "value2"  # Overwrites Thread 2!

# With lock - safe:
with self._lock:
    if key not in self._cache:
        self._cache[key] = compute_value()
```

```python
def get(self, key: CacheKey) -> CachedResponse | None:
    key_str = key.to_string()

    with self._lock:
        if key_str not in self._cache:
            self._misses += 1
            return None

        response = self._cache[key_str]

        # Check expiration
        if time.time() > response.cached_at + response.ttl:
            del self._cache[key_str]
            self._misses += 1
            return None

        # Update access order for LRU
        if key_str in self._access_order:
            self._access_order.remove(key_str)
        self._access_order.append(key_str)

        self._hits += 1
        return response
```

**Why `with self._lock`?**
```python
# This is equivalent to:
self._lock.acquire()
try:
    # ... do stuff ...
finally:
    self._lock.release()  # Always releases, even on exception
```

**Tradeoff**: Locks add overhead. For high-throughput, consider:
- `RLock` (reentrant lock)
- `threading.local()` (per-thread storage)
- Lock-free data structures
- External cache (Redis)

---

## 10. Singleton Pattern

### File: `src/denialops/tasks/manager.py`

```python
class TaskManager:
    _instance: "TaskManager | None" = None
    _lock = Lock()

    def __new__(cls) -> "TaskManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        # ... initialization ...
        self._initialized = True
```

**Why singleton for TaskManager?**
- Global task registry
- All code sees same task state
- No need to pass manager everywhere

**Why thread-safe singleton?**
```python
# Without lock - possible race condition:
# Thread 1: checks _instance is None ✓
# Thread 2: checks _instance is None ✓
# Thread 1: creates instance
# Thread 2: creates another instance!  💥

# With lock - safe:
with cls._lock:
    if cls._instance is None:
        cls._instance = ...
```

**Why check `_initialized`?**
```python
# __new__ returns existing instance
# But __init__ still gets called!

manager1 = TaskManager()  # __new__ creates, __init__ initializes
manager2 = TaskManager()  # __new__ returns existing, __init__ would re-init!

# Solution: guard in __init__
def __init__(self):
    if self._initialized:
        return  # Skip re-initialization
```

**Singleton downsides:**
- Global state (harder to test)
- Hidden dependencies
- Can't have multiple instances

**Alternative for testing:**
```python
# In tests, reset singleton:
TaskManager._instance = None
```

---

## Summary: Pattern Decision Tree

```
Need to choose implementation at runtime?
├─ Yes → Strategy Pattern (LLM clients)
└─ No
   │
   Need to add behavior without modifying class?
   ├─ Yes → Decorator Pattern (CachedLLMClient)
   └─ No
      │
      Need exactly one instance?
      ├─ Yes → Singleton (TaskManager)
      └─ No
         │
         Need to hide complex construction?
         ├─ Yes → Factory (create_llm_client)
         └─ No
            │
            Need to define interface for multiple implementations?
            ├─ Yes → Abstract Base Class (BaseLLMClient)
            └─ No → Keep it simple!
```

---

## Next Steps

1. **Read code with this guide open** - Match patterns to code
2. **Try to break patterns** - What happens if you violate them?
3. **Implement a new feature** - Which patterns help?
4. **Refactor without patterns** - Feel the pain, then add them back
