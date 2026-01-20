# Interview Topics Demonstrated in DenialOps

This project demonstrates concepts commonly asked in software engineering interviews. Use this as a study guide.

---

## System Design Topics

### 1. API Design

**Interview Question**: "Design an API for a document processing system"

**What DenialOps demonstrates:**
- RESTful resource naming (`/cases`, `/cases/{id}/documents`)
- Appropriate HTTP methods (POST for create, GET for read)
- Status codes (201 Created, 404 Not Found, 202 Accepted)
- Versioning (`/api/v1/`)
- Request/response models

**Key files:** `api/routes.py`, `api/health.py`

**Discussion points:**
```
Interviewer: "Why POST for /run instead of GET?"
You: "GET should be idempotent and cacheable. Running a pipeline
     has side effects (creates artifacts, costs money), so POST
     is appropriate. Also, we might want to pass options in the body."

Interviewer: "Why return 202 Accepted for async operations?"
You: "202 means 'request received, processing started, but not complete.'
     The client can poll the task endpoint to check progress.
     Returning 200 would imply the work is done."
```

### 2. Caching

**Interview Question**: "How would you add caching to reduce latency?"

**What DenialOps demonstrates:**
- Cache key design (hashing for privacy)
- TTL-based expiration
- LRU eviction policy
- Cache abstraction (memory vs Redis)
- Cache-aside pattern

**Key files:** `cache/base.py`, `cache/memory.py`, `llm/cached_client.py`

**Discussion points:**
```
Interviewer: "How do you decide what to cache?"
You: "For LLM responses, I cache low-temperature (deterministic) calls
     since the same prompt gives the same response. High-temperature
     calls are intentionally random, so caching would be wrong."

Interviewer: "How do you handle cache invalidation?"
You: "For LLM caches, TTL-based invalidation is sufficient because
     responses don't change for the same model version. For user data,
     I'd use write-through or event-based invalidation."
```

### 3. Background Processing

**Interview Question**: "How would you handle long-running operations?"

**What DenialOps demonstrates:**
- Async task submission
- Task status tracking
- Progress reporting
- Cancellation support
- Result retrieval

**Key files:** `tasks/manager.py`, `tasks/pipeline.py`, `api/tasks.py`

**Discussion points:**
```
Interviewer: "What if the server crashes during task execution?"
You: "Current implementation loses in-progress tasks. For production,
     I'd use a persistent queue like Celery+Redis or AWS SQS. Tasks
     would be checkpointed, and another worker could resume."

Interviewer: "How would you scale to 1000 concurrent tasks?"
You: "Current in-memory approach won't scale. I'd:
     1. Use distributed task queue (Celery)
     2. Separate worker processes
     3. Horizontal scaling of workers
     4. Task prioritization for important cases"
```

### 4. Health Checks & Monitoring

**Interview Question**: "How do you know if your service is healthy?"

**What DenialOps demonstrates:**
- Liveness vs readiness probes
- Component health checks
- Graceful degradation
- Safe config exposure

**Key files:** `api/health.py`

**Discussion points:**
```
Interviewer: "Why separate liveness and readiness?"
You: "Liveness asks 'is the process running?' - if no, restart it.
     Readiness asks 'can it handle traffic?' - if no, stop sending traffic.

     Example: App starts (liveness passes) but database isn't connected
     yet (readiness fails). We don't want to restart the app, just
     wait for the database."
```

---

## Coding Pattern Topics

### 5. SOLID Principles

**S - Single Responsibility**
```python
# Each class has one job:
class OpenAIClient:      # Talk to OpenAI
class CaseStorage:       # Store files
class DenialRouter:      # Route cases
```

**O - Open/Closed**
```python
# Open for extension, closed for modification:
class BaseLLMClient(ABC):   # Closed - don't modify
class NewProvider(BaseLLMClient):  # Open - extend it
```

**L - Liskov Substitution**
```python
# Subclasses can replace parent:
def process(client: BaseLLMClient):
    client.complete(...)  # Works with any implementation
```

**I - Interface Segregation**
```python
# Small, focused interfaces:
class CacheBackend(ABC):
    def get(...): ...
    def set(...): ...
    # Not: def get_all(), def migrate(), def backup()
```

**D - Dependency Inversion**
```python
# Depend on abstractions:
class CachedLLMClient:
    def __init__(self, client: BaseLLMClient, cache: CacheBackend):
        # Not: def __init__(self, openai_client: OpenAIClient, redis: Redis)
```

### 6. Design Patterns

| Pattern | Where Used | Why |
|---------|------------|-----|
| **Strategy** | LLM clients | Swap providers at runtime |
| **Decorator** | CachedLLMClient | Add caching without modifying client |
| **Factory** | create_llm_client() | Hide complex construction |
| **Singleton** | TaskManager | Global task registry |
| **Repository** | CaseStorage | Abstract data access |
| **Template Method** | Pipeline stages | Common flow, specific steps |

### 7. Dependency Injection

**Interview Question**: "Why use dependency injection?"

```python
# Without DI (hard to test):
class CaseService:
    def __init__(self):
        self.storage = CaseStorage("/data")  # Hardcoded!
        self.llm = OpenAIClient(os.environ["KEY"])  # Hardcoded!

# With DI (testable):
class CaseService:
    def __init__(self, storage: CaseStorage, llm: BaseLLMClient):
        self.storage = storage
        self.llm = llm

# In tests:
service = CaseService(MockStorage(), MockLLM())
```

**FastAPI DI:**
```python
# Framework handles injection:
def endpoint(storage: StorageDep, settings: SettingsDep):
    # storage and settings are injected automatically
```

---

## Data Structure Topics

### 8. LRU Cache Implementation

**Interview Question**: "Implement an LRU cache"

**Key insight**: Need O(1) for both get and put

```python
# DenialOps approach (simple):
class MemoryCache:
    _cache: dict[str, Value]       # O(1) lookup
    _access_order: list[str]       # Track recency

    def get(self, key):
        # O(n) to update access order - not optimal!
        self._access_order.remove(key)
        self._access_order.append(key)
```

**Optimal approach (OrderedDict or doubly-linked list):**
```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)  # O(1)!
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Remove oldest
```

### 9. Hash-Based Cache Keys

**Interview Question**: "How would you design cache keys?"

```python
# Bad: Raw prompt as key
cache_key = prompt  # Privacy issue, potentially huge key

# Good: Hash the prompt
prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
cache_key = f"{provider}:{model}:{prompt_hash}"
```

**Discussion:**
- Why hash? Privacy, fixed size
- Why truncate? Full SHA256 is 64 chars, 16 is enough
- Collision risk? 16 hex chars = 64 bits = 18 quintillion possibilities

---

## Concurrency Topics

### 10. Thread Safety

**Interview Question**: "Is this code thread-safe?"

```python
# Not thread-safe:
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1  # Read, add, write - not atomic!

# Thread-safe:
from threading import Lock

class Counter:
    def __init__(self):
        self.count = 0
        self._lock = Lock()

    def increment(self):
        with self._lock:
            self.count += 1
```

**Where DenialOps handles this:**
- `MemoryCache` uses Lock for cache operations
- `TaskManager` uses Lock for task registry

### 11. Async/Await

**Interview Question**: "Explain async/await"

```python
# Sync - blocks the thread:
def fetch_data():
    response = requests.get(url)  # Thread waits here
    return response.json()

# Async - yields control:
async def fetch_data():
    response = await httpx.get(url)  # Thread can do other work
    return response.json()
```

**Key concepts in DenialOps:**
- Async endpoints (FastAPI)
- Background tasks (asyncio.create_task)
- Streaming responses (async generators)

---

## Testing Topics

### 12. Test Isolation

**Interview Question**: "How do you test code with external dependencies?"

```python
# Option 1: Mocking
def test_with_mock():
    mock_llm = Mock()
    mock_llm.complete.return_value = "mocked response"
    result = process(mock_llm)
    assert result == expected

# Option 2: Dependency injection
def test_with_fake():
    fake_llm = FakeLLMClient(responses={"prompt": "response"})
    result = process(fake_llm)
    assert result == expected

# Option 3: Integration test
def test_integration():
    real_llm = create_llm_client(...)  # Uses real API
    result = process(real_llm)
    # More confidence, but slow and flaky
```

**DenialOps approach:**
- Heuristic fallbacks when no API key
- Test client fixtures
- Isolated test data directories

### 13. Test Fixtures

**Interview Question**: "How do you manage test data?"

```python
# pytest fixtures in DenialOps:
@pytest.fixture
def client():
    """Provides test HTTP client"""
    with TestClient(app) as client:
        yield client
    # Cleanup happens after yield

@pytest.fixture
def sample_case(client):
    """Creates a case and returns its ID"""
    response = client.post("/api/v1/cases", json={"mode": "fast"})
    return response.json()["case_id"]

def test_something(client, sample_case):
    # Both fixtures are injected
    response = client.get(f"/api/v1/cases/{sample_case}/artifacts")
```

---

## Production Topics

### 14. Configuration Management

**Interview Question**: "How do you handle configuration?"

**12-Factor App approach (used in DenialOps):**
1. Store config in environment variables
2. Same code runs in all environments
3. No secrets in code

```python
# Development
API_PORT=8000
CACHE_BACKEND=memory

# Production
API_PORT=80
CACHE_BACKEND=redis
REDIS_URL=redis://prod-redis:6379
```

### 15. Error Handling

**Interview Question**: "How do you handle errors in production?"

**Principles demonstrated:**
```python
# 1. Don't expose internal errors to users
except Exception as e:
    logger.exception("Internal error")  # Full details in logs
    return {"error": "Something went wrong"}  # Safe message to user

# 2. Use appropriate status codes
raise HTTPException(status_code=404, detail="Case not found")
raise HTTPException(status_code=503, detail="Service unavailable")

# 3. Graceful degradation
if not llm_available:
    return heuristic_result()  # Fallback, not failure
```

### 16. Containerization

**Interview Question**: "Why use Docker? How would you deploy this?"

**Benefits demonstrated:**
- Reproducible builds (same everywhere)
- Isolation (dependencies don't conflict)
- Easy deployment (ship container, not code)

**Multi-stage build:**
```dockerfile
FROM python:3.11-slim as builder
# Install build tools, compile, etc.

FROM python:3.11-slim as production
# Copy only what's needed - smaller image
```

---

## Practice Questions

### Easy
1. What HTTP method would you use to create a new case?
2. Why use Pydantic instead of plain dictionaries?
3. What's the difference between `@property` and a regular method?

### Medium
4. Explain how the decorator pattern is used in CachedLLMClient
5. Why do we need both liveness and readiness endpoints?
6. How would you test the LLM integration without making real API calls?

### Hard
7. Design a system to process 10,000 cases per day with <5 min latency
8. How would you add real-time progress updates using WebSockets?
9. What changes would be needed to run this in a serverless environment?

---

## How to Use This for Interview Prep

1. **Code walkthrough practice**: Explain each file as if in an interview
2. **Whiteboard design**: Draw the architecture from memory
3. **Trade-off discussions**: For each decision, know the alternatives
4. **Extend the system**: Practice adding features (see exercises doc)
5. **Debug scenarios**: "The LLM is slow, how do you investigate?"

Remember: Interviewers care more about **how you think** than the specific answer.
