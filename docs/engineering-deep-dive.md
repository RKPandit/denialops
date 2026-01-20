# DenialOps Engineering Deep Dive

A comprehensive guide to understanding every architectural decision, tradeoff, and alternative approach in this project.

## Table of Contents

1. [Project Architecture](#1-project-architecture)
2. [API Design Decisions](#2-api-design-decisions)
3. [Data Layer Patterns](#3-data-layer-patterns)
4. [LLM Integration Patterns](#4-llm-integration-patterns)
5. [Caching Strategies](#5-caching-strategies)
6. [Async & Background Processing](#6-async--background-processing)
7. [Testing Strategies](#7-testing-strategies)
8. [Production Readiness](#8-production-readiness)
9. [What We Could Do Differently](#9-what-we-could-do-differently)

---

## 1. Project Architecture

### Current Structure
```
src/denialops/
├── api/           # HTTP layer (FastAPI routes)
├── cache/         # Caching abstraction
├── llm/           # LLM client wrappers
├── models/        # Pydantic data models
├── pipeline/      # Business logic (extraction, routing, generation)
├── tasks/         # Background task processing
├── utils/         # Shared utilities (storage, etc.)
├── config.py      # Configuration management
└── main.py        # Application entry point
```

### Why This Structure?

**Layered Architecture** - We separate concerns into layers:
```
HTTP Layer (api/) → Business Logic (pipeline/) → Data/External (llm/, storage)
```

**Why layers matter:**
- **Testability**: You can test business logic without HTTP
- **Flexibility**: Swap FastAPI for Flask without touching business logic
- **Clarity**: New developers know where to find things

### Alternative Architectures

| Architecture | Pros | Cons | When to Use |
|--------------|------|------|-------------|
| **Layered** (current) | Simple, clear boundaries | Can become rigid | Small-medium projects |
| **Hexagonal/Ports & Adapters** | Very flexible, testable | More complex | Large projects, many integrations |
| **Microservices** | Independent scaling, team autonomy | Network complexity, eventual consistency | Large teams, high scale |
| **Serverless** | No server management, auto-scaling | Cold starts, vendor lock-in | Event-driven, variable load |

### What a Senior Engineer Would Ask

1. "Will this architecture scale to 10 engineers working simultaneously?"
2. "Can we deploy parts independently?"
3. "What happens when requirements change dramatically?"

**Honest answer for DenialOps**: Our layered approach is appropriate for a small team (1-5 engineers) and moderate complexity. If this grew to 20+ engineers, we'd consider splitting into services.

---

## 2. API Design Decisions

### RESTful Design

We use REST with resource-based URLs:
```
POST /api/v1/cases              # Create a case
POST /api/v1/cases/{id}/documents  # Upload document to case
POST /api/v1/cases/{id}/run     # Run pipeline
GET  /api/v1/cases/{id}/artifacts  # List results
```

### Why REST over alternatives?

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **REST** (current) | Simple, cacheable, well-understood | Over/under fetching, many round trips | CRUD operations, public APIs |
| **GraphQL** | Client specifies data needs, single endpoint | Complex, caching harder, learning curve | Complex data relationships, mobile apps |
| **gRPC** | Fast (binary), strongly typed, streaming | Browser support limited, harder to debug | Internal services, high performance |
| **WebSocket** | Real-time bidirectional | Connection management, stateful | Chat, live updates, gaming |

### API Versioning

We use URL versioning: `/api/v1/cases`

**Alternatives:**
```python
# URL versioning (current)
/api/v1/cases
/api/v2/cases

# Header versioning
GET /api/cases
Accept: application/vnd.denialops.v1+json

# Query parameter
GET /api/cases?version=1
```

**Tradeoff Analysis:**

| Method | Pros | Cons |
|--------|------|------|
| URL (current) | Obvious, easy to test in browser | URLs change, breaks bookmarks |
| Header | Clean URLs, HTTP-compliant | Hidden, harder to test |
| Query param | Simple | Not RESTful, can be cached incorrectly |

**Why we chose URL**: Explicitness wins for developer experience. When debugging, you immediately see which version you're hitting.

### Response Design Decision

We return structured responses:
```python
class CreateCaseResponse(BaseModel):
    case_id: str
    mode: CaseMode
    created_at: datetime
```

**Why Pydantic models over dicts?**
- **Type safety**: IDE autocomplete, catch errors early
- **Validation**: Automatic request validation
- **Documentation**: Auto-generated OpenAPI docs
- **Serialization**: Handles datetime, enums automatically

**Alternative**: Plain dicts are faster to write but lose all these benefits.

---

## 3. Data Layer Patterns

### Current Approach: File-Based Storage

```python
class CaseStorage:
    def __init__(self, base_path: Path):
        self.base_path = base_path

    def store_artifact(self, case_id: str, filename: str, data: Any):
        path = self.base_path / case_id / filename
        # Write to file
```

### Why File Storage?

**For this project:**
- Simple to implement and debug
- No database setup required
- Easy to inspect data (just look at files)
- Artifacts (PDFs, generated docs) are naturally files

### When to Use What

| Storage | Pros | Cons | Best For |
|---------|------|------|----------|
| **Files** (current) | Simple, portable, inspectable | No queries, no transactions, scaling limits | Artifacts, small projects |
| **SQLite** | SQL queries, ACID, single file | Single writer, no network access | Local apps, prototypes |
| **PostgreSQL** | Full SQL, ACID, concurrent writes | Setup complexity, operational overhead | Production web apps |
| **MongoDB** | Flexible schema, good for documents | No joins, eventual consistency modes | Document-heavy, evolving schemas |
| **S3/GCS** | Unlimited scale, durability, CDN | Network latency, eventual consistency | Large files, static assets |

### Repository Pattern (What We Could Add)

```python
# Abstract interface
class CaseRepository(ABC):
    @abstractmethod
    def save(self, case: Case) -> None: ...

    @abstractmethod
    def find_by_id(self, case_id: str) -> Case | None: ...

# File implementation
class FileCaseRepository(CaseRepository):
    def save(self, case: Case) -> None:
        # Write to file

# Database implementation
class PostgresCaseRepository(CaseRepository):
    def save(self, case: Case) -> None:
        # Write to database
```

**Why we didn't do this**: Added complexity for a single storage backend. Add this abstraction when you need multiple backends.

---

## 4. LLM Integration Patterns

### Current Design

```python
class BaseLLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, system: str | None = None) -> str: ...

class OpenAIClient(BaseLLMClient): ...
class AnthropicClient(BaseLLMClient): ...

class LLMClient:
    """Unified client that delegates to provider-specific implementation"""
    def __init__(self, provider: LLMProvider, api_key: str):
        if provider == LLMProvider.OPENAI:
            self._client = OpenAIClient(api_key)
        else:
            self._client = AnthropicClient(api_key)
```

### Why This Pattern?

**Strategy Pattern**: The `LLMClient` delegates to interchangeable implementations.

```
┌─────────────┐     uses      ┌───────────────┐
│  LLMClient  │──────────────▶│ BaseLLMClient │
└─────────────┘               └───────────────┘
                                     △
                                     │ implements
                    ┌────────────────┼────────────────┐
                    │                │                │
             ┌──────┴─────┐  ┌───────┴──────┐  ┌─────┴──────┐
             │OpenAIClient│  │AnthropicClient│  │MockClient  │
             └────────────┘  └──────────────┘  └────────────┘
```

**Benefits:**
- Add new providers without changing calling code
- Easy to mock for testing
- Provider-specific optimizations possible

### Wrapper Pattern for Cross-Cutting Concerns

```python
class CachedLLMClient(BaseLLMClient):
    def __init__(self, client: BaseLLMClient, cache: CacheBackend):
        self._client = client
        self._cache = cache

    def complete(self, prompt: str, ...) -> str:
        # Check cache first
        cached = self._cache.get(cache_key)
        if cached:
            return cached.content

        # Call underlying client
        result = self._client.complete(prompt, ...)

        # Store in cache
        self._cache.set(cache_key, result)
        return result
```

**This is the Decorator Pattern**: Add behavior without modifying original class.

**Why not inheritance?**
```python
# Bad: Inheritance creates rigid hierarchies
class CachedOpenAIClient(OpenAIClient):
    # Now you need CachedAnthropicClient too
    # What about CachedRetryingOpenAIClient?
```

**Composition is more flexible:**
```python
# Good: Compose behaviors
client = CachedLLMClient(
    RetryingLLMClient(
        OpenAIClient(api_key)
    ),
    cache=MemoryCache()
)
```

### Retry Logic Analysis

```python
class RetryConfig:
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
```

**Why exponential backoff?**

```
Attempt 1: Wait 1 second
Attempt 2: Wait 2 seconds
Attempt 3: Wait 4 seconds
Attempt 4: Wait 8 seconds
...
```

**Alternatives:**
| Strategy | Pattern | Use Case |
|----------|---------|----------|
| **Fixed delay** | 1s, 1s, 1s | Simple, predictable load |
| **Exponential** (current) | 1s, 2s, 4s, 8s | Rate limits, cascading failures |
| **Exponential + jitter** | 1s±rand, 2s±rand | Many clients, thundering herd |
| **Circuit breaker** | Stop trying after N failures | Failing dependencies |

**What we should add**: Jitter to prevent thundering herd:
```python
delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
```

---

## 5. Caching Strategies

### Cache Key Design

```python
@dataclass
class CacheKey:
    provider: str
    model: str
    prompt_hash: str  # SHA256 truncated to 16 chars
    system_hash: str | None
```

**Why hash the prompts?**
1. **Privacy**: Don't store raw prompts in cache keys
2. **Size**: Fixed-size keys regardless of prompt length
3. **Speed**: String comparison is faster with fixed-size keys

**Why truncate to 16 chars?**
- Full SHA256 = 64 chars (256 bits)
- 16 chars = 64 bits = 18 quintillion possibilities
- Collision probability is negligible for our scale

### Cache Invalidation

> "There are only two hard things in Computer Science: cache invalidation and naming things." — Phil Karlton

**Our approach**: TTL-based expiration
```python
cached_at: float  # Unix timestamp
ttl: int  # Seconds until expiration
```

**Alternatives:**

| Strategy | How It Works | Pros | Cons |
|----------|--------------|------|------|
| **TTL** (current) | Expire after N seconds | Simple, predictable | May serve stale data |
| **LRU** (also current) | Evict least recently used | Memory bounded | Popular items may expire |
| **Write-through** | Update cache on every write | Always fresh | Write latency |
| **Cache-aside** | App manages cache explicitly | Flexible | More code |
| **Event-based** | Invalidate on data change | Always fresh | Complex, needs events |

**Why we chose TTL + LRU:**
- LLM responses don't change (same prompt → same response for low temperature)
- TTL handles model updates (new model version = different responses)
- LRU prevents memory overflow

### Memory vs Redis

```python
class MemoryCache(CacheBackend):
    """In-process cache - fast but not shared"""

class RedisCache(CacheBackend):
    """External cache - shared across processes"""
```

**Decision Matrix:**

| Scenario | Use Memory | Use Redis |
|----------|------------|-----------|
| Single server | ✅ | ❌ Overkill |
| Multiple servers | ❌ Duplicated | ✅ Shared |
| Serverless | ❌ No persistence | ✅ Persistent |
| Development | ✅ Simple | ❌ Extra setup |
| Cache size > RAM | ❌ OOM risk | ✅ Can be larger |

---

## 6. Async & Background Processing

### Why Background Tasks?

**Problem**: Pipeline takes 10-30 seconds (multiple LLM calls)

**Without background tasks:**
```
Client ──POST /run──▶ Server ──────────────▶ Response
                      │                      │
                      └──── 30 seconds ──────┘

HTTP timeout risk, poor UX
```

**With background tasks:**
```
Client ──POST /run/async──▶ Server ──▶ 202 Accepted + task_id
       ──GET /tasks/{id}───▶ Server ──▶ {status: "running", progress: 0.5}
       ──GET /tasks/{id}───▶ Server ──▶ {status: "completed", result: {...}}
```

### Task Manager Design

```python
class TaskManager:
    _instance: "TaskManager | None" = None  # Singleton
    _tasks: dict[str, TaskResult] = {}
    _running_tasks: dict[str, asyncio.Task] = {}
```

**Why Singleton?**
- Global task registry accessible from anywhere
- Prevents duplicate task managers
- Simple for in-process task management

**Tradeoff**: Singletons are global state, harder to test. We accept this for simplicity.

### Alternatives to Our Approach

| Approach | How It Works | Pros | Cons |
|----------|--------------|------|------|
| **In-process** (current) | asyncio tasks in same process | Simple, no infra | Lost on restart, single server |
| **Celery + Redis** | Distributed task queue | Reliable, scalable | Complex setup, more infra |
| **AWS SQS + Lambda** | Cloud-native queue | Serverless, scalable | Vendor lock-in, cold starts |
| **Temporal** | Workflow orchestration | Durable, complex workflows | Learning curve, infra |
| **PostgreSQL + polling** | Tasks in DB, workers poll | Simple, durable | Polling overhead |

**When to upgrade from in-process:**
- Need task persistence across restarts
- Need horizontal scaling (multiple servers)
- Tasks take > 5 minutes (long-running)

### Progress Tracking

```python
def update_progress(self, task_id: str, progress: float, message: str):
    self._tasks[task_id].progress = progress
    self._tasks[task_id].progress_message = message
```

**Why track progress?**
- User feedback (loading bar)
- Debugging (where did it fail?)
- Monitoring (which stage is slow?)

**Alternative: Event streaming**
```python
# Instead of polling progress, stream events
async def stream_progress(task_id: str):
    async for event in task_events(task_id):
        yield f"data: {event}\n\n"
```

We use polling (simpler) but streaming would be better UX.

---

## 7. Testing Strategies

### Test Pyramid

```
        /\
       /  \      E2E Tests (few)
      /    \     - Full API calls
     /──────\    - Slow, flaky
    /        \
   /   Integ  \  Integration Tests (some)
  /    Tests   \ - Multiple components
 /──────────────\- Real dependencies
/                \
/   Unit Tests    \  Unit Tests (many)
/──────────────────\ - Single function
                     - Fast, isolated
```

### What We Have

```python
# Unit test - tests one function in isolation
def test_extract_dates():
    text = "Service date: 01/15/2024"
    dates = extract_dates(text)
    assert dates.service_date == "2024-01-15"

# Integration test - tests API endpoint
def test_create_case(client: TestClient):
    response = client.post("/api/v1/cases", json={"mode": "fast"})
    assert response.status_code == 201
```

### Testing LLM Code

**Challenge**: LLM calls are:
- Expensive ($$$)
- Slow (seconds)
- Non-deterministic

**Our solution**: Heuristic fallbacks that work without LLM
```python
def extract_case_facts(text, llm_api_key=None):
    if llm_api_key:
        return _extract_with_llm(text)
    else:
        return _extract_with_heuristics(text)  # Testable!
```

**Alternatives:**

| Approach | Pros | Cons |
|----------|------|------|
| **Heuristic fallback** (current) | Tests without LLM, fast | Different code paths |
| **Mock LLM responses** | Test actual LLM code path | Mocks can drift from reality |
| **Record/replay** | Real responses, deterministic | Responses become stale |
| **LLM in tests** | Tests real behavior | Slow, expensive, flaky |

### Fixture Design

```python
@pytest.fixture
def client():
    """Create test client with test database"""
    with TestClient(app) as client:
        yield client

@pytest.fixture
def sample_denial_letter():
    """Provide sample document for testing"""
    return "Dear Patient, Your claim was denied..."
```

**Why fixtures?**
- Reusable setup across tests
- Automatic cleanup (via `yield`)
- Clear dependencies

---

## 8. Production Readiness

### Health Checks

```python
@router.get("/health/live")      # Am I running?
@router.get("/health/ready")     # Can I serve traffic?
@router.get("/health/detailed")  # What's my full status?
```

**Why separate liveness vs readiness?**

**Kubernetes uses these differently:**
- **Liveness**: If fails, **restart the pod**
- **Readiness**: If fails, **stop sending traffic**

```yaml
# Kubernetes config
livenessProbe:
  httpGet:
    path: /health/live
  failureThreshold: 3      # Restart after 3 failures

readinessProbe:
  httpGet:
    path: /health/ready
  failureThreshold: 1      # Remove from load balancer immediately
```

**Real-world scenario:**
1. App starts, liveness passes (process running)
2. Readiness fails (database not connected yet)
3. No traffic sent until readiness passes
4. Later: database goes down
5. Readiness fails → traffic stops
6. Liveness passes → no restart (app is fine, DB isn't)

### Configuration Management

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )

    api_port: int = 8000
    openai_api_key: str = ""  # From environment
```

**Why Pydantic Settings?**
- Type validation (port must be int)
- Environment variable binding
- Default values
- `.env` file support

**12-Factor App Principle**: Store config in environment variables
- Same code, different config per environment
- No secrets in code

### Docker Multi-Stage Build

```dockerfile
# Stage 1: Build
FROM python:3.11-slim as builder
RUN pip install .

# Stage 2: Production
FROM python:3.11-slim as production
COPY --from=builder /opt/venv /opt/venv
```

**Why multi-stage?**
- Build tools not in final image (smaller)
- Build-time secrets not in final image (secure)
- Faster deployments (smaller image to transfer)

**Image size comparison:**
| Approach | Size |
|----------|------|
| Full Python + all build tools | ~1.2 GB |
| Slim Python + all build tools | ~500 MB |
| Multi-stage (current) | ~200 MB |
| Distroless | ~100 MB |

---

## 9. What We Could Do Differently

### If Starting Over

**1. Use a database from day 1**
```python
# Instead of file storage
class Case(SQLModel, table=True):
    id: str = Field(primary_key=True)
    mode: str
    created_at: datetime
```
**Why**: Querying, relationships, transactions

**2. Add OpenTelemetry tracing**
```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("extract_facts")
def extract_case_facts(text):
    ...
```
**Why**: Debugging production issues, performance analysis

**3. Use dependency injection framework**
```python
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    llm_client = providers.Singleton(LLMClient, api_key=config.api_key)
    case_service = providers.Factory(CaseService, llm=llm_client)
```
**Why**: Cleaner testing, explicit dependencies

### What We Did Well

1. **Type hints everywhere** - Catches bugs, enables tooling
2. **Pydantic models** - Validation, serialization, documentation
3. **Abstraction at right level** - CacheBackend, BaseLLMClient
4. **Tests for business logic** - Pipeline is well-tested
5. **Incremental complexity** - Started simple, added features

### Production Improvements Needed

| Feature | Why | Effort |
|---------|-----|--------|
| **Proper logging** (structured JSON) | Debugging, monitoring | Low |
| **Metrics** (Prometheus) | Performance, alerting | Medium |
| **Rate limiting** | Prevent abuse | Low |
| **Authentication** | Security | Medium |
| **Database** | Querying, durability | High |
| **Distributed tracing** | Debug across services | Medium |
| **CI/CD pipeline** | Automated testing/deploy | Medium |

---

## Questions to Ask Yourself

When making architectural decisions, ask:

1. **What's the simplest thing that could work?**
   - Start simple, add complexity when needed

2. **What are the failure modes?**
   - Network timeout? Retry with backoff
   - Service down? Circuit breaker
   - Data corruption? Validation

3. **How will this scale?**
   - 10x users? Probably fine
   - 100x users? Need to think about it
   - 1000x users? Probably need to redesign

4. **How will this be debugged in production?**
   - Logs? Metrics? Traces?
   - Can I reproduce the issue locally?

5. **What happens when requirements change?**
   - New LLM provider? Easy (strategy pattern)
   - New storage backend? Medium (add abstraction)
   - Complete rewrite of pipeline? Hard (core logic)

---

## Recommended Learning Path

1. **Read the code** in this order:
   - `config.py` - How configuration works
   - `main.py` - How the app starts
   - `api/routes.py` - How requests are handled
   - `pipeline/` - How business logic works
   - `llm/client.py` - How LLM integration works

2. **Modify and break things**:
   - Add a new API endpoint
   - Add a new LLM provider
   - Add a new cache backend
   - Break something and fix it

3. **Study the patterns**:
   - Strategy pattern (LLM clients)
   - Decorator pattern (cached client)
   - Repository pattern (storage)
   - Factory pattern (create_llm_client)

4. **Read these books**:
   - "Designing Data-Intensive Applications" by Martin Kleppmann
   - "Clean Architecture" by Robert Martin
   - "Building Microservices" by Sam Newman
