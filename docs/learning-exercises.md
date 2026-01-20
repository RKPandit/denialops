# Hands-On Learning Exercises

Practical exercises to deepen your understanding of production software engineering.

---

## Level 1: Understanding the Codebase

### Exercise 1.1: Trace a Request
**Goal**: Understand the full request lifecycle

**Task**: Trace what happens when a client calls `POST /api/v1/cases`

1. Start at `main.py` - how does FastAPI find the route?
2. Find the route in `api/routes.py` - what function handles it?
3. What dependencies are injected? (Look for `Depends()`)
4. What does `CaseStorage` do?
5. What response is returned?

**Document your findings**:
```
Request Flow:
1. FastAPI receives POST /api/v1/cases
2. Routes to create_case() in routes.py
3. Injects StorageDep (CaseStorage instance)
4. Creates UUID for case_id
5. Calls storage.create_case()
6. Returns CreateCaseResponse
```

### Exercise 1.2: Understand Dependency Injection
**Goal**: Learn how FastAPI DI works

**Read** `api/dependencies.py` and answer:
1. What is `Annotated[CaseStorage, Depends(get_storage)]`?
2. Why do we use `Depends()` instead of creating objects directly?
3. How would you test a function that uses these dependencies?

**Try**: Create a mock storage for testing:
```python
# In tests/conftest.py
@pytest.fixture
def mock_storage():
    # Create an in-memory storage
    pass
```

### Exercise 1.3: Configuration Deep Dive
**Goal**: Understand 12-factor app configuration

**Experiment**:
1. Create a `.env` file with `API_PORT=9000`
2. Start the server - what port does it use?
3. Set environment variable `API_PORT=8080` - which wins?
4. Remove `.env` - what's the default?

**Learn**: Pydantic settings precedence:
1. Environment variables (highest)
2. `.env` file
3. Default values (lowest)

---

## Level 2: Modify and Extend

### Exercise 2.1: Add a New Endpoint
**Goal**: Practice adding API functionality

**Task**: Add `GET /api/v1/cases/{case_id}` to return case metadata

1. Add route in `api/routes.py`
2. Create response model
3. Add test in `tests/test_api.py`
4. Run tests to verify

**Checklist**:
- [ ] Route added with proper path
- [ ] Response model defined with Pydantic
- [ ] Error handling for case not found
- [ ] Test passes

### Exercise 2.2: Add Request Logging
**Goal**: Learn about middleware and logging

**Task**: Log every API request with:
- Timestamp
- Method and path
- Response status code
- Duration in milliseconds

**Approach 1: Middleware**
```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} {response.status_code} {duration:.2f}ms")
    return response
```

**Approach 2: Dependency**
```python
async def log_request(request: Request):
    logger.info(f"Incoming: {request.method} {request.url.path}")
```

**Compare**: Which approach is better and why?

### Exercise 2.3: Add a New LLM Provider
**Goal**: Practice the strategy pattern

**Task**: Add support for a hypothetical "LocalLLM" provider

1. Create `LocalLLMClient(BaseLLMClient)` in `llm/client.py`
2. Add `LOCAL = "local"` to `LLMProvider` enum
3. Update `LLMClient.__init__` to handle new provider
4. Write tests with mock responses

**Bonus**: Make it actually work with [Ollama](https://ollama.ai/)

### Exercise 2.4: Add SQLite Storage
**Goal**: Practice the repository pattern

**Task**: Create an alternative storage backend using SQLite

1. Create `SqliteStorage` class with same interface as `CaseStorage`
2. Add configuration option to choose storage backend
3. Ensure all existing tests pass with new storage
4. Compare performance

**Schema**:
```sql
CREATE TABLE cases (
    id TEXT PRIMARY KEY,
    mode TEXT,
    created_at TIMESTAMP,
    metadata JSON
);

CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    case_id TEXT REFERENCES cases(id),
    doc_type TEXT,
    content BLOB
);
```

---

## Level 3: Production Patterns

### Exercise 3.1: Add Circuit Breaker
**Goal**: Learn resilience patterns

**Problem**: If the LLM API is down, we keep hammering it, making things worse.

**Solution**: Circuit breaker pattern
```
CLOSED (normal) ──────▶ OPEN (failing)
     │                      │
     │ failures < threshold │ after timeout
     │                      ▼
     └──────────────── HALF-OPEN (testing)
```

**Task**: Implement a circuit breaker for LLM calls

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failures = 0
        self.state = "closed"
        self.last_failure_time = None

    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
            else:
                raise CircuitOpenError()

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        self.failures = 0
        self.state = "closed"

    def _on_failure(self):
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.state = "open"
            self.last_failure_time = time.time()
```

### Exercise 3.2: Add Rate Limiting
**Goal**: Protect your API from abuse

**Task**: Limit API calls to 100 requests per minute per IP

**Options**:
1. In-memory (simple, single server)
2. Redis-based (distributed)
3. Use `slowapi` library

**Implement with sliding window**:
```python
class RateLimiter:
    def __init__(self, max_requests=100, window_seconds=60):
        self.requests = defaultdict(list)  # ip -> [timestamps]

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds

        # Remove old requests
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if t > window_start
        ]

        if len(self.requests[client_ip]) >= self.max_requests:
            return False

        self.requests[client_ip].append(now)
        return True
```

### Exercise 3.3: Add Prometheus Metrics
**Goal**: Learn observability

**Task**: Add metrics for:
- Request count by endpoint and status
- Request duration histogram
- LLM call duration
- Cache hit/miss ratio

**Setup**:
```python
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)
```

**Questions to answer with metrics**:
1. What's the p99 latency?
2. Which endpoint is slowest?
3. What's the error rate?
4. Is caching effective?

### Exercise 3.4: Add Structured Logging
**Goal**: Production-grade logging

**Current** (unstructured):
```
2024-01-15 10:30:45 INFO Processing case abc-123
```

**Better** (structured JSON):
```json
{
  "timestamp": "2024-01-15T10:30:45Z",
  "level": "info",
  "message": "Processing case",
  "case_id": "abc-123",
  "user_id": "user-456",
  "duration_ms": 1234
}
```

**Task**: Configure structured logging with `structlog`
```python
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()
logger.info("processing_case", case_id="abc-123", duration_ms=1234)
```

---

## Level 4: Architecture Challenges

### Exercise 4.1: Split into Microservices
**Goal**: Understand service boundaries

**Task**: Design how to split DenialOps into services

Current monolith:
```
┌─────────────────────────────────────┐
│             DenialOps               │
│  ┌─────┐ ┌─────────┐ ┌───────────┐ │
│  │ API │ │ Pipeline│ │ LLM Client│ │
│  └─────┘ └─────────┘ └───────────┘ │
└─────────────────────────────────────┘
```

Design microservices version:
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ API Gateway │────▶│ Case Service │────▶│ LLM Service │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Storage    │
                    └──────────────┘
```

**Questions**:
1. How do services communicate? (REST, gRPC, message queue)
2. How is data shared? (Each service owns its data)
3. How do you handle transactions across services?
4. How do you trace requests across services?

### Exercise 4.2: Design for 10x Scale
**Goal**: Think about scalability

**Current scale**: 100 cases/day, single server

**Target scale**: 1000 cases/day, multiple servers

**Design document**:
1. What's the bottleneck? (LLM calls)
2. What needs to scale horizontally?
3. What state needs to be shared?
4. How to handle failures?

### Exercise 4.3: Add Event Sourcing
**Goal**: Advanced data pattern

**Current**: Store current state
```python
case.status = "completed"
storage.save(case)
```

**Event sourcing**: Store events that led to current state
```python
events = [
    CaseCreated(case_id, timestamp=t1),
    DocumentUploaded(case_id, doc_id, timestamp=t2),
    PipelineStarted(case_id, timestamp=t3),
    PipelineCompleted(case_id, result, timestamp=t4),
]
```

**Benefits**:
- Complete audit trail
- Can rebuild state at any point in time
- Debugging (what happened?)

**Task**: Design event-sourced version of case handling

---

## Level 5: Production Incident Simulation

### Exercise 5.1: Debug a Memory Leak
**Setup**: Add a bug that causes memory growth
```python
# Hidden somewhere in the code
_global_cache = []

def process_something(data):
    _global_cache.append(data)  # Never cleaned up!
```

**Task**: Find and fix it using:
- Memory profiling (`memory_profiler`)
- Heap dumps
- Monitoring graphs

### Exercise 5.2: Handle Cascading Failure
**Scenario**: LLM API returns 500 errors

**Task**: Ensure the system degrades gracefully:
1. Circuit breaker trips after N failures
2. Requests get queued or rejected
3. Health endpoint reports degraded
4. System recovers when LLM is back

### Exercise 5.3: Database Migration
**Scenario**: Need to add a new field to case metadata

**Task**: Perform zero-downtime migration:
1. Add field as nullable
2. Deploy code that writes new field
3. Backfill existing data
4. Deploy code that requires new field
5. Make field non-nullable

---

## Reflection Questions

After each exercise, ask yourself:

1. **What was harder than expected?**
2. **What would I do differently next time?**
3. **What patterns did I recognize?**
4. **What would break if requirements changed?**
5. **How would I explain this to a junior developer?**

---

## Resources for Further Learning

### Books
- "Designing Data-Intensive Applications" - Kleppmann
- "Clean Architecture" - Martin
- "Site Reliability Engineering" - Google
- "Building Microservices" - Newman

### Online
- [The Twelve-Factor App](https://12factor.net/)
- [Martin Fowler's Blog](https://martinfowler.com/)
- [AWS Architecture Blog](https://aws.amazon.com/blogs/architecture/)
- [High Scalability Blog](http://highscalability.com/)

### Practice
- [System Design Primer](https://github.com/donnemartin/system-design-primer)
- [Exercism](https://exercism.org/) for language practice
- Build your own: cache, database, message queue
