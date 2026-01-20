# DenialOps

AI-powered insurance claim denial understanding and action copilot.

## Overview

DenialOps helps patients and healthcare providers understand insurance claim denials and take appropriate action. Given a denial letter, it:

1. **Extracts** structured information (denial reason, codes, dates, amounts)
2. **Routes** to the best action path (prior auth, claim correction, appeal)
3. **Generates** personalized action plans and document templates
4. **Predicts** success likelihood with grounded reasoning

## Quick Start

### Option 1: Local Development

```bash
# Clone and setup
git clone https://github.com/RKPandit/denialops.git
cd denialops

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run tests
pytest

# Start the server
python -m uvicorn denialops.main:app --reload
```

### Option 2: Docker

```bash
# Development with hot reload
docker-compose --profile dev up api-dev

# Production with Redis caching
docker-compose up -d

# Check logs
docker-compose logs -f api
```

## API Usage

### Base URL
- Local: `http://localhost:8000`
- API Version: `/api/v1`

### Create and Process a Case

```bash
# 1. Create a case
curl -X POST http://localhost:8000/api/v1/cases \
  -H "Content-Type: application/json" \
  -d '{"mode": "fast"}'

# Response: {"case_id": "abc-123", "mode": "fast", "created_at": "..."}

# 2. Upload denial letter
curl -X POST http://localhost:8000/api/v1/cases/abc-123/documents \
  -F "file=@denial_letter.pdf" \
  -F "doc_type=denial_letter"

# 3a. Run pipeline (synchronous)
curl -X POST http://localhost:8000/api/v1/cases/abc-123/run

# 3b. Run pipeline (asynchronous - recommended for production)
curl -X POST http://localhost:8000/api/v1/cases/abc-123/run/async
# Response: {"task_id": "pipeline-abc-123", "status": "pending"}

# Check task status
curl http://localhost:8000/api/v1/tasks/pipeline-abc-123

# 4. Get results
curl http://localhost:8000/api/v1/cases/abc-123/artifacts
curl http://localhost:8000/api/v1/cases/abc-123/artifacts/action_plan.json
curl http://localhost:8000/api/v1/cases/abc-123/artifacts/appeal_letter.md
```

### Streaming Responses

```bash
# Stream LLM completion (Server-Sent Events)
curl -X POST http://localhost:8000/api/v1/stream/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain this denial", "max_tokens": 500}'

# Stream case summary
curl -X POST http://localhost:8000/api/v1/cases/abc-123/stream/summary
```

### Health Checks

```bash
# Basic health (for load balancers)
curl http://localhost:8000/health

# Kubernetes liveness probe
curl http://localhost:8000/health/live

# Kubernetes readiness probe
curl http://localhost:8000/health/ready

# Detailed status (debugging)
curl http://localhost:8000/health/detailed
```

## API Endpoints

### Cases
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/cases` | Create a new case |
| POST | `/api/v1/cases/{id}/documents` | Upload document |
| POST | `/api/v1/cases/{id}/run` | Run pipeline (sync) |
| POST | `/api/v1/cases/{id}/run/async` | Run pipeline (async) |
| GET | `/api/v1/cases/{id}/artifacts` | List artifacts |
| GET | `/api/v1/cases/{id}/artifacts/{name}` | Get artifact |

### Tasks
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/tasks` | List all tasks |
| GET | `/api/v1/tasks/{id}` | Get task status |
| POST | `/api/v1/tasks/{id}/cancel` | Cancel task |

### Streaming
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/stream/completion` | Stream LLM response |
| POST | `/api/v1/cases/{id}/stream/summary` | Stream case summary |
| GET | `/api/v1/stream/health` | Test streaming |

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Basic health check |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe |
| GET | `/health/detailed` | Full system status |

## Operating Modes

- **Fast Mode**: Works with just a denial letter. Provides general guidance.
- **Verified Mode**: Include your SBC/EOC for grounded, cited recommendations.

## Supported Routes

| Route | When Used |
|-------|-----------|
| `prior_auth_needed` | Prior authorization missing or incomplete |
| `claim_correction_resubmit` | Coding, modifier, or billing error |
| `medical_necessity_appeal` | Medical necessity determination |

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-xxx          # or ANTHROPIC_API_KEY
LLM_PROVIDER=openai            # or anthropic
LLM_MODEL=gpt-4o               # or claude-3-5-sonnet-20241022

# Optional
ENVIRONMENT=dev                # dev or prod
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Caching
CACHE_BACKEND=memory           # memory, redis, or none
CACHE_TTL=3600                 # seconds
REDIS_URL=redis://localhost:6379

# Storage
ARTIFACTS_PATH=./artifacts
MAX_UPLOAD_SIZE=10485760       # 10MB
ARTIFACT_RETENTION_DAYS=7
```

### Docker Environment

Create `.env` file for docker-compose:
```bash
OPENAI_API_KEY=sk-xxx
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
```

## Project Structure

```
denialops/
├── src/denialops/
│   ├── api/                 # FastAPI routes
│   │   ├── routes.py        # Case endpoints
│   │   ├── health.py        # Health checks
│   │   ├── streaming.py     # SSE streaming
│   │   └── tasks.py         # Background tasks
│   ├── cache/               # Caching layer
│   │   ├── memory.py        # In-memory LRU cache
│   │   └── redis.py         # Redis cache
│   ├── llm/                 # LLM integration
│   │   ├── client.py        # OpenAI/Anthropic clients
│   │   ├── cached_client.py # Caching wrapper
│   │   └── streaming.py     # Streaming support
│   ├── models/              # Pydantic models
│   ├── pipeline/            # Processing stages
│   ├── tasks/               # Background processing
│   └── utils/               # Utilities
├── tests/                   # Test suite (102 tests)
├── docs/                    # Documentation
├── Dockerfile               # Production container
├── docker-compose.yml       # Local development
└── pyproject.toml           # Dependencies
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=denialops

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Type check
mypy src/
```

## Docker Commands

```bash
# Build image
docker build -t denialops .

# Run with Redis (production-like)
docker-compose up -d

# Run development mode (hot reload)
docker-compose --profile dev up api-dev

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: denialops
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: denialops:latest
        ports:
        - containerPort: 8000
        env:
        - name: CACHE_BACKEND
          value: redis
        - name: REDIS_URL
          value: redis://redis:6379
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Documentation

### Learning Resources
- [Engineering Deep Dive](docs/engineering-deep-dive.md) - Architecture decisions and tradeoffs
- [Code Patterns Explained](docs/code-patterns-explained.md) - Line-by-line pattern explanations
- [Learning Exercises](docs/learning-exercises.md) - Hands-on practice exercises
- [Interview Topics](docs/interview-topics.md) - System design interview prep

### Phase Documentation
- [Phase 5 Learnings](docs/phase-5-learnings.md) - Production readiness features

### Reference
- [Spec](docs/spec.md) - Product specification
- [Implementation Plan](docs/implementation-plan.md) - Build checklist
- [Threat Model](docs/threat-model.md) - Security considerations

## Architecture Highlights

- **LLM Caching**: Reduces costs by caching deterministic LLM responses
- **Streaming**: Real-time response delivery via Server-Sent Events
- **Background Tasks**: Async pipeline execution with progress tracking
- **Health Checks**: Kubernetes-ready liveness and readiness probes
- **Graceful Degradation**: Heuristic fallbacks when LLM unavailable

## Test Coverage

```
102 tests covering:
- API endpoints
- Pipeline stages
- LLM integration
- Caching
- Background tasks
- Streaming
```

## Disclaimers

- This tool is for informational purposes only
- Not legal or medical advice
- Verify all information with your insurance company

## License

MIT
