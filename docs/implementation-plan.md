# DenialOps — Implementation Plan

## Overview

This document is the build checklist for MVP v1. Follow sections in order.

---

## Phase 0: Project Setup

### 0.1 Repository Structure
```
denialops/
├── src/
│   └── denialops/
│       ├── __init__.py
│       ├── main.py              # FastAPI app entry
│       ├── api/
│       │   ├── __init__.py
│       │   ├── routes.py        # API endpoints
│       │   └── dependencies.py  # Request dependencies
│       ├── models/
│       │   ├── __init__.py
│       │   ├── case.py          # Pydantic models for case
│       │   ├── documents.py     # Document models
│       │   └── schemas.py       # Re-export JSON schema validators
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── extract_text.py
│       │   ├── extract_facts.py
│       │   ├── extract_plan_rules.py
│       │   ├── router.py
│       │   ├── generate_plan.py
│       │   └── generate_docs.py
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── client.py        # LLM client abstraction
│       │   └── prompts.py       # Prompt templates
│       └── utils/
│           ├── __init__.py
│           ├── pdf.py           # PDF extraction utilities
│           ├── validation.py    # Schema validation
│           └── storage.py       # Artifact storage
├── schemas/
│   ├── case_facts.schema.json
│   ├── route.schema.json
│   ├── action_plan.schema.json
│   └── plan_rules.schema.json
├── data/
│   └── samples/                 # Test denial letters
├── artifacts/                   # Generated case artifacts (gitignored)
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_pipeline/
│   │   ├── test_extract_text.py
│   │   ├── test_extract_facts.py
│   │   ├── test_router.py
│   │   └── test_generate.py
│   └── golden/                  # Golden test cases
├── eval/
│   ├── run_eval.py
│   ├── cases/                   # Eval case definitions
│   └── reports/                 # Eval output
├── docs/
│   ├── spec.md
│   ├── implementation-plan.md
│   ├── eval.md
│   └── threat-model.md
├── pyproject.toml
├── Makefile
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── .gitignore
```

### 0.2 Dependencies (pyproject.toml)
```toml
[project]
name = "denialops"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.5.0",
    "pdfminer.six>=20231228",
    "python-multipart>=0.0.6",
    "jsonschema>=4.21.0",
    "httpx>=0.26.0",
    "anthropic>=0.18.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.2.0",
    "mypy>=1.8.0",
    "pre-commit>=3.6.0",
]
```

### 0.3 Makefile Targets
```makefile
.PHONY: install lint typecheck test eval run docker-build docker-run

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests
	ruff check --fix src tests

typecheck:
	mypy src

test:
	pytest tests -v --cov=src/denialops

eval:
	python eval/run_eval.py

run:
	uvicorn denialops.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t denialops:latest .

docker-run:
	docker compose up
```

### 0.4 Setup Tasks
- [ ] Create `pyproject.toml` with dependencies
- [ ] Create `Makefile` with targets
- [ ] Create `.env.example` with required env vars
- [ ] Update `.gitignore` (add `artifacts/`, `.env`)
- [ ] Create directory structure
- [ ] Run `make install` to verify setup
- [ ] Create `ruff.toml` for linting config
- [ ] Create `mypy.ini` or `pyproject.toml` mypy section

---

## Phase 1: Vertical Slice (End-to-End Proof)

**Goal:** Upload PDF → extract text → extract facts → route → return JSON. No generation yet.

### 1.1 Storage Layer
File: `src/denialops/utils/storage.py`

- [ ] `create_case_directory(case_id: str) -> Path`
- [ ] `store_document(case_id: str, file: UploadFile, doc_type: str) -> Path`
- [ ] `store_artifact(case_id: str, filename: str, content: str | dict) -> Path`
- [ ] `list_artifacts(case_id: str) -> list[ArtifactInfo]`
- [ ] `get_artifact(case_id: str, filename: str) -> str | dict`

### 1.2 PDF Text Extraction
File: `src/denialops/pipeline/extract_text.py`

- [ ] `extract_text_from_pdf(pdf_path: Path) -> ExtractedText`
  - Use pdfminer.six
  - Return text with page boundaries preserved
  - Handle text files directly (passthrough)
- [ ] `ExtractedText` model with `pages: list[PageText]`

### 1.3 Case Facts Extraction
File: `src/denialops/pipeline/extract_facts.py`

- [ ] `extract_case_facts(text: ExtractedText, user_context: dict | None) -> CaseFacts`
- [ ] Deterministic extractors (run first):
  - [ ] `extract_dates(text)` - regex for dates
  - [ ] `extract_codes(text)` - regex for CPT, ICD-10, HCPCS
  - [ ] `extract_amounts(text)` - regex for dollar amounts
  - [ ] `extract_contact_info(text)` - phone, fax, address patterns
- [ ] LLM extraction for unstructured fields:
  - [ ] `extract_denial_reason(text)` - LLM call
  - [ ] `extract_payer_info(text)` - LLM call
- [ ] Merge deterministic + LLM results
- [ ] Validate against `case_facts.schema.json`

### 1.4 Case Router
File: `src/denialops/pipeline/router.py`

- [ ] `route_case(facts: CaseFacts, plan_rules: PlanRules | None) -> RouteDecision`
- [ ] Signal detection:
  - [ ] `detect_pa_signals(facts)` → weight
  - [ ] `detect_coding_signals(facts)` → weight
  - [ ] `detect_medical_necessity_signals(facts)` → weight
- [ ] Route selection logic (highest weight wins)
- [ ] Confidence calculation
- [ ] Validate against `route.schema.json`

### 1.5 API Endpoints (Minimal)
File: `src/denialops/api/routes.py`

- [ ] `POST /cases` - Create case, return case_id
- [ ] `POST /cases/{case_id}/documents` - Upload document
- [ ] `POST /cases/{case_id}/run` - Run pipeline (sync), return route + facts
- [ ] `GET /cases/{case_id}/artifacts` - List artifacts
- [ ] `GET /cases/{case_id}/artifacts/{filename}` - Get artifact content

### 1.6 Vertical Slice Acceptance
- [ ] Can create case via API
- [ ] Can upload PDF
- [ ] Pipeline produces `case_facts.json` and `route.json`
- [ ] Both files validate against schemas
- [ ] `make test` passes with basic tests

---

## Phase 2: Generation (Action Plan + Documents)

### 2.1 LLM Client
File: `src/denialops/llm/client.py`

- [ ] `LLMClient` class with provider abstraction
- [ ] Support Anthropic Claude (primary)
- [ ] Structured output parsing
- [ ] Rate limiting / retry logic
- [ ] Token usage tracking

### 2.2 Prompt Templates
File: `src/denialops/llm/prompts.py`

- [ ] `EXTRACT_DENIAL_REASON` - For fact extraction
- [ ] `GENERATE_ACTION_PLAN` - Main plan generation
- [ ] `GENERATE_APPEAL_LETTER` - For medical_necessity_appeal
- [ ] `GENERATE_PA_CHECKLIST` - For prior_auth_needed
- [ ] `GENERATE_RESUBMIT_CHECKLIST` - For claim_correction_resubmit
- [ ] `GENERATE_CALL_SCRIPT` - For all routes

### 2.3 Action Plan Generator
File: `src/denialops/pipeline/generate_plan.py`

- [ ] `generate_action_plan(facts: CaseFacts, route: RouteDecision, plan_rules: PlanRules | None) -> ActionPlan`
- [ ] Grounding enforcement:
  - [ ] All dates must come from `facts.dates` or be calculated from them
  - [ ] All codes must come from `facts.denial_codes` or `facts.service`
  - [ ] Unknown fields explicitly marked
- [ ] Citation injection (Verified mode)
- [ ] Validate against `action_plan.schema.json`
- [ ] Generate markdown version

### 2.4 Document Pack Generator
File: `src/denialops/pipeline/generate_docs.py`

- [ ] `generate_document_pack(facts: CaseFacts, plan: ActionPlan, route: str) -> list[GeneratedDoc]`
- [ ] Route-specific generation:
  - [ ] `medical_necessity_appeal` → appeal_letter.md
  - [ ] `prior_auth_needed` → pa_checklist.md
  - [ ] `claim_correction_resubmit` → resubmit_checklist.md
  - [ ] All routes → call_script.md
- [ ] Disclaimer injection in all documents
- [ ] Assumptions section when data missing

### 2.5 Phase 2 Acceptance
- [ ] Pipeline produces `action_plan.json` and `action_plan.md`
- [ ] Route-specific documents generated
- [ ] All documents contain disclaimers
- [ ] No hallucinated codes/dates (manual verification on test cases)

---

## Phase 3: Verified Mode (Policy Extraction)

### 3.1 Plan Rules Extraction
File: `src/denialops/pipeline/extract_plan_rules.py`

- [ ] `extract_plan_rules(sbc_text: ExtractedText) -> PlanRules`
- [ ] Section detection (SBC has standard sections)
- [ ] Extract:
  - [ ] Deductibles / OOP max
  - [ ] Prior auth requirements
  - [ ] Medical necessity definition
  - [ ] Exclusions
  - [ ] Appeal rights / deadlines
- [ ] Page/section anchoring for citations
- [ ] Validate against `plan_rules.schema.json`

### 3.2 Citation Integration
- [ ] Update `generate_action_plan` to include citations when `plan_rules` available
- [ ] Citation format: `[SBC p.X, Section Y]`
- [ ] Update `generate_document_pack` to include citations in appeal letters

### 3.3 Phase 3 Acceptance
- [ ] Verified mode API works (upload SBC + denial letter)
- [ ] `plan_rules.json` extracted and validated
- [ ] Action plan includes citations
- [ ] Appeal letter includes policy citations

---

## Phase 4: Quality & Observability

### 4.1 Evaluation Harness
File: `eval/run_eval.py`

- [ ] Load test cases from `eval/cases/`
- [ ] Run pipeline on each
- [ ] Score extraction (field coverage, accuracy vs golden)
- [ ] Score routing (accuracy vs expected route)
- [ ] Score generation (grounding check, citation coverage)
- [ ] Output report to `eval/reports/`

### 4.2 Golden Test Cases
- [ ] Create 5-10 synthetic denial letters in `data/samples/`
- [ ] Create corresponding golden `case_facts.json` for each
- [ ] Create expected route labels
- [ ] Document in `eval/cases/case_manifest.json`

### 4.3 Logging & Monitoring
- [ ] Structured logging (JSON format)
- [ ] PII redaction in logs
- [ ] Request tracing (correlation IDs)
- [ ] LLM call logging (tokens, latency, cost)

### 4.4 CI Pipeline
File: `.github/workflows/ci.yml`

- [ ] Lint (ruff)
- [ ] Typecheck (mypy)
- [ ] Unit tests (pytest)
- [ ] Integration tests
- [ ] Eval harness (on sample data)

---

## Phase 5: Containerization & Deployment

### 5.1 Docker
- [ ] Create `Dockerfile` (multi-stage, slim image)
- [ ] Create `docker-compose.yml` for local dev
- [ ] Health check endpoint (`GET /health`)
- [ ] Environment variable configuration

### 5.2 Configuration
- [ ] `.env.example` with all required vars
- [ ] `ANTHROPIC_API_KEY`
- [ ] `ARTIFACTS_PATH`
- [ ] `LOG_LEVEL`
- [ ] `ENVIRONMENT` (dev/staging/prod)

---

## Milestone Checklist

| Milestone | Deliverable | Acceptance |
|-----------|-------------|------------|
| M0 | Project setup | `make install && make lint` passes |
| M1 | Vertical slice | PDF → facts → route (no generation) |
| M2 | Generation | Action plan + document pack |
| M3 | Verified mode | Policy extraction + citations |
| M4 | Quality | Eval harness + CI green |
| M5 | Deployment | Docker + health checks |

---

## Tech Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Python version | 3.11+ | Type hints, performance |
| Web framework | FastAPI | Async, Pydantic integration, OpenAPI |
| PDF extraction | pdfminer.six | No binary deps, text PDF support |
| LLM provider | Anthropic Claude | Strong extraction, structured output |
| Schema validation | jsonschema | Standard, language-agnostic schemas |
| Testing | pytest | Standard, good async support |
| Linting | ruff | Fast, replaces flake8+black+isort |
| Type checking | mypy | Standard, strict mode |

---

## Open Questions (resolve before coding)

1. **LLM model choice**: Claude 3.5 Sonnet vs Claude 3 Opus for extraction accuracy vs cost?
2. **Async pipeline**: Run stages async or sync for MVP?
3. **Artifact storage**: Local filesystem vs MinIO/S3 for MVP?
4. **Rate limiting**: Client-side or rely on Anthropic's limits?
