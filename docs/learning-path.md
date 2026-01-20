# DenialOps Learning Path

A structured guide to learn AI/ML engineering from this codebase.

## How to Use This Guide

1. Read files in the order listed
2. For each file, understand the "Why" before the "How"
3. Try modifying the code to test your understanding
4. Complete the exercises at the end of each section

---

## Phase 1: Foundations (Start Here)

### 1.1 Project Structure
**Read First:** Look at the directory layout

```bash
tree -L 2 src/denialops
```

```
src/denialops/
├── __init__.py          # Package initialization
├── main.py              # Application entry point
├── config.py            # Configuration management
├── api/                 # HTTP layer
├── models/              # Data structures
├── pipeline/            # Business logic
├── llm/                 # AI/LLM integration
└── utils/               # Utilities
```

**Key Concept:** Separation of concerns - each folder has ONE job.

**Why This Matters:** Senior engineers design systems that are easy to modify. When you need to change the LLM provider, you only touch `llm/`. When API changes, only `api/` changes.

---

### 1.2 Configuration Management
**File:** `src/denialops/config.py`

```bash
cat src/denialops/config.py
```

**What to Learn:**
- Pydantic Settings for type-safe configuration
- Environment variable loading
- Provider abstraction (LLMProvider enum)
- Computed properties (`llm_api_key`)

**Key Pattern:**
```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    llm_provider: LLMProvider = LLMProvider.OPENAI
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    @property
    def llm_api_key(self) -> str:
        """Return the right key based on provider."""
        if self.llm_provider == LLMProvider.OPENAI:
            return self.openai_api_key
        return self.anthropic_api_key
```

**Exercise:** Add a new setting `LLM_TEMPERATURE` with a default of 0.0.

---

### 1.3 Data Models
**Files:** `src/denialops/models/` (all files)

**Read Order:**
1. `case.py` - Core domain models
2. `route.py` - Routing decision models
3. `action_plan.py` - Output models
4. `documents.py` - Document handling

**What to Learn:**
- Pydantic BaseModel for validation
- Enums for type safety
- Optional fields with defaults
- Nested models
- Model serialization (`.model_dump()`)

**Key Pattern:**
```python
class DenialCode(BaseModel):
    """Strongly typed code representation."""
    code: str
    code_type: CodeType          # Enum, not string!
    description: str | None = None  # Optional with default

class CaseFacts(BaseModel):
    """Nested models for complex data."""
    denial_codes: list[DenialCode] = []
    service: ServiceInfo | None = None
```

**Why This Matters:** In production AI systems, data quality issues cause most bugs. Pydantic catches errors at the boundary (API input, LLM output) before they corrupt your system.

**Exercise:** Add a new field `urgency: str` to `CaseFacts` with possible values "low", "medium", "high".

---

## Phase 2: LLM Integration (Core AI Skills)

### 2.1 LLM Client Abstraction
**File:** `src/denialops/llm/client.py`

This is the most important file for AI engineering. Read it carefully.

**What to Learn:**
- Abstract Base Class pattern for provider abstraction
- Factory pattern for client creation
- Lazy initialization (property with `_client`)
- Error handling across providers

**Architecture:**
```
                    ┌─────────────┐
                    │  LLMClient  │  <- Unified interface
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
    │ OpenAIClient│ │AnthropicClient│ │ Future...  │
    └─────────────┘ └─────────────┘ └─────────────┘
```

**Key Pattern - Provider Abstraction:**
```python
class BaseLLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, system: str | None, ...) -> str:
        pass

class OpenAIClient(BaseLLMClient):
    def complete(self, prompt, system, ...):
        # OpenAI-specific implementation

class AnthropicClient(BaseLLMClient):
    def complete(self, prompt, system, ...):
        # Anthropic-specific implementation
```

**Why This Matters:**
- Switch providers without changing business logic
- A/B test different models easily
- Fall back to backup providers on failure
- This pattern is used at every major AI company

**Exercise:** Add a `MockClient` that returns predefined responses for testing.

---

### 2.2 Prompt Engineering
**File:** `src/denialops/pipeline/extract_facts.py` (top section)

**Read:** Lines 1-93 (the prompts)

**What to Learn:**
- System prompt design (role, constraints)
- User prompt structure (input tagging, output schema)
- Handling edge cases in prompts
- JSON output formatting

**Prompt Anatomy:**
```python
EXTRACTION_SYSTEM_PROMPT = """
You are an expert at...     # 1. Role setting
Extract structured...        # 2. Task definition
Always respond with JSON...  # 3. Output constraint
If information is not...     # 4. Edge case handling
"""

EXTRACTION_USER_PROMPT = """
<denial_letter>              # 5. Input boundaries
{text}
</denial_letter>

Extract... JSON object:      # 6. Output schema
{{
  "field": "description"
}}

Important:                   # 7. Clarifications/Examples
- Only extract CPT codes...
- Distinguish between...
"""
```

**Key Techniques:**
| Technique | Example | Purpose |
|-----------|---------|---------|
| XML tags | `<denial_letter>` | Clear input boundaries |
| JSON schema | `{{"field": "type"}}` | Structured output |
| Negative examples | "not zip codes" | Prevent common errors |
| Escape braces | `{{` `}}` | Literal braces in f-strings |

**Exercise:** Write a prompt that extracts contact information (name, email, phone) from an email signature.

---

### 2.3 LLM Response Handling
**File:** `src/denialops/pipeline/extract_facts.py`

**Read:** Functions `_extract_with_llm`, `_parse_llm_response`, `_build_case_facts_from_llm`

**What to Learn:**
- Calling LLM with proper parameters
- Parsing JSON from LLM output (handling markdown wrappers)
- Mapping unstructured LLM output to typed models
- Error handling and logging

**Key Pattern - Robust JSON Parsing:**
```python
def _parse_llm_response(response: str) -> dict:
    # LLMs sometimes wrap JSON in markdown
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        json_str = response[start:end].strip()
    elif "```" in response:
        # Generic code block
        start = response.find("```") + 3
        end = response.find("```", start)
        json_str = response[start:end].strip()
    else:
        json_str = response.strip()

    return json.loads(json_str)
```

**Why This Matters:** LLMs are unpredictable. Production code must handle variations in output format.

**Exercise:** Add handling for when the LLM returns `null` vs `None` vs missing fields.

---

### 2.4 Hybrid Extraction (Regex + LLM)
**File:** `src/denialops/pipeline/extract_facts.py`

**Read:** Functions `_extract_dates`, `_extract_codes`, `_extract_amounts`, `_extract_contact_info`

**What to Learn:**
- When to use regex vs LLM
- Pattern matching for structured data
- Combining deterministic and probabilistic approaches

**The 80/20 Rule:**
```
┌─────────────────────────────────────────────────────┐
│                    Extraction Task                   │
├──────────────────────┬──────────────────────────────┤
│   Regex (Fast/Free)  │     LLM (Slow/Expensive)     │
├──────────────────────┼──────────────────────────────┤
│ • Phone numbers      │ • Denial reason summary      │
│ • Dates              │ • Service description        │
│ • Dollar amounts     │ • Understanding context      │
│ • Claim numbers      │ • Ambiguous fields           │
│ • CPT codes (5 digit)│ • Free-form text             │
└──────────────────────┴──────────────────────────────┘
```

**Key Pattern:**
```python
def extract_case_facts(...):
    # Try LLM first (better quality)
    if llm_api_key:
        try:
            return _extract_with_llm(...)
        except Exception as e:
            logger.warning(f"LLM failed: {e}")

    # Fallback to regex (always works)
    return _extract_with_heuristics(...)
```

**Why This Matters:**
- LLM calls cost money ($0.01-0.10 per call)
- LLM calls are slow (500ms-5s)
- Regex is instant and free
- Use LLM for what it's good at, regex for the rest

**Exercise:** Add regex extraction for email addresses in denial letters.

---

## Phase 3: Pipeline Architecture

### 3.1 Pipeline Design
**File:** `src/denialops/api/routes.py` - Function `run_pipeline`

**What to Learn:**
- Staged pipeline pattern
- Artifact storage at each stage
- Error handling in pipelines
- Dependency injection (StorageDep, SettingsDep)

**Pipeline Flow:**
```python
async def run_pipeline(case_id, storage, settings):
    # Stage 1: Extract text (deterministic)
    extracted = extract_text(doc_path)
    storage.store_artifact("extracted_text.txt", extracted)

    # Stage 2: Extract facts (LLM)
    facts = extract_case_facts(text, llm_api_key)
    storage.store_artifact("case_facts.json", facts)

    # Stage 3: Route (rules)
    route = route_case(facts)
    storage.store_artifact("route.json", route)

    # Stage 4: Generate plan (template)
    plan = generate_action_plan(facts, route)
    storage.store_artifact("action_plan.json", plan)

    # Stage 5: Generate docs (template)
    docs = generate_document_pack(facts, plan)
    for name, content in docs.items():
        storage.store_artifact(name, content)
```

**Why Store Artifacts at Each Stage?**
1. **Debugging:** See exactly where things went wrong
2. **Resumability:** Restart from any stage
3. **Auditing:** Required for compliance (healthcare)
4. **Caching:** Skip stages if inputs unchanged

**Exercise:** Add a new stage that sends an email notification after the pipeline completes.

---

### 3.2 Rule-Based Routing
**File:** `src/denialops/pipeline/router.py`

**What to Learn:**
- Signal detection from unstructured data
- Weighted scoring for classification
- Confidence calculation
- Explainable AI (reasoning)

**Pattern - Signal Detection:**
```python
def _detect_signals(facts: CaseFacts) -> dict[str, bool]:
    text = facts.denial_reason.lower()

    return {
        "prior_auth_missing": any(phrase in text for phrase in [
            "prior authorization", "pre-certification"
        ]),
        "medical_necessity": any(phrase in text for phrase in [
            "not medically necessary", "experimental"
        ]),
        "has_carc_197": any(c.code == "CO-197" for c in facts.denial_codes),
    }

def _calculate_scores(signals: dict) -> dict[RouteType, float]:
    scores = {route: 0.0 for route in RouteType}

    if signals["prior_auth_missing"]:
        scores[RouteType.PRIOR_AUTH_NEEDED] += 0.4
    if signals["has_carc_197"]:
        scores[RouteType.PRIOR_AUTH_NEEDED] += 0.3
    # ... more rules

    return scores
```

**When Rules vs ML:**
| Use Rules When | Use ML When |
|----------------|-------------|
| Few categories (<10) | Many categories |
| Explainability required | Patterns too complex |
| No training data | Have labeled examples |
| Domain experts available | Need to discover patterns |

**Exercise:** Add a new signal for "timely filing" denials and route them appropriately.

---

### 3.3 Template-Based Generation
**File:** `src/denialops/pipeline/generate_docs.py`

**What to Learn:**
- Template design for documents
- Personalization with extracted data
- Markdown generation
- Route-specific content

**Pattern:**
```python
def _generate_call_script(facts: CaseFacts, plan: ActionPlan) -> str:
    # Safe defaults for missing data
    payer_name = facts.payer.name if facts.payer else "[Insurance Company]"
    member_id = facts.payer.member_id if facts.payer else "[Your Member ID]"

    # Template with placeholders
    script = f"""# Call Script: {payer_name}

## Information to Have Ready
- Member ID: {member_id}
- Claim Number: {claim_number}
"""

    # Route-specific sections
    if plan.route == RouteType.PRIOR_AUTH_NEEDED:
        script += """
"What do I need to do to get prior authorization?"
"""

    return script
```

**Exercise:** Create a template for a fax cover sheet.

---

## Phase 4: API & Testing

### 4.1 FastAPI Endpoints
**File:** `src/denialops/api/routes.py`

**What to Learn:**
- REST API design
- Request/Response models
- Dependency injection
- File uploads
- Error handling

**Key Patterns:**
```python
# Dependency injection
@router.post("/cases/{case_id}/run")
async def run_pipeline(
    case_id: CaseIdDep,      # Validated path parameter
    storage: StorageDep,      # Injected storage service
    settings: SettingsDep,    # Injected configuration
) -> RunPipelineResponse:
    ...

# File upload handling
@router.post("/cases/{case_id}/documents")
async def upload_document(
    file: Annotated[UploadFile, File(description="Document")],
    doc_type: Annotated[DocumentType, Form(description="Type")],
) -> UploadDocumentResponse:
    content = await file.read()
    ...
```

**Exercise:** Add a DELETE endpoint to remove a case.

---

### 4.2 Testing Strategy
**Files:** `tests/` directory

**What to Learn:**
- Unit tests for components
- Integration tests for API
- Fixtures and factories
- Mocking external services

**Test Organization:**
```
tests/
├── conftest.py              # Shared fixtures
├── test_api.py              # API integration tests
└── test_pipeline/
    ├── test_extract_facts.py  # Unit tests
    └── test_router.py         # Unit tests
```

**Key Patterns:**
```python
# Fixture for test client
@pytest.fixture
def client():
    return TestClient(app)

# Unit test
def test_extract_dates():
    text = "Appeal within 180 days"
    dates = _extract_dates(text)
    assert dates.appeal_deadline_days == 180

# Integration test
def test_create_case(client):
    response = client.post("/cases", json={"mode": "fast"})
    assert response.status_code == 201
```

**Exercise:** Write a test for the `generate_call_script` function.

---

## Phase 5: Production LLM Patterns

### 5.1 Retry Logic with Exponential Backoff
**File:** `src/denialops/llm/client.py`
**Documentation:** `docs/phase-2-learnings.md`

**What to Learn:**
- Retry logic with exponential backoff
- Handling rate limits and transient failures
- Token usage tracking and cost estimation
- Latency monitoring

**Key Pattern - Exponential Backoff:**
```python
class RetryConfig:
    def get_delay(self, attempt: int) -> float:
        """Calculate delay: 1s, 2s, 4s, 8s... up to max_delay."""
        delay = self.initial_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)

@retry_with_backoff(RetryConfig(max_retries=3))
def make_api_call():
    return client.chat.completions.create(...)
```

**Why This Matters:** LLM APIs fail frequently (rate limits, timeouts). Production systems must handle failures gracefully.

**Exercise:** Add jitter (random variation) to the backoff delay to prevent thundering herd.

---

### 5.2 Token Usage Tracking
**File:** `src/denialops/llm/client.py`

**What to Learn:**
- Tracking input/output tokens per call
- Cost estimation by model
- Aggregate usage across sessions

**Key Pattern:**
```python
@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def estimated_cost(self) -> float:
        pricing = {"gpt-4o": {"input": 5.0, "output": 15.0}}
        input_cost = (self.prompt_tokens / 1_000_000) * pricing[model]["input"]
        output_cost = (self.completion_tokens / 1_000_000) * pricing[model]["output"]
        return input_cost + output_cost
```

**Exercise:** Add a budget alert that warns when costs exceed a threshold.

---

### 5.3 Prompt Template Organization
**File:** `src/denialops/llm/prompts.py`

**What to Learn:**
- Centralized prompt management
- Version-controlled prompts
- Prompt validation

**Key Pattern:**
```python
class PromptLibrary:
    EXTRACT_FACTS_SYSTEM = EXTRACT_FACTS_SYSTEM
    EXTRACT_FACTS_USER = EXTRACT_FACTS_USER

    @classmethod
    def validate_prompt(cls, prompt: str, required_vars: list[str]) -> bool:
        return all(f"{{{var}}}" in prompt for var in required_vars)
```

**Exercise:** Add a prompt versioning system that tracks prompt changes over time.

---

## Phase 6: Verified Mode with Source Citations

### 6.1 SBC Document Extraction
**Files:** `src/denialops/models/plan_rules.py`, `src/denialops/pipeline/extract_plan_rules.py`
**Documentation:** `docs/phase-3-learnings.md`

**What to Learn:**
- Extracting structured data from policy documents
- Heuristic fallback when LLM unavailable
- Tracking extraction quality/confidence

**Key Pattern - Two-Tier Extraction:**
```python
def extract_plan_rules(case_id, text, llm_api_key=""):
    if llm_api_key:
        try:
            return _extract_with_llm(...)
        except Exception:
            logger.warning("LLM failed, using heuristics")

    return _extract_with_heuristics(...)  # Fallback
```

**Exercise:** Add extraction for a new section (e.g., "Network Requirements").

---

### 6.2 Source Citations for Grounding
**File:** `src/denialops/models/plan_rules.py`

**What to Learn:**
- Citing sources for LLM-extracted information
- Preventing hallucination through grounding
- Building trust in AI recommendations

**Key Pattern:**
```python
class SourceCitation(BaseModel):
    source_page: int | None
    source_section: str | None
    source_quote: str | None

    def format_citation(self) -> str:
        parts = []
        if self.source_page:
            parts.append(f"p.{self.source_page}")
        return f"[SBC {', '.join(parts)}]" if parts else ""

# Usage in recommendations:
recommendation = f"You have {deadline_days} days to appeal {citation.format_citation()}"
# Output: "You have 180 days to appeal [SBC p.8, Appeals]"
```

**Why This Matters:** In healthcare/legal domains, recommendations must be traceable to source documents. Citations build trust and enable verification.

**Exercise:** Add validation that checks if citations reference actual page numbers in the document.

---

### 6.3 Progressive Enhancement
**File:** `src/denialops/pipeline/generate_plan.py`

**What to Learn:**
- Providing value at different input levels
- Handling missing optional data gracefully
- Mode-specific recommendations

**Pattern:**
```
User uploads denial letter only
    └─► Fast Mode: Generic guidance

User uploads denial letter + SBC
    └─► Verified Mode: Cited, plan-specific guidance
```

**Key Pattern:**
```python
def _identify_assumptions(facts, mode, plan_rules=None):
    if mode == "fast" and not plan_rules:
        assumptions.append(Assumption(
            assumption="Plan-specific details not verified",
            how_to_verify="Upload your SBC for verified guidance",
        ))
    elif plan_rules and plan_rules.extraction_quality.confidence < 0.7:
        assumptions.append(Assumption(
            assumption="Plan rules extracted with moderate confidence",
        ))
```

**Exercise:** Add a "Premium Mode" that combines SBC with external policy databases.

---

## Phase 7: Advanced LLM Features

### 7.1 Personalized Summaries
**Files:** `src/denialops/pipeline/llm_generation.py`
**Documentation:** `docs/phase-4-learnings.md`

**What to Learn:**
- Generating user-friendly summaries from structured data
- Graceful degradation (LLM → heuristics)
- Combining multiple data sources for context

**Key Pattern:**
```python
def generate_personalized_summary(facts, route, plan_rules=None, llm_api_key=""):
    if llm_api_key:
        try:
            return _generate_with_llm(...)
        except Exception:
            pass
    return _generate_with_heuristics(...)

def _generate_with_heuristics(facts, route, plan_rules):
    # Always works, even without LLM
    return PersonalizedSummary(
        situation_summary=f"Your claim was denied: {facts.denial_reason}",
        recommendation=_get_route_recommendation(route),
        is_llm_generated=False,
    )
```

**Exercise:** Add urgency levels based on days until appeal deadline.

---

### 7.2 Grounding Validation
**File:** `src/denialops/pipeline/llm_generation.py`

**What to Learn:**
- Detecting hallucinations in generated content
- Validating claims against source facts
- Building trust in AI-generated output

**Key Pattern:**
```python
def validate_grounding(content, facts, plan_rules=None):
    # Extract claims from generated content
    found_codes = extract_codes_from_content(content)
    found_dates = extract_dates_from_content(content)

    # Build set of known valid values
    known_codes = set(c.code for c in facts.denial_codes)
    known_dates = collect_dates_from_facts(facts)

    # Detect hallucinations
    hallucinated_codes = [c for c in found_codes if c not in known_codes]

    return GroundingResult(
        is_grounded=len(hallucinated_codes) == 0,
        hallucinated_codes=hallucinated_codes,
    )
```

**Why This Matters:** LLMs can generate plausible-sounding but incorrect information. Grounding validation catches made-up codes, dates, and amounts before they reach users.

**Exercise:** Add validation for monetary amounts.

---

### 7.3 Success Prediction
**File:** `src/denialops/pipeline/llm_generation.py`

**What to Learn:**
- Multi-factor scoring for predictions
- Explainable AI (factors for/against)
- Combining heuristics with LLM reasoning

**Key Pattern:**
```python
def _predict_with_heuristics(facts, route, plan_rules):
    score = 0.5  # Start at 50%
    factors_for, factors_against = [], []

    # Route-specific factors
    if route.route == RouteType.CLAIM_CORRECTION_RESUBMIT:
        factors_for.append("Coding errors are often correctable")
        score += 0.2

    # Deadline factor
    if facts.dates.appeal_deadline:
        days_left = (deadline - date.today()).days
        if days_left > 30:
            factors_for.append("Sufficient time remaining")
            score += 0.1

    return SuccessPrediction(
        likelihood="high" if score >= 0.7 else "medium",
        score=score,
        factors_for=factors_for,
        factors_against=factors_against,
    )
```

**Exercise:** Add a factor for "similar past appeals" from a database.

---

### 7.4 Multi-Document Support (EOB)
**Files:** `src/denialops/models/eob.py`, `src/denialops/pipeline/extract_eob.py`

**What to Learn:**
- Extracting financial data from EOB documents
- Enriching case facts from multiple sources
- Pattern matching for healthcare documents

**Key Pattern - Information Fusion:**
```python
# In pipeline:

# Extract from denial letter
facts = extract_case_facts(denial_text)

# Extract from SBC (if uploaded)
plan_rules = extract_plan_rules(sbc_text) if sbc_doc else None

# Extract from EOB (if uploaded)
eob_facts = extract_eob_facts(eob_text) if eob_doc else None

# Enrich facts with EOB information
if eob_facts and eob_facts.denial_codes and not facts.denial_codes:
    facts.denial_codes = eob_facts.denial_codes
```

**Exercise:** Add extraction for a new document type (e.g., medical records).

---

## Phase 8: Production Readiness

### 8.1 Read the Technical Architecture
**File:** `docs/technical-architecture.md`

Read the entire document, focusing on:
- Production Considerations section
- Cost Optimization
- Scaling Considerations

### 8.2 What's Missing for Production

| Component | Current | Production |
|-----------|---------|------------|
| Database | File storage | PostgreSQL |
| Caching | None | Redis |
| Auth | None | OAuth/API keys |
| Monitoring | None | Prometheus/Grafana |
| Logging | Basic | Structured (JSON) |
| Rate Limiting | None | Per-user limits |

**Exercise:** Implement a simple in-memory cache for LLM responses.

---

## Learning Exercises (Do These!)

### Beginner
1. [ ] Add a new field to `CaseFacts` and update the extraction prompt
2. [ ] Add a new sample denial letter and test the pipeline
3. [ ] Write unit tests for `_extract_amounts`

### Intermediate
4. [ ] Add a new route type (e.g., "timely_filing_issue")
5. [ ] Add a new extraction field to `PlanRules` (e.g., "copay_rules")
6. [ ] Add request logging middleware

### Advanced
7. [ ] Add jitter to retry backoff to prevent thundering herd
8. [ ] Implement grounding validation (check generated content against source facts)
9. [ ] Implement A/B testing between OpenAI and Anthropic
10. [ ] Add OpenTelemetry tracing for LLM calls

---

## Senior AI Engineer Skills Checklist

After studying this codebase, you should understand:

### LLM Integration
- [ ] Multi-provider abstraction
- [ ] Prompt engineering techniques
- [ ] Structured output extraction
- [ ] Error handling and fallbacks
- [ ] Cost optimization strategies
- [ ] Retry logic with exponential backoff
- [ ] Token tracking and cost estimation

### System Design
- [ ] Pipeline architecture
- [ ] Artifact storage patterns
- [ ] Configuration management
- [ ] Dependency injection
- [ ] Progressive enhancement (Fast → Verified modes)

### Grounding & Trust
- [ ] Source citations for LLM outputs
- [ ] Heuristic fallbacks when LLM unavailable
- [ ] Extraction quality tracking
- [ ] Preventing hallucination

### Production Readiness
- [ ] Testing strategies for AI systems
- [ ] Monitoring and observability
- [ ] Security (secrets management)
- [ ] Scaling considerations

### Software Engineering
- [ ] Type safety with Pydantic
- [ ] API design with FastAPI
- [ ] Clean code organization
- [ ] Documentation practices

---

## Next Steps

1. **Clone and run** the project locally
2. **Read files** in the order above
3. **Do the exercises** - learning requires practice
4. **Build something similar** for a different domain
5. **Contribute** improvements back to this repo

Good luck on your journey to becoming a Senior AI Engineer!
