# DenialOps Technical Architecture

A deep-dive into how DenialOps works, designed for engineers looking to understand production AI/ML system design.

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Pipeline Design Pattern](#pipeline-design-pattern)
4. [LLM Integration](#llm-integration)
5. [Prompt Engineering](#prompt-engineering)
6. [Data Models & Schemas](#data-models--schemas)
7. [Routing Logic](#routing-logic)
8. [Error Handling & Fallbacks](#error-handling--fallbacks)
9. [Testing Strategy](#testing-strategy)
10. [Production Considerations](#production-considerations)
11. [Key Learnings for AI/ML Engineers](#key-learnings-for-aiml-engineers)

---

## System Overview

DenialOps is an **LLM-powered document understanding and action generation system**. It demonstrates several key patterns used in production AI systems:

- **Structured extraction** from unstructured documents
- **Multi-provider LLM abstraction** (OpenAI, Anthropic)
- **Hybrid extraction** (deterministic + LLM)
- **Rule-based routing** with confidence scoring
- **Template-based generation** with personalization

### Core Problem

Insurance denial letters are unstructured documents with critical information buried in legal/medical text. The system needs to:
1. Extract structured data reliably
2. Classify the denial type
3. Generate actionable guidance

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              API Layer                                   │
│                         (FastAPI + Pydantic)                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Pipeline Stages                                │
│                                                                          │
│  ┌──────────┐   ┌──────────────┐   ┌────────┐   ┌────────┐   ┌───────┐ │
│  │ Extract  │──▶│ Extract Facts│──▶│ Route  │──▶│ Action │──▶│ Docs  │ │
│  │   Text   │   │    (LLM)     │   │        │   │  Plan  │   │       │ │
│  └──────────┘   └──────────────┘   └────────┘   └────────┘   └───────┘ │
│       │                │                │            │           │      │
│       ▼                ▼                ▼            ▼           ▼      │
│   PDF/TXT         OpenAI/         Rule-based    Template     Markdown  │
│   Parser          Anthropic       Classifier    Engine       Generator │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Storage Layer                                   │
│                    (File-based Artifact Store)                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Design Pattern

The system uses a **staged pipeline pattern** - a common architecture for document processing systems.

### Why Staged Pipelines?

1. **Debuggability**: Each stage produces artifacts you can inspect
2. **Resumability**: Can restart from any stage
3. **Testability**: Each stage can be unit tested independently
4. **Flexibility**: Easy to swap implementations (e.g., different LLM providers)

### Pipeline Stages

```python
# src/denialops/api/routes.py - Simplified

async def run_pipeline(case_id: str):
    # Stage 1: Extract raw text
    extracted = extract_text(doc_path)           # Deterministic
    storage.store_artifact("extracted_text.txt", extracted)

    # Stage 2: Extract structured facts
    facts = extract_case_facts(text, llm_api_key)  # LLM-powered
    storage.store_artifact("case_facts.json", facts)

    # Stage 3: Route to action path
    route = route_case(facts)                      # Rule-based
    storage.store_artifact("route.json", route)

    # Stage 4: Generate action plan
    plan = generate_action_plan(facts, route)      # Template + Logic
    storage.store_artifact("action_plan.json", plan)

    # Stage 5: Generate documents
    docs = generate_document_pack(facts, plan)     # Template-based
    for name, content in docs.items():
        storage.store_artifact(name, content)
```

### Key Design Decision: Artifacts at Every Stage

Every stage writes its output to storage. This enables:
- **Observability**: See exactly what each stage produced
- **Debugging**: Find where things went wrong
- **Caching**: Skip stages if inputs haven't changed
- **Auditing**: Required for healthcare compliance

---

## LLM Integration

### Multi-Provider Abstraction

The system supports multiple LLM providers through an abstraction layer:

```python
# src/denialops/llm/client.py

class BaseLLMClient(ABC):
    """Abstract base class - defines the interface."""

    @abstractmethod
    def complete(self, prompt: str, system: str | None,
                 max_tokens: int, temperature: float) -> str:
        pass

class OpenAIClient(BaseLLMClient):
    """OpenAI implementation."""
    def complete(self, prompt, system, max_tokens, temperature):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

class AnthropicClient(BaseLLMClient):
    """Anthropic implementation."""
    def complete(self, prompt, system, max_tokens, temperature):
        kwargs = {"model": self.model, "max_tokens": max_tokens,
                  "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

class LLMClient:
    """Unified client - Factory pattern."""
    def __init__(self, provider: LLMProvider, api_key: str, model: str):
        if provider == LLMProvider.OPENAI:
            self._client = OpenAIClient(api_key, model)
        else:
            self._client = AnthropicClient(api_key, model)
```

### Why This Pattern Matters

1. **Vendor flexibility**: Switch providers without changing business logic
2. **A/B testing**: Compare model performance easily
3. **Fallback chains**: Try Provider A, fall back to Provider B
4. **Cost optimization**: Route simple queries to cheaper models

---

## Prompt Engineering

### Structured Extraction Prompt

The extraction prompt is the core AI component. Here's how it's designed:

```python
# src/denialops/pipeline/extract_facts.py

EXTRACTION_SYSTEM_PROMPT = """You are an expert at analyzing health insurance
denial letters. Extract structured information from the denial letter text
provided. Always respond with valid JSON. Be precise and accurate.
If information is not present in the letter, use null for that field."""

EXTRACTION_USER_PROMPT = """Analyze this insurance denial letter and extract
the following information as JSON:

<denial_letter>
{text}
</denial_letter>

Extract and return a JSON object with these fields:
{{
  "denial_reason": "The full denial reason as stated in the letter",
  "denial_reason_summary": "A short 5-10 word summary",
  "denial_codes": [
    {{"code": "string", "code_type": "CPT|HCPCS|ICD_10_CM|CARC|RARC"}}
  ],
  "service": {{
    "description": "Description of the service denied",
    "cpt_codes": ["list of CPT codes"],
    "date_of_service": "YYYY-MM-DD or null",
    "provider_name": "Name of provider"
  }},
  ...
}}

Important:
- Only extract CPT codes that are explicitly procedure codes (5 digits like
  72148), not zip codes or other numbers
- Distinguish between the denial date (letter date) and date of service
- For amounts, extract numeric values only (no $ signs)

Return ONLY valid JSON, no other text."""
```

### Prompt Engineering Best Practices Used

| Technique | Example | Why It Works |
|-----------|---------|--------------|
| **Role setting** | "You are an expert at analyzing health insurance denial letters" | Activates relevant knowledge |
| **XML tags** | `<denial_letter>` | Clear input boundaries |
| **JSON schema** | Explicit field definitions | Structured, parseable output |
| **Negative examples** | "not zip codes or other numbers" | Prevents common errors |
| **Explicit format** | "Return ONLY valid JSON" | Reduces wrapper text |
| **Null handling** | "use null for that field" | Handles missing data gracefully |

### Temperature = 0

```python
response = client.complete(
    prompt=prompt,
    system=EXTRACTION_SYSTEM_PROMPT,
    max_tokens=2000,
    temperature=0.0,  # Deterministic extraction
)
```

For extraction tasks, we want **deterministic, consistent** results. Temperature=0 ensures the same input produces the same output.

---

## Data Models & Schemas

### Pydantic Models

Every data structure is defined with Pydantic for type safety and validation:

```python
# src/denialops/models/case.py

class DenialCode(BaseModel):
    """A denial or procedure code."""
    code: str
    code_type: CodeType  # Enum: CPT, HCPCS, ICD_10_CM, CARC, RARC
    description: str | None = None

class CaseFacts(BaseModel):
    """Extracted facts from a denial letter."""
    case_id: str
    extraction_timestamp: datetime
    source_document: str

    denial_reason: str
    denial_reason_summary: str
    denial_codes: list[DenialCode] = []

    service: ServiceInfo | None = None
    payer: PayerInfo | None = None
    dates: CaseDates | None = None
    amounts: CaseAmounts | None = None
    contact_info: ContactInfo | None = None

    extraction_confidence: ExtractionConfidence | None = None
    missing_info: list[MissingInfo] = []
```

### Why Pydantic?

1. **Runtime validation**: Catches data errors early
2. **Documentation**: Auto-generates OpenAPI schemas
3. **Serialization**: `.model_dump(mode="json")` for storage
4. **IDE support**: Full autocomplete and type checking

### JSON Schemas

External contracts are defined in JSON Schema format:

```json
// schemas/case_facts.schema.json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "CaseFacts",
  "type": "object",
  "required": ["case_id", "denial_reason", "denial_codes"],
  "properties": {
    "denial_codes": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "code": {"type": "string"},
          "code_type": {"enum": ["CPT", "HCPCS", "ICD_10_CM", "CARC"]}
        }
      }
    }
  }
}
```

---

## Routing Logic

### Rule-Based Classification

The router uses **signal detection** with **weighted confidence scoring**:

```python
# src/denialops/pipeline/router.py

def route_case(facts: CaseFacts) -> RouteDecision:
    """Route case to appropriate action path."""
    signals = _detect_signals(facts)
    scores = _calculate_route_scores(signals)

    # Select highest scoring route
    best_route = max(scores.items(), key=lambda x: x[1])

    return RouteDecision(
        route=best_route[0],
        confidence=best_route[1],
        reasoning=_generate_reasoning(signals),
        alternative_routes=_get_alternatives(scores),
    )

def _detect_signals(facts: CaseFacts) -> dict[str, bool]:
    """Detect signals from case facts."""
    text = facts.denial_reason.lower()

    return {
        "prior_auth_missing": any(phrase in text for phrase in [
            "prior authorization", "pre-authorization", "pre-certification"
        ]),
        "coding_issue": any(phrase in text for phrase in [
            "coding", "modifier", "invalid code", "incorrect code"
        ]),
        "medical_necessity": any(phrase in text for phrase in [
            "medical necessity", "not medically necessary", "experimental"
        ]),
        "has_carc_197": any(c.code == "CO-197" for c in facts.denial_codes),
        # ... more signals
    }

def _calculate_route_scores(signals: dict) -> dict[RouteType, float]:
    """Calculate confidence score for each route."""
    scores = {route: 0.0 for route in RouteType}

    # Prior Auth signals
    if signals["prior_auth_missing"]:
        scores[RouteType.PRIOR_AUTH_NEEDED] += 0.4
    if signals["has_carc_197"]:  # CO-197 = Prior Auth required
        scores[RouteType.PRIOR_AUTH_NEEDED] += 0.3

    # Medical Necessity signals
    if signals["medical_necessity"]:
        scores[RouteType.MEDICAL_NECESSITY_APPEAL] += 0.5

    # Normalize to [0, 1]
    max_score = max(scores.values()) or 1
    return {k: v / max_score for k, v in scores.items()}
```

### Why Rule-Based Routing (Not ML)?

1. **Explainability**: Can show exactly why a route was chosen
2. **Controllability**: Easy to adjust rules based on domain expertise
3. **No training data needed**: Works on day 1
4. **Auditability**: Required for healthcare compliance

### When to Use ML Classification Instead

- When you have labeled training data (thousands of examples)
- When patterns are too complex for rules
- When you need to handle novel denial types automatically

---

## Error Handling & Fallbacks

### Graceful Degradation Pattern

```python
# src/denialops/pipeline/extract_facts.py

def extract_case_facts(case_id, text, llm_api_key, ...):
    """Extract facts with LLM, fall back to heuristics."""

    # Try LLM extraction first
    if llm_api_key:
        try:
            return _extract_with_llm(...)
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            # Fall through to heuristics

    # Fallback: heuristic extraction
    return _extract_with_heuristics(...)
```

### Why Fallbacks Matter

1. **API failures**: LLM APIs can timeout or rate limit
2. **Cost control**: Can disable LLM for low-priority requests
3. **Development**: Works without API keys for testing
4. **Resilience**: System keeps working even if LLM is down

### JSON Parsing Robustness

LLMs sometimes wrap JSON in markdown code blocks:

```python
def _parse_llm_response(response: str) -> dict:
    """Parse JSON from LLM response, handling markdown."""

    # Handle ```json ... ``` wrapper
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
```

---

## Testing Strategy

### Unit Tests for Each Component

```python
# tests/test_pipeline/test_extract_facts.py

def test_extract_dates():
    """Test date extraction from various formats."""
    text = "Date of service: 01/15/2024. Appeal within 180 days."

    dates = _extract_dates(text)

    assert dates.appeal_deadline_days == 180

def test_extract_codes():
    """Test CPT/CARC code extraction."""
    text = "CPT 72148, denial code CO-197"

    codes = _extract_codes(text)

    assert any(c.code == "72148" and c.code_type == CodeType.CPT
               for c in codes)
    assert any(c.code == "CO-197" and c.code_type == CodeType.CARC
               for c in codes)
```

### Integration Tests for API

```python
# tests/test_api.py

def test_create_case(client):
    """Test case creation endpoint."""
    response = client.post("/cases", json={"mode": "fast"})

    assert response.status_code == 201
    assert "case_id" in response.json()

def test_upload_document_no_case(client):
    """Test uploading to non-existent case."""
    response = client.post(
        "/cases/nonexistent/documents",
        files={"file": ("test.txt", b"content")},
        data={"doc_type": "denial_letter"}
    )

    assert response.status_code == 404
```

### Why No LLM Tests?

LLM outputs are non-deterministic and expensive. Instead:
- Mock LLM responses for unit tests
- Use heuristic fallback for CI
- Manual eval for quality (see `docs/eval.md`)

---

## Verified Mode: SBC Document Extraction

### Overview

**Verified Mode** enables plan-specific, citation-backed recommendations by extracting rules from the user's Summary of Benefits and Coverage (SBC) document.

```
Fast Mode (denial letter only):
  "You should file an appeal within the deadline"

Verified Mode (denial letter + SBC):
  "You have 180 days to file an appeal [SBC p.8, Appeals]"
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Document Processing                           │
│                                                                      │
│  Denial Letter ─────► extract_case_facts() ───► CaseFacts           │
│                                                                      │
│  SBC Document ──────► extract_plan_rules() ───► PlanRules           │
│                              │                      │                │
│                              ▼                      ▼                │
│                        ┌─────────────────────────────────┐          │
│                        │ generate_action_plan()          │          │
│                        │   + generate_document_pack()    │          │
│                        │                                 │          │
│                        │ Combines facts + rules          │          │
│                        │ Adds source citations           │          │
│                        └─────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

### PlanRules Model

```python
# src/denialops/models/plan_rules.py

class PlanRules(BaseModel):
    """Extracted rules from SBC/EOC documents."""

    case_id: str
    source_document: str
    document_type: PlanDocumentType  # SBC, EOC, COC, SPD

    # Plan identification
    plan_info: PlanInfo

    # Cost sharing
    deductibles: Deductibles | None = None
    out_of_pocket_max: OutOfPocketMax | None = None

    # Coverage rules (with source citations)
    prior_authorization_rules: list[PriorAuthRule] = []
    medical_necessity_criteria: list[MedicalNecessityCriteria] = []
    exclusions: list[Exclusion] = []

    # Appeal process
    appeal_rights: AppealRights | None = None

    # Quality tracking
    extraction_quality: ExtractionQuality
```

### Source Citations

Every extracted rule includes its source for grounding:

```python
class SourceCitation(BaseModel):
    source_page: int | None
    source_section: str | None
    source_quote: str | None

    def format_citation(self) -> str:
        """[SBC p.8, Appeals]"""
        parts = []
        if self.source_page:
            parts.append(f"p.{self.source_page}")
        if self.source_section:
            parts.append(self.source_section)
        return f"[SBC {', '.join(parts)}]" if parts else ""

# Usage
rule = plan_rules.appeal_rights
citation = rule.get_citation().format_citation()
message = f"You have {rule.internal_appeal_deadline_days} days {citation}"
# "You have 180 days [SBC p.8, Appeals]"
```

### Two-Tier Extraction Strategy

```python
def extract_plan_rules(case_id, text, llm_api_key="", ...):
    """Extract with LLM if available, else heuristics."""

    if llm_api_key:
        try:
            return _extract_with_llm(...)  # Confidence: 0.5-0.9
        except Exception as e:
            logger.warning(f"LLM failed: {e}")

    return _extract_with_heuristics(...)  # Confidence: 0.4
```

### Why Citations Matter

1. **Trust**: Users can verify AI recommendations against source documents
2. **Compliance**: Healthcare advice must be traceable
3. **Debugging**: Easy to identify extraction errors
4. **Legal protection**: Clear provenance for generated guidance

---

## Production Considerations

### What's Needed for Production

| Component | Current State | Production Ready |
|-----------|---------------|------------------|
| LLM abstraction | ✅ Multi-provider | Add retry, circuit breaker |
| Error handling | ✅ Fallbacks | Add structured logging |
| Storage | ⚠️ File-based | Switch to S3/database |
| Authentication | ❌ None | Add API keys, OAuth |
| Rate limiting | ❌ None | Add per-user limits |
| Caching | ❌ None | Cache LLM responses |
| Monitoring | ❌ None | Add Prometheus metrics |
| Secrets | ⚠️ .env file | Use vault/secrets manager |

### Cost Optimization

LLM calls are expensive. Production systems need:

```python
# Example: Caching layer
def extract_with_cache(text: str, cache: Redis) -> CaseFacts:
    # Hash the input for cache key
    cache_key = hashlib.sha256(text.encode()).hexdigest()

    # Check cache first
    if cached := cache.get(cache_key):
        return CaseFacts.model_validate_json(cached)

    # Call LLM
    result = _extract_with_llm(text)

    # Cache for 24 hours
    cache.setex(cache_key, 86400, result.model_dump_json())

    return result
```

### Scaling Considerations

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Load      │────▶│   FastAPI   │────▶│   LLM       │
│  Balancer   │     │  Workers    │     │   Gateway   │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                    │
                           ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  PostgreSQL │     │   Redis     │
                    │  (metadata) │     │  (cache)    │
                    └─────────────┘     └─────────────┘
```

---

## Key Learnings for AI/ML Engineers

### 1. LLM Applications Are Software Engineering + Prompt Engineering

The code structure, error handling, and testing matter as much as the prompts.

### 2. Hybrid Approaches Win

Pure LLM extraction is expensive and sometimes wrong. Combine:
- **Regex** for structured data (phone numbers, codes)
- **LLM** for unstructured understanding (denial reasons)
- **Rules** for classification (routing)

### 3. Always Have Fallbacks

```python
# Pattern you'll use everywhere
try:
    result = expensive_llm_call()
except Exception:
    result = cheap_fallback()
```

### 4. Prompt Engineering Patterns

| Pattern | Use When |
|---------|----------|
| JSON mode | Need structured output |
| Few-shot examples | Showing format/style |
| Chain-of-thought | Complex reasoning |
| System prompts | Setting context/role |
| Temperature=0 | Extraction tasks |
| Temperature=0.7+ | Creative generation |

### 5. Observability is Critical

Every LLM call should log:
- Input (or hash)
- Output
- Latency
- Token count
- Cost

### 6. Design for Testability

- Abstract LLM behind interface (easy to mock)
- Save intermediate artifacts (debug production issues)
- Separate deterministic from non-deterministic code

### 7. The 80/20 Rule

For most extraction tasks:
- Regex handles 80% of cases
- LLM handles the complex 20%
- This is 10x cheaper than LLM-for-everything

---

## Further Reading

- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [LangChain Docs](https://python.langchain.com/) - More sophisticated chains
- [OpenAI Cookbook](https://cookbook.openai.com/) - Production patterns
- [Anthropic Claude Docs](https://docs.anthropic.com/) - Claude-specific patterns

---

## File Reference

| File | Purpose |
|------|---------|
| `src/denialops/pipeline/extract_facts.py` | LLM extraction + fallback |
| `src/denialops/pipeline/extract_plan_rules.py` | SBC/EOC extraction (Verified Mode) |
| `src/denialops/pipeline/router.py` | Rule-based classification |
| `src/denialops/pipeline/generate_plan.py` | Action plan generation with citations |
| `src/denialops/pipeline/generate_docs.py` | Document generation with policy citations |
| `src/denialops/llm/client.py` | Multi-provider abstraction + retry logic |
| `src/denialops/llm/prompts.py` | Centralized prompt templates |
| `src/denialops/models/case.py` | Core data models |
| `src/denialops/models/plan_rules.py` | PlanRules + SourceCitation models |
| `src/denialops/api/routes.py` | Pipeline orchestration |
| `schemas/*.json` | External contracts |
| `docs/phase-2-learnings.md` | Production LLM patterns |
| `docs/phase-3-learnings.md` | Verified Mode and source citations |
