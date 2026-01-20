# Phase 4: Advanced LLM Features - Implementation Learnings

## Overview

Phase 4 adds advanced LLM-powered features to the DenialOps pipeline:
- **Personalized Summaries**: Generate user-friendly situation summaries
- **Grounding Validation**: Detect and prevent hallucinations in generated content
- **Success Prediction**: Predict appeal success likelihood
- **Multi-Document Support**: Combine information from denial letters, SBCs, and EOBs

## New Components

### 1. LLM Generation Module (`pipeline/llm_generation.py`)

Core module providing three main capabilities:

```python
# Generate personalized summary of the user's situation
summary = generate_personalized_summary(
    facts=facts,
    route=route_decision,
    plan_rules=plan_rules,
    llm_api_key=settings.llm_api_key,
)

# Predict success likelihood
prediction = predict_success(
    facts=facts,
    route=route_decision,
    plan_rules=plan_rules,
)

# Validate generated content for hallucinations
grounding = validate_grounding(
    content=appeal_letter_content,
    facts=facts,
    plan_rules=plan_rules,
)
```

### 2. EOB Extraction Module (`pipeline/extract_eob.py`)

Extracts structured information from Explanation of Benefits documents:

```python
eob_facts = extract_eob_facts(
    case_id=case_id,
    text=extracted_text,
    llm_api_key=settings.llm_api_key,
)
```

Extracts:
- Claim status (paid, denied, partially paid, pending)
- Provider information (name, NPI, network status)
- Financial details (billed, allowed, paid amounts)
- Member costs (deductible, copay, coinsurance)
- Denial codes and reasons
- Accumulator information (YTD deductible, OOP max)

### 3. EOB Data Models (`models/eob.py`)

Pydantic models for EOB data:
- `EOBFacts` - Main container for extracted EOB information
- `ServiceLine` - Individual service line items
- `ProviderInfo` - Provider details
- `MemberCostSummary` - Patient cost breakdown
- `AccumulatorInfo` - Year-to-date accumulator status
- `ClaimStatus` - Enum for claim status types

## Key Patterns

### 1. Graceful Degradation

All Phase 4 features work without an LLM by falling back to heuristics:

```python
def generate_personalized_summary(...) -> PersonalizedSummary:
    if llm_api_key:
        try:
            return _generate_with_llm(...)
        except Exception:
            pass
    return _generate_with_heuristics(...)
```

### 2. Grounding Validation

Prevent hallucinations by validating generated content against source facts:

```python
def validate_grounding(...) -> GroundingResult:
    # Extract codes, dates, and amounts from generated content
    found_codes = extract_codes_from_content(content)
    found_dates = extract_dates_from_content(content)

    # Compare against known facts
    hallucinated_codes = [c for c in found_codes if c not in known_codes]
    hallucinated_dates = [d for d in found_dates if d not in known_dates]

    return GroundingResult(
        is_grounded=len(hallucinated_codes) == 0,
        hallucinated_codes=hallucinated_codes,
        hallucinated_dates=hallucinated_dates,
    )
```

### 3. Multi-Document Information Fusion

Combine information from multiple documents:

```python
# In routes.py run_pipeline()

# Extract from denial letter
facts = extract_case_facts(denial_text)

# Extract from SBC if uploaded
if sbc_doc:
    plan_rules = extract_plan_rules(sbc_text)

# Extract from EOB if uploaded
if eob_doc:
    eob_facts = extract_eob_facts(eob_text)

    # Enrich case facts with EOB information
    if eob_facts.denial_codes and not facts.denial_codes:
        facts.denial_codes = eob_facts.denial_codes
```

### 4. Success Prediction Heuristics

Multi-factor scoring for appeal success likelihood:

```python
def _predict_with_heuristics(facts, route, plan_rules):
    score = 0.5  # Start at 50%
    factors_for = []
    factors_against = []

    # Route-specific factors
    if route.route == RouteType.CLAIM_CORRECTION_RESUBMIT:
        factors_for.append("Coding errors are often correctable")
        score += 0.2

    # Deadline compliance
    if facts.dates.appeal_deadline:
        days_left = (appeal_deadline - date.today()).days
        if days_left > 30:
            factors_for.append("Sufficient time remaining")
            score += 0.1

    # Plan rules availability
    if plan_rules:
        factors_for.append("Plan policy details available")
        score += 0.1

    return SuccessPrediction(
        likelihood="high" if score >= 0.7 else "medium" if score >= 0.4 else "low",
        score=score,
        factors_for=factors_for,
        factors_against=factors_against,
    )
```

## API Enhancements

### Enhanced Pipeline Response

The `RunPipelineResponse` now includes:

```python
class RunPipelineResponse(BaseModel):
    status: str
    route: RouteType | None
    confidence: float | None
    success_prediction: SuccessPredictionResponse | None  # New
    grounding_validation: GroundingValidationResponse | None  # New
    artifacts: list[str]
    error: str | None
```

### New Artifacts Generated

- `personalized_summary.json` - User-friendly situation summary
- `success_prediction.json` - Appeal success factors and score
- `grounding_validation.json` - Validation results for generated content
- `eob_facts.json` - Extracted EOB information (if EOB uploaded)

## Testing Strategy

### Test Coverage

Phase 4 adds 38 new tests covering:

1. **LLM Generation Tests** (`test_llm_generation.py`)
   - Personalized summary generation (4 tests)
   - Success prediction (4 tests)
   - Grounding validation (6 tests)

2. **EOB Extraction Tests** (`test_extract_eob.py`)
   - Claim number extraction (4 tests)
   - Provider info extraction (3 tests)
   - Amount extraction (3 tests)
   - Member cost extraction (4 tests)
   - Denial info extraction (2 tests)
   - Claim status determination (3 tests)
   - Integration tests (5 tests)

### Testing Heuristics Without LLM

All tests use `llm_api_key=""` to test heuristic fallbacks:

```python
def test_generates_summary_without_llm(self, sample_case_facts, sample_route_decision):
    result = generate_personalized_summary(
        facts=sample_case_facts,
        route=sample_route_decision,
        llm_api_key="",  # Forces heuristic path
    )

    assert result.is_llm_generated is False
    assert result.situation_summary  # Still generates content
```

## File Structure

```
src/denialops/
├── models/
│   ├── eob.py                    # EOB data models (new)
│   └── __init__.py               # Updated exports
├── pipeline/
│   ├── extract_eob.py            # EOB extraction (new)
│   ├── llm_generation.py         # LLM features (new)
│   └── __init__.py               # Updated exports
└── api/
    └── routes.py                 # Enhanced pipeline endpoint

tests/test_pipeline/
├── test_extract_eob.py           # EOB extraction tests (new)
└── test_llm_generation.py        # LLM generation tests (new)
```

## Key Learnings

### 1. Graceful Degradation is Essential

Users may not have LLM API access. Every feature should work (with reduced quality) using heuristics alone.

### 2. Grounding Prevents Hallucination

LLMs can generate plausible-sounding but incorrect information. Validating against source facts catches:
- Made-up procedure codes
- Incorrect dates
- Fabricated dollar amounts

### 3. Multi-Document Enrichment

Information from multiple documents can be combined to provide a more complete picture:
- Denial letter provides the core denial reason
- SBC provides plan-specific rules and deadlines
- EOB provides financial details and denial codes

### 4. Test Date Sensitivity

When testing date-based logic (like deadline calculations), use future dates to ensure tests remain valid:

```python
# Bad: Uses past date
appeal_deadline=date(2024, 8, 1)

# Good: Uses future date
appeal_deadline=date(2026, 8, 1)
```

### 5. Dataclass vs Pydantic

Phase 4 uses dataclasses for return types from LLM functions (simpler) while Pydantic models are used for serializable data:

```python
# Dataclass for internal return type
@dataclass
class PersonalizedSummary:
    situation_summary: str
    is_llm_generated: bool

# Pydantic for API response and storage
class SuccessPredictionResponse(BaseModel):
    likelihood: str
    score: float
```

## Next Steps

Potential Phase 5 enhancements:
- Streaming LLM responses for real-time feedback
- Caching for repeated document analysis
- Batch processing for multiple cases
- A/B testing for LLM prompts
- User feedback integration for prediction improvement
