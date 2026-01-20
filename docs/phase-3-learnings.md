# Phase 3 Learnings: Verified Mode with Source Citations

This document explains what we built in Phase 3 and the AI/ML engineering concepts behind the "Verified Mode" feature.

---

## Table of Contents

1. [Overview](#overview)
2. [SBC Document Extraction](#1-sbc-document-extraction)
3. [Source Citations for Grounding](#2-source-citations-for-grounding)
4. [Verified vs Fast Mode](#3-verified-vs-fast-mode)
5. [Code Walkthrough](#4-code-walkthrough)
6. [Key Takeaways](#5-key-takeaways)

---

## Overview

Phase 3 introduced **Verified Mode** - the ability to extract policy rules from SBC (Summary of Benefits and Coverage) documents and use them to provide grounded, citation-backed recommendations.

| Feature | Why It Matters |
|---------|----------------|
| SBC Extraction | Extracts actual policy rules from user's plan documents |
| Source Citations | Every recommendation cites the specific page/section in the SBC |
| Grounded Recommendations | Advice is based on actual policy language, not assumptions |
| Heuristic Fallback | Works even without LLM access |

**Files Created/Modified:**
- `src/denialops/models/plan_rules.py` (NEW - 297 lines)
- `src/denialops/pipeline/extract_plan_rules.py` (NEW - 454 lines)
- `src/denialops/pipeline/generate_plan.py` (Updated - added plan_rules support)
- `src/denialops/pipeline/generate_docs.py` (Updated - added citations)
- `tests/test_pipeline/test_extract_plan_rules.py` (NEW - 10 tests)

---

## 1. SBC Document Extraction

### The Problem

In "Fast Mode", our recommendations were generic:
- "You have the right to appeal"
- "Prior authorization may be required"
- "Check your plan for deadlines"

Without the actual policy document, we couldn't provide specific guidance.

### The Solution: Extract Rules from SBC

SBC (Summary of Benefits and Coverage) documents contain:
- Deductibles and out-of-pocket maximums
- Prior authorization requirements
- Medical necessity definitions
- Appeal rights and deadlines
- Exclusions and limitations

We extract this structured information using LLM or heuristics.

### The Data Model

```python
class PlanRules(BaseModel):
    """Extracted rules from SBC/EOC documents (Verified mode)."""

    case_id: str
    source_document: str
    document_type: PlanDocumentType
    extracted_at: datetime

    # Plan identification
    plan_info: PlanInfo

    # Cost sharing
    deductibles: Deductibles | None = None
    out_of_pocket_max: OutOfPocketMax | None = None

    # Coverage rules
    prior_authorization_rules: list[PriorAuthRule] = []
    medical_necessity_criteria: list[MedicalNecessityCriteria] = []
    exclusions: list[Exclusion] = []

    # Appeal process
    appeal_rights: AppealRights | None = None

    # Quality metrics
    extraction_quality: ExtractionQuality
```

### LLM Extraction vs Heuristics

We use a **two-tier extraction strategy**:

```python
def extract_plan_rules(case_id, text, llm_api_key="", ...):
    # Try LLM extraction if API key provided
    if llm_api_key:
        try:
            return _extract_with_llm(...)
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")

    # Fallback to heuristic extraction
    return _extract_with_heuristics(...)
```

| Method | Confidence | Use Case |
|--------|-----------|----------|
| LLM | 0.5-0.9 | Production with API access |
| Heuristics | 0.4 | Offline/testing/fallback |

### Heuristic Extraction

Regex patterns for common SBC formats:

```python
# Extract deductible amounts
deductible_pattern = r"\$([0-9,]+)\s*(?:individual|single)?\s*deductible"

# Extract appeal deadlines
appeal_patterns = [
    r"(\d+)\s*(?:calendar\s*)?days?\s*(?:to|for)\s*appeal",
    r"within\s*(\d+)\s*(?:calendar\s*)?days?\s*(?:of|from)",
]
```

### Key Learning

> **Always have a fallback.** LLM APIs can fail. Heuristics provide degraded but functional service when the LLM is unavailable.

---

## 2. Source Citations for Grounding

### The Problem

LLM-generated content can "hallucinate" - making up information that sounds plausible but isn't in the source documents. This is dangerous for legal/medical advice.

### The Solution: Cite Every Claim

Every extracted rule includes its source:

```python
class SourceCitation(BaseModel):
    """Citation to source document for grounding."""

    source_page: int | None      # Page number
    source_section: str | None   # Section name
    source_quote: str | None     # Direct quote

    def format_citation(self) -> str:
        """Format citation for display."""
        parts = []
        if self.source_page:
            parts.append(f"p.{self.source_page}")
        if self.source_section:
            parts.append(self.source_section)
        return f"[SBC {', '.join(parts)}]" if parts else ""
```

### How Citations Flow Through the System

```
SBC Document
    │
    ▼
extract_plan_rules()
    │
    ├─► PriorAuthRule(source_page=5, source_section="PA Requirements")
    │
    ├─► AppealRights(source_page=8, internal_appeal_deadline_days=180)
    │
    ▼
generate_action_plan()
    │
    ├─► "You have 180 days to appeal [SBC p.8]"
    │
    ▼
generate_appeal_letter()
    │
    └─► **Your Plan's Appeal Rights** (from SBC)
        - You have 180 days to file an appeal [SBC p.8, Appeals]
```

### Example Output with Citations

**Without Citations (Fast Mode):**
> "You should file an appeal with supporting clinical documentation."

**With Citations (Verified Mode):**
> "You should file an appeal with supporting clinical documentation. Your plan allows 180 days to file an appeal. [SBC p.8, Appeals]"

### Key Learning

> **Grounding prevents hallucination.** Always cite sources for LLM-generated content, especially for legal/medical information.

---

## 3. Verified vs Fast Mode

### Mode Comparison

| Aspect | Fast Mode | Verified Mode |
|--------|-----------|---------------|
| SBC Required | No | Yes |
| Processing Time | Faster | Slower |
| Recommendations | Generic | Plan-specific |
| Citations | None | Full citations |
| Assumptions | Many | Fewer |
| Confidence | Lower | Higher |

### How Mode Affects Output

```python
def _identify_assumptions(facts, mode, plan_rules=None):
    assumptions = []

    if mode == "fast" and not plan_rules:
        # Fast mode warning
        assumptions.append(Assumption(
            assumption="Plan-specific coverage details not verified",
            impact="Recommendations may not reflect your specific plan",
            how_to_verify="Upload your SBC for verified guidance",
        ))

    elif plan_rules and plan_rules.extraction_quality.confidence < 0.7:
        # Verified mode with low confidence
        assumptions.append(Assumption(
            assumption="Plan rules extracted with moderate confidence",
            impact="Some plan details may need verification",
            how_to_verify="Review your SBC document directly",
        ))

    return assumptions
```

### User Flow

```
User uploads denial letter
    │
    ├─► No SBC → Fast Mode (generic guidance)
    │
    └─► With SBC → Verified Mode (cited guidance)
            │
            ├─► Extract plan rules
            │
            ├─► Generate action plan with citations
            │
            └─► Generate documents with policy references
```

### Key Learning

> **Progressive enhancement.** Provide value at every level of input. More documents = better guidance.

---

## 4. Code Walkthrough

### File: `src/denialops/models/plan_rules.py`

```
Lines 1-30:    Enums (PlanDocumentType, PlanType)
Lines 32-54:   SourceCitation model
Lines 56-71:   PlanInfo model
Lines 73-96:   Cost sharing models (Deductibles, OutOfPocketMax)
Lines 98-120:  PriorAuthRule model with citations
Lines 122-143: MedicalNecessityCriteria model
Lines 145-167: Exclusion model
Lines 169-199: AppealRights model
Lines 201-212: TimelyFiling model
Lines 214-226: ExtractionQuality model
Lines 228-297: Main PlanRules model with helper methods
```

### File: `src/denialops/pipeline/extract_plan_rules.py`

```
Lines 1-26:    Imports and logger
Lines 27-124:  LLM prompts for SBC extraction
Lines 126-170: Main extract_plan_rules function
Lines 172-204: LLM extraction function
Lines 206-221: JSON response parser
Lines 223-377: Build PlanRules from LLM response
Lines 379-389: Date parsing helper
Lines 391-454: Heuristic extraction fallback
```

### How Extraction Flows

```
extract_plan_rules(case_id, text, llm_api_key)
    │
    ├─► LLM path:
    │   │
    │   ├─► _extract_with_llm()
    │   │       │
    │   │       ├─► create_llm_client()
    │   │       │
    │   │       ├─► client.complete(prompt, system)
    │   │       │
    │   │       ├─► _parse_llm_response()
    │   │       │
    │   │       └─► _build_plan_rules_from_llm()
    │   │
    │   └─► Return PlanRules (confidence: 0.5-0.9)
    │
    └─► Heuristic path:
        │
        ├─► _extract_with_heuristics()
        │       │
        │       ├─► Regex for deductibles
        │       │
        │       ├─► Regex for appeal deadlines
        │       │
        │       └─► Regex for PA mentions
        │
        └─► Return PlanRules (confidence: 0.4)
```

### Integration Points

**routes.py:**
```python
# Stage 2b: Extract plan rules from SBC if uploaded
plan_rules = None
sbc_doc = next((d for d in documents if d.get("doc_type") in ("sbc", "eoc")), None)
if sbc_doc:
    sbc_extracted = extract_text(sbc_path)
    plan_rules = extract_plan_rules(case_id, sbc_extracted, llm_api_key)
    storage.store_artifact(case_id, "plan_rules.json", plan_rules.model_dump())

# Stage 4: Generate action plan (now with plan_rules)
action_plan = generate_action_plan(facts, route, mode, plan_rules=plan_rules)

# Stage 5: Generate documents (now with citations)
documents = generate_document_pack(facts, action_plan, plan_rules=plan_rules)
```

---

## 5. Key Takeaways

### For Senior AI/ML Engineers

1. **Ground LLM outputs** - Always cite sources for generated content

2. **Graceful degradation** - Have heuristic fallbacks when LLM fails

3. **Track extraction quality** - Include confidence scores and warnings

4. **Progressive enhancement** - More input = better output

5. **Structured extraction** - Use JSON schemas to get structured data from LLM

### Design Patterns Used

| Pattern | Where | Why |
|---------|-------|-----|
| **Strategy** | LLM vs Heuristics | Switch extraction methods at runtime |
| **Builder** | `_build_plan_rules_from_llm` | Construct complex objects step by step |
| **Null Object** | `plan_rules: PlanRules | None` | Handle missing data gracefully |
| **Citation Pattern** | `SourceCitation` | Track provenance of extracted data |
| **Progressive Enhancement** | Fast → Verified | Better output with more input |

### Anti-Patterns Avoided

| Anti-Pattern | What We Did Instead |
|--------------|-------------------|
| Assuming LLM is always available | Heuristic fallback |
| Trusting LLM output blindly | Source citations |
| All-or-nothing extraction | Partial extraction with quality scores |
| Hardcoded policy rules | Extracted from user's documents |

---

## Quick Reference

### Extract Plan Rules

```python
from denialops.pipeline import extract_plan_rules
from denialops.models.documents import ExtractedText

# Extract from SBC text
plan_rules = extract_plan_rules(
    case_id="case-123",
    text=extracted_text,       # ExtractedText object
    llm_api_key="sk-...",      # Optional - heuristics if empty
    llm_model="gpt-4o",        # Optional
    llm_provider="openai",     # "openai" or "anthropic"
)

# Access extracted data
print(plan_rules.appeal_rights.internal_appeal_deadline_days)  # 180
print(plan_rules.prior_authorization_rules[0].service_category)  # "Imaging"
print(plan_rules.extraction_quality.confidence)  # 0.75
```

### Use Citations

```python
# Get citation for a rule
rule = plan_rules.appeal_rights
citation = rule.get_citation()
print(citation.format_citation())  # "[SBC p.8, Appeals]"

# Include in generated text
deadline = rule.internal_appeal_deadline_days
message = f"You have {deadline} days to appeal {citation.format_citation()}"
# "You have 180 days to appeal [SBC p.8, Appeals]"
```

### Check Extraction Quality

```python
if plan_rules.extraction_quality.confidence < 0.5:
    print("Warning: Low confidence extraction")
    print(f"Warnings: {plan_rules.extraction_quality.warnings}")
```

---

## Next Steps

Phase 4 could include:

1. **LLM-powered summaries** - Use LLM to generate personalized situation summaries

2. **Grounding validation** - Check generated content against source facts

3. **Multi-document extraction** - Combine rules from multiple documents

4. **Appeal success prediction** - ML model to predict appeal success likelihood
