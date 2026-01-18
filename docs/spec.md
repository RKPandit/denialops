# DenialOps — MVP v1 Build Spec

## Product Summary

**DenialOps** is a Claim Denial Understanding + Next-Best-Action + Document Pack Generator.

Given a denial letter and optional plan documents, produce:
- Plain-language denial explanation
- Next-best-action route (not always appeal)
- Evidence checklist + deadline timeline
- Generated document pack (appeal letter, reconsideration request, call script)

## Target Users

| User | Need |
|------|------|
| Patient / caregiver | Plain-English explanation + what to do next + draft documents |
| Provider office staff | Checklist + missing evidence + PA/reconsideration pack |
| Power user | Structured case summary for tracking and escalation |

---

## Operating Modes

### Fast Mode (minimum viable input)
**Inputs:**
- Denial letter (PDF text-based or `.txt`)
- Optional: EOB (PDF)
- Optional: user context JSON (procedure, diagnosis, date_of_service, provider, urgency)

**Output behavior:**
- Helpful but labeled: *"Plan-specific coverage not verified."*
- No policy citations

### Verified Mode (grounded output)
**Inputs:**
- Everything in Fast mode, plus one of:
  - SBC (Summary of Benefits and Coverage), OR
  - EOC / Certificate of Coverage

**Output behavior:**
- Citations to exact document section/page
- Higher confidence routing
- Stronger appeal drafts

---

## Supported Routes (MVP v1)

| Route | Trigger | Output Focus |
|-------|---------|--------------|
| `prior_auth_needed` | Missing PA, wrong PA pathway, incomplete PA | PA form guidance, missing items checklist |
| `claim_correction_resubmit` | Coding/modifier/NPI/eligibility issue | Coding checklist, resubmission guidance |
| `medical_necessity_appeal` | Clinical evidence needed (Verified mode preferred) | Appeal letter, evidence checklist, clinician letter template |

Routes NOT in MVP v1: `exclusion`, `admin_denial`, `escalation`, `benefits_verification`

---

## Inputs

### Required
- Denial letter: PDF (text-based) or `.txt`

### Optional (Fast mode)
- EOB: PDF
- User context JSON:
```json
{
  "payer_name": "string",
  "plan_name": "string",
  "state": "string (2-letter)",
  "service_name": "string",
  "date_of_service": "YYYY-MM-DD",
  "provider_name": "string",
  "urgency": "standard | expedited"
}
```

### Optional (Verified mode)
- SBC or EOC: PDF (text-based)

---

## Outputs (per case)

All artifacts written to: `./artifacts/{case_id}/`

### 1. `case_facts.json`
Structured extraction from denial letter. Must validate against schema.

### 2. `route.json`
Selected route + confidence + reasoning. Must validate against schema.

### 3. `action_plan.json`
Structured action plan with steps, deadlines, missing items. Must validate against schema.

### 4. `action_plan.md`
Human-readable markdown version of action plan.

### 5. Document Pack (route-dependent)
- `appeal_letter.md` (for `medical_necessity_appeal`)
- `resubmit_checklist.md` (for `claim_correction_resubmit`)
- `pa_checklist.md` (for `prior_auth_needed`)
- `call_script.md` (all routes)

### 6. `plan_rules.json` (Verified mode only)
Extracted policy rules with section/page anchors.

---

## API (FastAPI)

### Create case
```
POST /cases
Content-Type: application/json

Request: { "mode": "fast" | "verified", "user_context": {...} }
Response: { "case_id": "uuid" }
```

### Upload document
```
POST /cases/{case_id}/documents
Content-Type: multipart/form-data

Form fields:
  - file: binary
  - doc_type: "denial_letter" | "eob" | "sbc" | "eoc"

Response: { "document_id": "uuid", "stored_path": "..." }
```

### Run pipeline
```
POST /cases/{case_id}/run
Response: {
  "status": "completed" | "failed",
  "route": "prior_auth_needed" | "claim_correction_resubmit" | "medical_necessity_appeal",
  "artifacts": ["case_facts.json", "route.json", "action_plan.json", ...]
}
```

### Get artifacts
```
GET /cases/{case_id}/artifacts
Response: { "artifacts": [{ "name": "...", "size": ..., "path": "..." }] }
```

### Get single artifact
```
GET /cases/{case_id}/artifacts/{filename}
Response: file content
```

---

## Pipeline Stages

### 1. `extract_text`
- PDF → plain text (pdfminer.six)
- Preserve page boundaries for citation anchoring
- Store as `extracted_text.txt`

### 2. `extract_case_facts`
- Deterministic extraction first (regex for codes, dates, amounts)
- LLM-assisted extraction for unstructured fields
- Output must validate against `case_facts.schema.json`
- Extract:
  - denial_reason (free text)
  - denial_codes (CPT, HCPCS, ICD-10, internal payer codes)
  - service_description
  - date_of_service
  - date_of_denial
  - appeal_deadline
  - payer_name, plan_hints
  - contact_info (phone, fax, address)
  - amounts (billed, allowed, patient_responsibility)

### 3. `extract_plan_rules` (Verified mode only)
- Parse SBC/EOC into structured rules
- Extract: benefits, exclusions, PA requirements, appeal timelines
- Output: `plan_rules.json` with page/section anchors

### 4. `route_case`
- Rule-based routing with optional ML classifier
- Inputs: case_facts + plan_rules (if available)
- Output: route + confidence + reasoning
- Routing logic:
  - If denial mentions "prior authorization" / "PA" / "not pre-certified" → `prior_auth_needed`
  - If denial mentions "coding" / "modifier" / "invalid code" / "NPI" → `claim_correction_resubmit`
  - If denial mentions "not medically necessary" / "medical necessity" → `medical_necessity_appeal`
  - Default with low confidence → `medical_necessity_appeal`

### 5. `generate_action_plan`
- LLM generation with structured facts as input
- Must not invent codes/dates not present in case_facts
- Uncertain fields marked explicitly as "Unknown - user input needed"
- In Verified mode: must cite plan_rules with section references
- Output: `action_plan.json` + `action_plan.md`

### 6. `generate_document_pack`
- Route-specific document generation
- Templates + LLM personalization
- All documents include:
  - Assumptions section (when data missing)
  - Disclaimer: "This is not legal or medical advice"
  - Citations (Verified mode)

---

## System Boundaries

### Will NOT do (MVP v1)
- OCR for image-based PDFs (text PDFs only)
- Portal scraping or authentication
- Real-time eligibility checks
- PHI retention beyond session (artifacts are local/ephemeral)
- Legal or medical advice (explicit disclaimers required)

### Security / Compliance
- All uploaded documents stored locally in `./artifacts/`
- No external transmission of PHI
- Logs must not contain PHI (redact before logging)
- LLM calls must use appropriate data handling (no training on user data)

---

## Acceptance Criteria

### Functional
- [ ] `POST /cases` creates case directory
- [ ] `POST /cases/{id}/documents` stores files correctly
- [ ] `POST /cases/{id}/run` produces all required artifacts
- [ ] All JSON artifacts validate against schemas
- [ ] Fast mode works without SBC/EOC
- [ ] Verified mode produces citations when SBC/EOC provided

### Quality
- [ ] `make eval` processes all files in `data/samples/` end-to-end
- [ ] Extraction captures denial_reason, codes, dates with >90% recall on golden set
- [ ] Routing accuracy >85% on labeled test cases
- [ ] Generated documents contain no hallucinated codes/dates (100% grounded)

### Engineering
- [ ] CI runs: lint, typecheck, unit tests, integration tests, eval
- [ ] All services containerized (Docker)
- [ ] API response time <30s for typical denial letter

---

## Demo Script (2-minute tryout)

```bash
# 1. Start the server
make run

# 2. Create a case
curl -X POST http://localhost:8000/cases \
  -H "Content-Type: application/json" \
  -d '{"mode": "fast"}'
# Returns: {"case_id": "abc123"}

# 3. Upload denial letter
curl -X POST http://localhost:8000/cases/abc123/documents \
  -F "file=@sample_denial.pdf" \
  -F "doc_type=denial_letter"

# 4. Run pipeline
curl -X POST http://localhost:8000/cases/abc123/run
# Returns: {"status": "completed", "route": "prior_auth_needed", ...}

# 5. View action plan
curl http://localhost:8000/cases/abc123/artifacts/action_plan.md
```

---

## Appendix: Route Decision Tree

```
denial_letter
    │
    ├─ contains "prior auth" / "PA" / "pre-certification"?
    │   └─ YES → prior_auth_needed
    │
    ├─ contains "coding" / "modifier" / "invalid" / "NPI"?
    │   └─ YES → claim_correction_resubmit
    │
    ├─ contains "medical necessity" / "not medically necessary"?
    │   └─ YES → medical_necessity_appeal
    │
    └─ DEFAULT (low confidence) → medical_necessity_appeal
```
