# DenialOps — MVP v0 Build Spec (Python-only)

## Goal
Given a denial letter (PDF or text), produce:
1) `case_facts.json` (structured extraction)
2) `route.json` (one of two routes)
3) `action_plan.md` (next steps, no policy citations in v0)

## Non-goals (v0)
- No UI/web frontend
- No insurer portal scraping
- No plan document parsing (Verified mode is v1)
- No OCR requirement (nice-to-have; text PDFs ok)

## Supported Routes (v0)
- `prior_auth_needed`
- `claim_correction_resubmit`

## Inputs
- Denial letter upload:
  - PDF (text-based) or `.txt`
- Optional user-provided metadata (JSON):
  - payer_name, plan_hint, state, service_name, date_of_service

## Outputs (artifacts per case)
Artifacts must be written to: `./artifacts/{case_id}/`
- `case_facts.json` (must validate schema)
- `route.json` (must validate schema)
- `action_plan.md` (markdown)

## API (FastAPI)
### Create case
POST `/cases`
Response: `{ "case_id": "..." }`

### Ingest document
POST `/cases/{case_id}/ingest`
Multipart upload: `file`
Stores original at `./artifacts/{case_id}/inputs/{filename}`

### Run pipeline (sync)
POST `/cases/{case_id}/run`
Response includes route + paths of artifacts.

### List artifacts
GET `/cases/{case_id}/artifacts`
Returns list of artifact files and sizes.

## Pipeline stages (v0)
1) `extract_text`
   - PDF -> text (pdfminer.six) OR accept plain text files
2) `extract_case_facts`
   - Use deterministic extraction first (regex/heuristics)
   - Optional: LLM call allowed, but output must match schema
3) `route_case`
   - Rule-based routing with explainability field
4) `generate_action_plan`
   - LLM or templated generation
   - Must not invent codes/dates not present in case_facts
   - If uncertain, must explicitly mark "Unknown" and ask user for missing info

## Acceptance criteria
- Running `make eval` processes all files in `data/samples/` end-to-end
- All `case_facts.json` and `route.json` validate against schemas
- CI runs: lint, typecheck, unit tests, eval
