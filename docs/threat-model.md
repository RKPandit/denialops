# DenialOps — Threat Model

## Overview

DenialOps processes sensitive health and financial information. This document identifies risks and mitigations.

---

## Data Classification

| Data Type | Sensitivity | Examples |
|-----------|-------------|----------|
| PHI (Protected Health Information) | High | Diagnosis codes, treatment descriptions, dates of service |
| PII (Personally Identifiable Information) | High | Names, member IDs, addresses, SSN (if present) |
| Financial | Medium | Billed amounts, payment info |
| Plan/Policy | Low | Coverage rules, plan names |
| System/Logs | Medium | Request traces, errors |

---

## Threat Categories

### T1: PHI/PII Exposure

#### T1.1 Logging PHI
**Risk:** PHI written to logs, exposing it to unauthorized access.

**Mitigations:**
- [ ] Implement log sanitization that redacts PHI patterns before writing
- [ ] Never log full document content
- [ ] Never log extracted `case_facts` without redaction
- [ ] Use structured logging with explicit allow-list of fields

#### T1.2 LLM Provider Data Retention
**Risk:** PHI sent to LLM provider may be retained for training.

**Mitigations:**
- [ ] Use Anthropic API with zero data retention agreement
- [ ] Document data handling in user-facing terms
- [ ] Consider on-premise/self-hosted LLM for enterprise

#### T1.3 Artifact Storage
**Risk:** Artifacts stored on disk may be accessed by unauthorized users.

**Mitigations:**
- [ ] Artifacts stored in user-specific directories
- [ ] Implement artifact expiration (delete after N days)
- [ ] Encrypt artifacts at rest (future)
- [ ] Clear artifacts on session end (optional)

### T2: Unauthorized Access

#### T2.1 API Access
**Risk:** Unauthorized users access other users' cases.

**Mitigations:**
- [ ] Case IDs are UUIDs (not guessable)
- [ ] MVP: Single-user local deployment (no auth needed)
- [ ] Future: Add authentication layer before multi-user deployment

#### T2.2 Path Traversal
**Risk:** Malicious filenames allow reading/writing outside artifact directory.

**Mitigations:**
- [ ] Validate and sanitize all filenames
- [ ] Use `pathlib` with strict path resolution
- [ ] Reject filenames containing `..`, `/`, `\`
- [ ] Store files with system-generated names, not user-provided

### T3: Injection Attacks

#### T3.1 Prompt Injection
**Risk:** Malicious content in denial letters manipulates LLM behavior.

**Mitigations:**
- [ ] Treat all document content as untrusted data
- [ ] Use clear delimiters in prompts (`<document>...</document>`)
- [ ] Validate LLM outputs against schemas (rejects unexpected structure)
- [ ] Implement output sanitization

#### T3.2 PDF Exploits
**Risk:** Malformed PDFs exploit parser vulnerabilities.

**Mitigations:**
- [ ] Use well-maintained PDF library (pdfminer.six)
- [ ] Run PDF parsing in sandboxed process (future)
- [ ] Limit PDF size (reject >10MB)
- [ ] Timeout PDF parsing (reject if >30s)

### T4: Output Integrity

#### T4.1 Hallucination
**Risk:** System generates false information that users act on.

**Mitigations:**
- [ ] Grounding checks (see eval.md)
- [ ] All codes/dates must trace to source
- [ ] Unknown fields explicitly marked
- [ ] Disclaimers on all generated documents

#### T4.2 Harmful Advice
**Risk:** System provides incorrect guidance causing financial/health harm.

**Mitigations:**
- [ ] Explicit disclaimers: "Not legal advice", "Not medical advice"
- [ ] Never advise skipping deadlines
- [ ] Encourage professional consultation for complex cases
- [ ] Conservative confidence scoring

### T5: Legal & Compliance

#### T5.1 Unauthorized Practice
**Risk:** System could be construed as practicing law or medicine.

**Mitigations:**
- [ ] Clear disclaimers on every output
- [ ] Position as "information tool" not "advisor"
- [ ] Don't guarantee outcomes
- [ ] Recommend professional review

#### T5.2 HIPAA (if applicable)
**Risk:** Non-compliant handling of PHI.

**Mitigations:**
- [ ] MVP: Local-only deployment (user controls their own PHI)
- [ ] No PHI transmitted to external services except LLM (with DPA)
- [ ] Future: BAA with LLM provider for enterprise
- [ ] Audit logging for PHI access

### T6: Scraping & External Access

#### T6.1 Portal Scraping
**Risk:** System attempts to access insurance portals, violating ToS.

**Mitigations:**
- [ ] **Explicit policy: No portal scraping**
- [ ] No authentication credential handling
- [ ] Only process user-uploaded documents
- [ ] Document this limitation clearly

#### T6.2 External URL Fetching
**Risk:** System fetches URLs that could be malicious or violate policies.

**Mitigations:**
- [ ] MVP: No URL fetching
- [ ] Future: Allowlist of public policy URLs only
- [ ] Never fetch user-provided URLs

---

## Security Controls

### Input Validation
```python
# File upload validation
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
ALLOWED_DOC_TYPES = {"denial_letter", "eob", "sbc", "eoc"}

def validate_upload(file: UploadFile, doc_type: str) -> None:
    if doc_type not in ALLOWED_DOC_TYPES:
        raise ValueError(f"Invalid doc_type: {doc_type}")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Invalid file extension: {ext}")

    # Check file size
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_FILE_SIZE:
        raise ValueError(f"File too large: {size} bytes")
```

### Path Sanitization
```python
def safe_path(base: Path, filename: str) -> Path:
    """Ensure path stays within base directory."""
    # Remove any path components
    safe_name = Path(filename).name

    # Reject suspicious patterns
    if ".." in safe_name or safe_name.startswith("."):
        raise ValueError(f"Invalid filename: {filename}")

    full_path = (base / safe_name).resolve()

    # Verify still within base
    if not str(full_path).startswith(str(base.resolve())):
        raise ValueError(f"Path traversal detected: {filename}")

    return full_path
```

### Log Redaction
```python
import re

PHI_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),  # SSN
    (r"\b[A-Z]{3}\d{9}\b", "[MEMBER_ID]"),  # Member ID pattern
    (r"\b\d{10}\b", "[PHONE/NPI]"),  # Phone or NPI
]

def redact_phi(text: str) -> str:
    """Redact PHI patterns from text for logging."""
    for pattern, replacement in PHI_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text
```

---

## Required Disclaimers

All generated documents must include:

```markdown
---
**IMPORTANT DISCLAIMERS**

This document was generated by an automated system and is provided for
informational purposes only.

- This is NOT legal advice. For legal questions, consult a licensed attorney.
- This is NOT medical advice. For medical questions, consult your healthcare provider.
- Coverage details may vary. Verify all information with your insurance company.
- Deadlines are estimates. Confirm actual deadlines with your insurer.

The creators of this tool are not responsible for decisions made based on
this information.
---
```

---

## Incident Response

### PHI Exposure
1. Immediately disable affected logging/storage
2. Identify scope of exposure
3. Purge exposed data
4. Document incident
5. Notify affected users (if identifiable)

### Hallucination Discovered
1. Add case to eval golden set
2. Investigate root cause
3. Update prompts/validation
4. Re-run eval to verify fix
5. Document in changelog

---

## Compliance Checklist

### MVP (Local Deployment)
- [ ] No PHI in logs
- [ ] Disclaimers on all outputs
- [ ] No external scraping
- [ ] Path traversal prevention
- [ ] Input validation
- [ ] Grounding checks passing

### Future (Multi-User)
- [ ] Authentication
- [ ] Authorization (user can only access own cases)
- [ ] Encryption at rest
- [ ] Audit logging
- [ ] BAA with LLM provider
- [ ] Penetration testing

---

## Appendix: PHI Fields Reference

Fields that may contain PHI and require protection:

| Schema | Field | PHI Type |
|--------|-------|----------|
| case_facts | payer.member_id | Identifier |
| case_facts | payer.group_number | Identifier |
| case_facts | service.provider_name | Name |
| case_facts | service.provider_npi | Identifier |
| case_facts | service.diagnosis_codes | Health info |
| case_facts | denial_reason | Health info |
| case_facts | contact_info.* | Contact |
| plan_rules | plan_info.* | Plan identifier |

---

## Review Schedule

- [ ] Review threat model quarterly
- [ ] Update after any security incident
- [ ] Update when adding new features that handle PHI
- [ ] External security review before production deployment
