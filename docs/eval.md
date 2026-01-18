# DenialOps — Evaluation Spec

## Purpose

This document defines how we measure quality and prevent "impressive but wrong" outputs. All evaluation criteria must pass before shipping.

---

## Evaluation Dimensions

### 1. Extraction Quality
How well does the system extract facts from denial letters?

### 2. Routing Accuracy
Does the system select the correct action route?

### 3. Generation Grounding
Are generated outputs faithful to extracted facts (no hallucination)?

### 4. Citation Coverage (Verified mode)
Are policy claims properly cited?

### 5. Actionability
Are outputs useful and complete for the target user?

---

## Golden Test Cases

### Case Manifest
File: `eval/cases/case_manifest.json`

```json
{
  "cases": [
    {
      "case_id": "pa_missing_001",
      "description": "Prior auth not obtained for MRI",
      "input_file": "data/samples/pa_missing_001.txt",
      "expected_route": "prior_auth_needed",
      "golden_facts": "eval/cases/pa_missing_001_facts.json",
      "mode": "fast"
    },
    {
      "case_id": "coding_error_001",
      "description": "Wrong modifier on surgical claim",
      "input_file": "data/samples/coding_error_001.txt",
      "expected_route": "claim_correction_resubmit",
      "golden_facts": "eval/cases/coding_error_001_facts.json",
      "mode": "fast"
    },
    {
      "case_id": "med_nec_001",
      "description": "Medical necessity denial for biologics",
      "input_file": "data/samples/med_nec_001.txt",
      "sbc_file": "data/samples/med_nec_001_sbc.txt",
      "expected_route": "medical_necessity_appeal",
      "golden_facts": "eval/cases/med_nec_001_facts.json",
      "golden_plan_rules": "eval/cases/med_nec_001_plan_rules.json",
      "mode": "verified"
    }
  ]
}
```

### Minimum Case Coverage
- 2 cases per route (6 total minimum)
- At least 2 Verified mode cases
- At least 1 case with ambiguous signals (tests confidence scoring)
- At least 1 case with missing information (tests graceful degradation)

---

## Extraction Metrics

### Required Fields Coverage
Every case must extract these fields (if present in source):

| Field | Required | Notes |
|-------|----------|-------|
| `denial_reason` | Yes | Must always extract |
| `dates.date_of_denial` | Yes | If present |
| `dates.appeal_deadline` | Yes | If present |
| `denial_codes` | Yes | All codes in letter |
| `payer.name` | Yes | If present |
| `contact_info.phone` | No | Best effort |
| `amounts.billed_amount` | No | Best effort |

### Extraction Scoring

```python
def score_extraction(extracted: CaseFacts, golden: CaseFacts) -> ExtractionScore:
    """
    Returns:
      - field_coverage: % of golden fields that are present in extracted
      - field_accuracy: % of extracted fields that match golden (exact or fuzzy)
      - code_recall: % of golden codes found in extracted
      - code_precision: % of extracted codes that are in golden
      - date_accuracy: % of dates within 1 day of golden
    """
```

### Thresholds
| Metric | Minimum | Target |
|--------|---------|--------|
| Field coverage | 80% | 95% |
| Field accuracy | 85% | 95% |
| Code recall | 90% | 98% |
| Code precision | 90% | 98% |
| Date accuracy | 95% | 100% |

---

## Routing Metrics

### Accuracy Calculation
```python
def score_routing(predicted: RouteDecision, expected_route: str) -> RoutingScore:
    """
    Returns:
      - correct: bool (predicted.route == expected_route)
      - confidence_calibrated: bool (high confidence when correct, low when wrong)
      - signals_present: list of signals detected
    """
```

### Thresholds
| Metric | Minimum | Target |
|--------|---------|--------|
| Route accuracy | 85% | 95% |
| High-confidence accuracy | 95% | 99% |

### Confidence Calibration
- When `confidence >= 0.8`: accuracy must be ≥95%
- When `confidence < 0.5`: system should flag for human review

---

## Grounding Checks (Hallucination Prevention)

### Rules
1. **No invented codes**: Every code in output must appear in `case_facts.denial_codes` or `case_facts.service.cpt_codes`
2. **No invented dates**: Every date in output must be derived from `case_facts.dates` or calculated from them
3. **No invented amounts**: Every dollar amount must come from `case_facts.amounts`
4. **No invented contact info**: Contact details must come from `case_facts.contact_info`
5. **Unknown is explicit**: If data is missing, output must say "Unknown" or "Not provided"

### Grounding Score
```python
def score_grounding(action_plan: ActionPlan, case_facts: CaseFacts) -> GroundingScore:
    """
    Returns:
      - grounded_codes: % of codes in plan that exist in facts
      - grounded_dates: % of dates in plan that are derivable from facts
      - unknown_marked: % of missing fields properly marked as unknown
      - hallucination_count: number of invented facts (should be 0)
    """
```

### Thresholds
| Metric | Minimum | Target |
|--------|---------|--------|
| Grounded codes | 100% | 100% |
| Grounded dates | 100% | 100% |
| Unknown marked | 100% | 100% |
| Hallucination count | 0 | 0 |

**Hallucination count > 0 is a blocking failure.**

---

## Citation Coverage (Verified Mode)

### Rules
1. Every policy claim in the action plan must have a citation
2. Citations must reference `plan_rules` with page/section
3. Claims without supporting `plan_rules` must be marked as "assumption"

### Citation Score
```python
def score_citations(action_plan: ActionPlan, plan_rules: PlanRules) -> CitationScore:
    """
    Returns:
      - citation_coverage: % of policy claims with citations
      - citation_accuracy: % of citations that correctly reference plan_rules
      - assumption_marked: % of ungrounded claims marked as assumptions
    """
```

### Thresholds
| Metric | Minimum | Target |
|--------|---------|--------|
| Citation coverage | 80% | 95% |
| Citation accuracy | 95% | 100% |
| Assumption marked | 100% | 100% |

---

## Actionability Score (Human Eval)

For a subset of cases, human evaluators score:

| Dimension | Scale | Description |
|-----------|-------|-------------|
| Clarity | 1-5 | Is the explanation understandable? |
| Completeness | 1-5 | Are all necessary steps included? |
| Accuracy | 1-5 | Is the guidance correct? |
| Usefulness | 1-5 | Would this help the user? |

### Thresholds
| Metric | Minimum | Target |
|--------|---------|--------|
| Average score | 3.5 | 4.5 |
| No score < 3 | Required | Required |

---

## Eval Harness

### Usage
```bash
# Run full eval
make eval

# Run specific case
python eval/run_eval.py --case pa_missing_001

# Run only extraction eval
python eval/run_eval.py --stage extraction

# Generate report
python eval/run_eval.py --report
```

### Output
File: `eval/reports/eval_YYYYMMDD_HHMMSS.json`

```json
{
  "run_id": "20240115_143022",
  "timestamp": "2024-01-15T14:30:22Z",
  "cases_evaluated": 10,
  "overall_pass": true,
  "extraction": {
    "field_coverage": 0.92,
    "field_accuracy": 0.89,
    "code_recall": 0.95,
    "code_precision": 0.93,
    "pass": true
  },
  "routing": {
    "accuracy": 0.90,
    "high_confidence_accuracy": 0.96,
    "pass": true
  },
  "grounding": {
    "hallucination_count": 0,
    "grounded_codes": 1.0,
    "grounded_dates": 1.0,
    "pass": true
  },
  "citations": {
    "coverage": 0.85,
    "accuracy": 0.98,
    "pass": true
  },
  "case_results": [
    {
      "case_id": "pa_missing_001",
      "pass": true,
      "extraction_score": {...},
      "routing_correct": true,
      "grounding_score": {...}
    }
  ],
  "failures": [],
  "warnings": ["Case med_nec_002 has low confidence (0.45)"]
}
```

---

## CI Integration

### Eval in CI
```yaml
# .github/workflows/ci.yml
eval:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Run eval
      run: make eval
    - name: Check pass
      run: |
        if grep -q '"overall_pass": false' eval/reports/latest.json; then
          echo "Eval failed"
          exit 1
        fi
    - name: Upload report
      uses: actions/upload-artifact@v4
      with:
        name: eval-report
        path: eval/reports/
```

### Blocking Failures
These failures block merge:
- Hallucination count > 0
- Route accuracy < 85%
- Extraction coverage < 80%
- Any golden case that previously passed now fails (regression)

### Warnings (non-blocking)
- Low confidence on any case
- Missing optional fields
- Citation coverage < 90%

---

## Regression Testing

### Golden Case Lock
Once a case passes, it becomes a regression test:
1. Save the successful output
2. Future runs must match or exceed the score
3. Regressions require explicit approval to update golden

### Version Tracking
Track eval scores across versions:
```
eval/history/
├── v0.1.0_eval.json
├── v0.2.0_eval.json
└── latest_eval.json -> v0.2.0_eval.json
```

---

## Creating New Test Cases

### Synthetic Denial Letter Template
```
[PAYER LETTERHEAD]

Date: [DATE]
Member ID: [MEMBER_ID]
Claim Number: [CLAIM_NUMBER]

Dear [MEMBER_NAME],

We have reviewed your claim for [SERVICE_DESCRIPTION] performed on [DATE_OF_SERVICE]
by [PROVIDER_NAME].

YOUR CLAIM HAS BEEN DENIED

Reason: [DENIAL_REASON]
Denial Code: [CODE]

[ADDITIONAL_DETAILS]

You have the right to appeal this decision within [DAYS] days.

To file an appeal:
Phone: [PHONE]
Fax: [FAX]
Address: [ADDRESS]

Sincerely,
[PAYER_NAME] Claims Department
```

### Golden Facts Template
Create corresponding `*_facts.json` with:
- All extractable fields populated
- Confidence scores set to 1.0 (golden = ground truth)
- `missing_info` populated for any intentionally missing data

---

## Appendix: Code Validation Patterns

### CPT Code
```regex
\b[0-9]{5}\b
```

### ICD-10-CM
```regex
\b[A-Z][0-9]{2}\.?[0-9A-Z]{0,4}\b
```

### HCPCS
```regex
\b[A-Z][0-9]{4}\b
```

### CARC (Claim Adjustment Reason Code)
```regex
\b(CO|PR|OA|PI|CR)-?[0-9]{1,3}\b
```

### Dollar Amount
```regex
\$[0-9,]+\.?[0-9]{0,2}
```

### Date (various formats)
```regex
\b(0?[1-9]|1[0-2])[-/](0?[1-9]|[12][0-9]|3[01])[-/]([0-9]{2}|[0-9]{4})\b
\b(January|February|...|December)\s+[0-9]{1,2},?\s+[0-9]{4}\b
```
