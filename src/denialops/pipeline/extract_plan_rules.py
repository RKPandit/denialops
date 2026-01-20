"""Plan rules extraction from SBC/EOC documents."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from denialops.llm import create_llm_client
from denialops.models.documents import ExtractedText
from denialops.models.plan_rules import (
    AppealRights,
    Deductibles,
    Exclusion,
    ExtractionQuality,
    MedicalNecessityCriteria,
    OutOfPocketMax,
    PlanDocumentType,
    PlanInfo,
    PlanRules,
    PlanType,
    PriorAuthRule,
    TimelyFiling,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Prompts for SBC Extraction
# =============================================================================

EXTRACT_PLAN_RULES_SYSTEM = """You are an expert at analyzing health insurance plan documents including Summary of Benefits and Coverage (SBC), Evidence of Coverage (EOC), and Certificate of Coverage (COC) documents.

Extract structured information accurately. For each piece of information, note the page number or section where you found it to enable citations.

Always respond with valid JSON. If information is not present, use null."""

EXTRACT_PLAN_RULES_USER = """Analyze this insurance plan document and extract the following information as JSON:

<document>
{text}
</document>

Extract and return a JSON object with these fields:
{{
  "plan_info": {{
    "plan_name": "Name of the plan or null",
    "plan_year": "YYYY or null",
    "payer_name": "Insurance company name",
    "plan_type": "HMO|PPO|EPO|POS|HDHP|Medicare|Medicaid|other",
    "state": "Two-letter state code or null",
    "effective_date": "YYYY-MM-DD or null",
    "termination_date": "YYYY-MM-DD or null"
  }},
  "deductibles": {{
    "individual_in_network": 0.00,
    "individual_out_of_network": 0.00,
    "family_in_network": 0.00,
    "family_out_of_network": 0.00,
    "source_page": 1
  }},
  "out_of_pocket_max": {{
    "individual_in_network": 0.00,
    "individual_out_of_network": 0.00,
    "family_in_network": 0.00,
    "family_out_of_network": 0.00,
    "source_page": 1
  }},
  "prior_authorization_rules": [
    {{
      "service_category": "Category like Imaging, Surgery, DME",
      "pa_required": true,
      "conditions": "When PA is required",
      "source_page": 1,
      "source_section": "Section name",
      "source_quote": "Direct quote from document"
    }}
  ],
  "medical_necessity_criteria": [
    {{
      "definition": "How plan defines medical necessity",
      "criteria": ["Criterion 1", "Criterion 2"],
      "source_page": 1,
      "source_section": "Section name",
      "source_quote": "Direct quote"
    }}
  ],
  "exclusions": [
    {{
      "exclusion": "What is excluded",
      "category": "cosmetic|experimental|other",
      "exceptions": "Any exceptions",
      "source_page": 1,
      "source_section": "Section name",
      "source_quote": "Direct quote"
    }}
  ],
  "appeal_rights": {{
    "internal_appeal_levels": 2,
    "internal_appeal_deadline_days": 180,
    "expedited_appeal_available": true,
    "expedited_criteria": "When available",
    "external_review_available": true,
    "external_review_deadline_days": 60,
    "appeal_address": "Mailing address",
    "appeal_phone": "Phone number",
    "appeal_fax": "Fax number",
    "source_page": 1,
    "source_section": "Appeals section"
  }},
  "timely_filing": {{
    "initial_claim_days": 365,
    "corrected_claim_days": 365,
    "source_page": 1
  }}
}}

Important:
- Extract dollar amounts as numbers without $ signs
- Include source_page for every section where you found the information
- Include direct quotes where policy language is important (medical necessity, exclusions)
- If a section is not found in the document, set it to null

Return ONLY valid JSON, no other text."""


# =============================================================================
# Main Extraction Function
# =============================================================================


def extract_plan_rules(
    case_id: str,
    text: ExtractedText,
    llm_api_key: str = "",
    llm_model: str = "",
    llm_provider: str = "openai",
) -> PlanRules:
    """
    Extract plan rules from SBC/EOC document.

    Args:
        case_id: Case identifier
        text: Extracted text from document
        llm_api_key: API key for LLM calls
        llm_model: Model to use for LLM calls
        llm_provider: LLM provider ("openai" or "anthropic")

    Returns:
        PlanRules with extracted information
    """
    content = text.full_text
    document_id = text.document_id

    # Try LLM extraction if API key provided
    if llm_api_key:
        try:
            return _extract_with_llm(
                case_id=case_id,
                document_id=document_id,
                content=content,
                llm_api_key=llm_api_key,
                llm_model=llm_model,
                llm_provider=llm_provider,
            )
        except Exception as e:
            logger.warning(f"LLM extraction failed, falling back to heuristics: {e}")

    # Fallback to heuristic extraction
    return _extract_with_heuristics(case_id, document_id, content)


def _extract_with_llm(
    case_id: str,
    document_id: str,
    content: str,
    llm_api_key: str,
    llm_model: str,
    llm_provider: str,
) -> PlanRules:
    """Extract plan rules using LLM."""
    logger.info(f"Extracting plan rules using LLM ({llm_provider}/{llm_model})")

    # Create LLM client
    client = create_llm_client(
        provider=llm_provider,
        api_key=llm_api_key,
        model=llm_model if llm_model else None,
    )

    # Call LLM for extraction
    prompt = EXTRACT_PLAN_RULES_USER.format(text=content[:12000])  # Limit to 12k chars
    response = client.complete(
        prompt=prompt,
        system=EXTRACT_PLAN_RULES_SYSTEM,
        max_tokens=3000,
        temperature=0.0,
    )

    # Parse JSON response
    llm_data = _parse_llm_response(response)

    # Build PlanRules from LLM response
    return _build_plan_rules_from_llm(case_id, document_id, llm_data, content)


def _parse_llm_response(response: str) -> dict[str, Any]:
    """Parse JSON from LLM response."""
    # Try to extract JSON from markdown code blocks
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


def _build_plan_rules_from_llm(
    case_id: str,
    document_id: str,
    llm_data: dict[str, Any],
    content: str,
) -> PlanRules:
    """Build PlanRules from LLM extracted data."""

    # Parse plan info
    plan_info_data = llm_data.get("plan_info", {})
    plan_type_str = plan_info_data.get("plan_type", "other")
    try:
        plan_type = PlanType(plan_type_str)
    except ValueError:
        plan_type = PlanType.OTHER

    plan_info = PlanInfo(
        plan_name=plan_info_data.get("plan_name"),
        plan_year=plan_info_data.get("plan_year"),
        payer_name=plan_info_data.get("payer_name"),
        plan_type=plan_type,
        state=plan_info_data.get("state"),
        effective_date=_parse_date(plan_info_data.get("effective_date")),
        termination_date=_parse_date(plan_info_data.get("termination_date")),
    )

    # Parse deductibles
    deductibles_data = llm_data.get("deductibles")
    deductibles = None
    if deductibles_data:
        deductibles = Deductibles(
            individual_in_network=deductibles_data.get("individual_in_network"),
            individual_out_of_network=deductibles_data.get("individual_out_of_network"),
            family_in_network=deductibles_data.get("family_in_network"),
            family_out_of_network=deductibles_data.get("family_out_of_network"),
            source_page=deductibles_data.get("source_page"),
        )

    # Parse out of pocket max
    oop_data = llm_data.get("out_of_pocket_max")
    out_of_pocket_max = None
    if oop_data:
        out_of_pocket_max = OutOfPocketMax(
            individual_in_network=oop_data.get("individual_in_network"),
            individual_out_of_network=oop_data.get("individual_out_of_network"),
            family_in_network=oop_data.get("family_in_network"),
            family_out_of_network=oop_data.get("family_out_of_network"),
            source_page=oop_data.get("source_page"),
        )

    # Parse prior auth rules
    pa_rules = []
    for rule_data in llm_data.get("prior_authorization_rules", []):
        pa_rules.append(
            PriorAuthRule(
                service_category=rule_data.get("service_category", "Unknown"),
                pa_required=rule_data.get("pa_required", False),
                conditions=rule_data.get("conditions"),
                source_page=rule_data.get("source_page"),
                source_section=rule_data.get("source_section"),
                source_quote=rule_data.get("source_quote"),
            )
        )

    # Parse medical necessity criteria
    mn_criteria = []
    for criteria_data in llm_data.get("medical_necessity_criteria", []):
        mn_criteria.append(
            MedicalNecessityCriteria(
                definition=criteria_data.get("definition"),
                criteria=criteria_data.get("criteria", []),
                source_page=criteria_data.get("source_page"),
                source_section=criteria_data.get("source_section"),
                source_quote=criteria_data.get("source_quote"),
            )
        )

    # Parse exclusions
    exclusions = []
    for excl_data in llm_data.get("exclusions", []):
        exclusions.append(
            Exclusion(
                exclusion=excl_data.get("exclusion", "Unknown"),
                category=excl_data.get("category"),
                exceptions=excl_data.get("exceptions"),
                source_page=excl_data.get("source_page"),
                source_section=excl_data.get("source_section"),
                source_quote=excl_data.get("source_quote"),
            )
        )

    # Parse appeal rights
    appeal_data = llm_data.get("appeal_rights")
    appeal_rights = None
    if appeal_data:
        appeal_rights = AppealRights(
            internal_appeal_levels=appeal_data.get("internal_appeal_levels"),
            internal_appeal_deadline_days=appeal_data.get("internal_appeal_deadline_days"),
            expedited_appeal_available=appeal_data.get("expedited_appeal_available"),
            expedited_criteria=appeal_data.get("expedited_criteria"),
            external_review_available=appeal_data.get("external_review_available"),
            external_review_deadline_days=appeal_data.get("external_review_deadline_days"),
            appeal_address=appeal_data.get("appeal_address"),
            appeal_phone=appeal_data.get("appeal_phone"),
            appeal_fax=appeal_data.get("appeal_fax"),
            source_page=appeal_data.get("source_page"),
            source_section=appeal_data.get("source_section"),
        )

    # Parse timely filing
    tf_data = llm_data.get("timely_filing")
    timely_filing = None
    if tf_data:
        timely_filing = TimelyFiling(
            initial_claim_days=tf_data.get("initial_claim_days"),
            corrected_claim_days=tf_data.get("corrected_claim_days"),
            source_page=tf_data.get("source_page"),
        )

    # Calculate extraction quality
    sections_found = sum(
        [
            1 if deductibles else 0,
            1 if out_of_pocket_max else 0,
            1 if pa_rules else 0,
            1 if mn_criteria else 0,
            1 if exclusions else 0,
            1 if appeal_rights else 0,
            1 if timely_filing else 0,
        ]
    )

    extraction_quality = ExtractionQuality(
        confidence=min(0.9, 0.5 + (sections_found * 0.1)),
        pages_processed=len(content) // 3000,  # Rough estimate
        sections_identified=sections_found,
        warnings=[],
    )

    return PlanRules(
        case_id=case_id,
        source_document=document_id,
        document_type=PlanDocumentType.SBC,
        extracted_at=datetime.now(timezone.utc),
        plan_info=plan_info,
        deductibles=deductibles,
        out_of_pocket_max=out_of_pocket_max,
        prior_authorization_rules=pa_rules,
        medical_necessity_criteria=mn_criteria,
        exclusions=exclusions,
        appeal_rights=appeal_rights,
        timely_filing=timely_filing,
        extraction_quality=extraction_quality,
    )


def _parse_date(date_str: str | None):
    """Parse a date string."""
    if not date_str:
        return None
    try:
        from datetime import datetime

        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None


def _extract_with_heuristics(
    case_id: str,
    document_id: str,
    content: str,
) -> PlanRules:
    """Extract plan rules using heuristic/regex methods (fallback)."""
    import re

    logger.info("Extracting plan rules using heuristics")

    # Extract deductibles
    deductible_pattern = r"\$([0-9,]+)\s*(?:individual|single)?\s*(?:in-network)?\s*deductible"
    deductible_match = re.search(deductible_pattern, content, re.IGNORECASE)
    individual_deductible = None
    if deductible_match:
        individual_deductible = float(deductible_match.group(1).replace(",", ""))

    deductibles = Deductibles(individual_in_network=individual_deductible)

    # Extract OOP max
    oop_pattern = r"\$([0-9,]+)\s*(?:individual|single)?\s*(?:out-of-pocket|oop)\s*(?:max|maximum|limit)"
    oop_match = re.search(oop_pattern, content, re.IGNORECASE)
    individual_oop = None
    if oop_match:
        individual_oop = float(oop_match.group(1).replace(",", ""))

    out_of_pocket_max = OutOfPocketMax(individual_in_network=individual_oop)

    # Extract appeal deadline - matches multiple formats:
    # "180 days to file appeal", "within 180 days", "180 calendar days of the denial"
    appeal_patterns = [
        r"(\d+)\s*(?:calendar\s*)?days?\s*(?:to|for)\s*(?:file|submit)?\s*(?:an?\s*)?(?:internal\s*)?appeal",
        r"within\s*(\d+)\s*(?:calendar\s*)?days?\s*(?:of|from)",
        r"(\d+)\s*(?:calendar\s*)?days?\s*(?:of|from)\s*(?:the\s*)?(?:denial|notice)",
    ]
    appeal_match = None
    for pattern in appeal_patterns:
        appeal_match = re.search(pattern, content, re.IGNORECASE)
        if appeal_match:
            break
    appeal_deadline = None
    if appeal_match:
        appeal_deadline = int(appeal_match.group(1))

    appeal_rights = AppealRights(internal_appeal_deadline_days=appeal_deadline)

    # Detect if prior auth is mentioned
    pa_rules = []
    if re.search(r"prior\s*(?:authorization|approval)", content, re.IGNORECASE):
        pa_rules.append(
            PriorAuthRule(
                service_category="Various services",
                pa_required=True,
                conditions="See plan document for specific services",
            )
        )

    return PlanRules(
        case_id=case_id,
        source_document=document_id,
        document_type=PlanDocumentType.SBC,
        extracted_at=datetime.now(timezone.utc),
        plan_info=PlanInfo(),
        deductibles=deductibles,
        out_of_pocket_max=out_of_pocket_max,
        prior_authorization_rules=pa_rules,
        appeal_rights=appeal_rights,
        extraction_quality=ExtractionQuality(
            confidence=0.4,
            warnings=["Extracted using heuristics only - limited accuracy"],
        ),
    )
