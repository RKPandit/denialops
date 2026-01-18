"""Case facts extraction from denial letter text."""

import json
import logging
import re
from datetime import date, datetime, timezone
from typing import Any

from denialops.llm import create_llm_client
from denialops.models.case import (
    CaseAmounts,
    CaseDates,
    CaseFacts,
    CodeType,
    ContactInfo,
    DenialCode,
    ExtractionConfidence,
    MissingInfo,
    PayerInfo,
    PlanType,
    RawTextSnippets,
    ServiceInfo,
)
from denialops.models.documents import ExtractedText

logger = logging.getLogger(__name__)

# LLM extraction prompt for denial letters
EXTRACTION_SYSTEM_PROMPT = """You are an expert at analyzing health insurance denial letters.
Extract structured information from the denial letter text provided.
Always respond with valid JSON. Be precise and accurate.
If information is not present in the letter, use null for that field."""

EXTRACTION_USER_PROMPT = """Analyze this insurance denial letter and extract the following information as JSON:

<denial_letter>
{text}
</denial_letter>

Extract and return a JSON object with these fields:
{{
  "denial_reason": "The full denial reason as stated in the letter",
  "denial_reason_summary": "A short 5-10 word summary of why the claim was denied",
  "denial_codes": [
    {{"code": "string", "code_type": "CPT|HCPCS|ICD_10_CM|ICD_10_PCS|CARC|RARC", "description": "optional description"}}
  ],
  "service": {{
    "description": "Description of the service that was denied",
    "cpt_codes": ["list of CPT codes for the service"],
    "diagnosis_codes": ["list of ICD-10 diagnosis codes"],
    "date_of_service": "YYYY-MM-DD or null",
    "provider_name": "Name of the provider",
    "facility_name": "Name of the facility"
  }},
  "payer": {{
    "name": "Insurance company name",
    "plan_name": "Specific plan name if mentioned",
    "plan_type": "employer|marketplace|medicare|medicaid|individual|unknown",
    "member_id": "Member/subscriber ID",
    "group_number": "Group number if mentioned",
    "claim_number": "Claim or reference number"
  }},
  "dates": {{
    "date_of_service": "YYYY-MM-DD or null",
    "date_of_denial": "YYYY-MM-DD - the date on the letter",
    "appeal_deadline_days": "Number of days allowed to appeal (integer)",
    "timely_filing_deadline": "YYYY-MM-DD or null"
  }},
  "amounts": {{
    "billed_amount": 0.00,
    "allowed_amount": 0.00,
    "paid_amount": 0.00,
    "patient_responsibility": 0.00
  }},
  "contact_info": {{
    "phone": "Phone number for appeals",
    "fax": "Fax number for appeals",
    "address": "Mailing address for appeals",
    "website": "Website if mentioned"
  }},
  "prior_auth_mentioned": true/false,
  "medical_necessity_mentioned": true/false,
  "coding_issue_mentioned": true/false,
  "timely_filing_mentioned": true/false
}}

Important:
- Only extract CPT codes that are explicitly procedure codes (5 digits like 72148), not zip codes or other numbers
- Distinguish between the denial date (letter date) and the date of service
- Extract the appeal deadline in days if mentioned (e.g., "within 180 days")
- For amounts, extract numeric values only (no $ signs in the output)

Return ONLY valid JSON, no other text."""


def extract_case_facts(
    case_id: str,
    text: ExtractedText,
    user_context: dict[str, Any] | None = None,
    llm_api_key: str = "",
    llm_model: str = "",
    llm_provider: str = "openai",
) -> CaseFacts:
    """
    Extract structured facts from denial letter text.

    Uses LLM extraction when API key is provided, with regex fallback for robustness.

    Args:
        case_id: Case identifier
        text: Extracted text from document
        user_context: Optional user-provided context
        llm_api_key: API key for LLM calls
        llm_model: Model to use for LLM calls
        llm_provider: LLM provider ("openai" or "anthropic")

    Returns:
        CaseFacts with extracted information
    """
    content = text.full_text
    user_context = user_context or {}

    # Try LLM extraction first if API key provided
    if llm_api_key:
        try:
            return _extract_with_llm(
                case_id=case_id,
                text=text,
                content=content,
                user_context=user_context,
                llm_api_key=llm_api_key,
                llm_model=llm_model,
                llm_provider=llm_provider,
            )
        except Exception as e:
            logger.warning(f"LLM extraction failed, falling back to heuristics: {e}")

    # Fallback to heuristic extraction
    return _extract_with_heuristics(case_id, text, content, user_context)


def _extract_with_llm(
    case_id: str,
    text: ExtractedText,
    content: str,
    user_context: dict[str, Any],
    llm_api_key: str,
    llm_model: str,
    llm_provider: str,
) -> CaseFacts:
    """Extract case facts using LLM."""
    logger.info(f"Extracting case facts using LLM ({llm_provider}/{llm_model})")

    # Create LLM client
    client = create_llm_client(
        provider=llm_provider,
        api_key=llm_api_key,
        model=llm_model if llm_model else None,
    )

    # Call LLM for extraction
    prompt = EXTRACTION_USER_PROMPT.format(text=content[:8000])  # Limit to 8k chars
    response = client.complete(
        prompt=prompt,
        system=EXTRACTION_SYSTEM_PROMPT,
        max_tokens=2000,
        temperature=0.0,
    )

    # Parse JSON response
    llm_data = _parse_llm_response(response)

    # Build CaseFacts from LLM response
    return _build_case_facts_from_llm(
        case_id=case_id,
        text=text,
        llm_data=llm_data,
        user_context=user_context,
    )


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


def _build_case_facts_from_llm(
    case_id: str,
    text: ExtractedText,
    llm_data: dict[str, Any],
    user_context: dict[str, Any],
) -> CaseFacts:
    """Build CaseFacts from LLM extracted data."""
    # Parse denial codes
    codes = []
    for code_data in llm_data.get("denial_codes", []):
        code_type_str = code_data.get("code_type", "").upper().replace("-", "_")
        try:
            code_type = CodeType(code_type_str)
        except ValueError:
            code_type = CodeType.CPT  # Default
        codes.append(
            DenialCode(
                code=code_data.get("code", ""),
                code_type=code_type,
                description=code_data.get("description"),
            )
        )

    # Parse dates
    dates_data = llm_data.get("dates", {})
    dates = CaseDates(
        date_of_service=_parse_date(dates_data.get("date_of_service")),
        date_of_denial=_parse_date(dates_data.get("date_of_denial")),
        appeal_deadline_days=dates_data.get("appeal_deadline_days"),
        timely_filing_deadline=_parse_date(dates_data.get("timely_filing_deadline")),
    )

    # Parse amounts
    amounts_data = llm_data.get("amounts", {})
    amounts = CaseAmounts(
        billed_amount=amounts_data.get("billed_amount"),
        allowed_amount=amounts_data.get("allowed_amount"),
        paid_amount=amounts_data.get("paid_amount"),
        patient_responsibility=amounts_data.get("patient_responsibility"),
    )

    # Parse contact info
    contact_data = llm_data.get("contact_info", {})
    contact_info = ContactInfo(
        phone=contact_data.get("phone"),
        fax=contact_data.get("fax"),
        address=contact_data.get("address"),
        website=contact_data.get("website"),
    )

    # Parse payer info
    payer_data = llm_data.get("payer", {})
    plan_type_str = payer_data.get("plan_type", "unknown")
    try:
        plan_type = PlanType(plan_type_str)
    except ValueError:
        plan_type = PlanType.UNKNOWN
    payer_info = PayerInfo(
        name=payer_data.get("name") or user_context.get("payer_name"),
        plan_name=payer_data.get("plan_name") or user_context.get("plan_name"),
        plan_type=plan_type,
        member_id=payer_data.get("member_id"),
        group_number=payer_data.get("group_number"),
        claim_number=payer_data.get("claim_number"),
    )

    # Parse service info
    service_data = llm_data.get("service", {})
    service_info = ServiceInfo(
        description=service_data.get("description") or user_context.get("service_name"),
        cpt_codes=service_data.get("cpt_codes", []),
        diagnosis_codes=service_data.get("diagnosis_codes", []),
        date_of_service=_parse_date(service_data.get("date_of_service")) or dates.date_of_service,
        provider_name=service_data.get("provider_name") or user_context.get("provider_name"),
        facility_name=service_data.get("facility_name"),
    )

    # Get denial reason
    denial_reason = llm_data.get("denial_reason", "")
    denial_summary = llm_data.get("denial_reason_summary", denial_reason[:100])

    # Track missing info
    missing_info = _identify_missing_info(dates, codes, amounts, contact_info, payer_info)

    # Calculate confidence (higher for LLM extraction)
    confidence = ExtractionConfidence(
        overall=0.92 if denial_reason else 0.7,
        denial_reason=0.95 if denial_reason else 0.5,
        dates=0.9 if dates.date_of_denial else 0.6,
        codes=0.9 if codes else 0.5,
        amounts=0.95 if amounts.billed_amount else 0.7,
    )

    return CaseFacts(
        case_id=case_id,
        extraction_timestamp=datetime.now(timezone.utc),
        source_document=text.document_id,
        denial_reason=denial_reason,
        denial_reason_summary=denial_summary,
        denial_codes=codes,
        service=service_info,
        payer=payer_info,
        dates=dates,
        amounts=amounts,
        contact_info=contact_info,
        extraction_confidence=confidence,
        missing_info=missing_info,
        raw_text_snippets=RawTextSnippets(
            denial_statement=denial_reason[:500] if len(denial_reason) > 500 else denial_reason,
        ),
    )


def _parse_date(date_str: str | None) -> date | None:
    """Parse a date string into a date object."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None


def _extract_with_heuristics(
    case_id: str,
    text: ExtractedText,
    content: str,
    user_context: dict[str, Any],
) -> CaseFacts:
    """Extract case facts using heuristic/regex methods."""
    logger.info("Extracting case facts using heuristics")

    # Deterministic extraction
    dates = _extract_dates(content)
    codes = _extract_codes(content)
    amounts = _extract_amounts(content)
    contact_info = _extract_contact_info(content)

    # Extract denial reason using heuristics
    denial_reason, denial_summary = _extract_denial_reason_heuristic(content)

    # Extract payer info
    payer_info = _extract_payer_info(content, user_context)

    # Extract service info
    service_info = _extract_service_info(content, codes, dates, user_context)

    # Track missing info
    missing_info = _identify_missing_info(dates, codes, amounts, contact_info, payer_info)

    # Calculate confidence
    confidence = _calculate_confidence(denial_reason, dates, codes, amounts, contact_info)

    return CaseFacts(
        case_id=case_id,
        extraction_timestamp=datetime.now(timezone.utc),
        source_document=text.document_id,
        denial_reason=denial_reason,
        denial_reason_summary=denial_summary,
        denial_codes=codes,
        service=service_info,
        payer=payer_info,
        dates=dates,
        amounts=amounts,
        contact_info=contact_info,
        extraction_confidence=confidence,
        missing_info=missing_info,
        raw_text_snippets=RawTextSnippets(
            denial_statement=denial_reason[:500] if len(denial_reason) > 500 else denial_reason,
        ),
    )


def _extract_dates(text: str) -> CaseDates:
    """Extract dates from text using regex patterns."""
    dates = CaseDates()

    # Common date patterns
    date_patterns = [
        r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})",  # MM/DD/YYYY or MM-DD-YYYY
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})",
    ]

    found_dates: list[date] = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                if isinstance(match, tuple) and len(match) == 3:
                    if match[0].isdigit():
                        # MM/DD/YYYY format
                        month, day, year = int(match[0]), int(match[1]), int(match[2])
                        if year < 100:
                            year += 2000
                        found_dates.append(date(year, month, day))
                    else:
                        # Month name format
                        month_names = {
                            "january": 1,
                            "february": 2,
                            "march": 3,
                            "april": 4,
                            "may": 5,
                            "june": 6,
                            "july": 7,
                            "august": 8,
                            "september": 9,
                            "october": 10,
                            "november": 11,
                            "december": 12,
                        }
                        month = month_names.get(match[0].lower(), 1)
                        day = int(match[1])
                        year = int(match[2])
                        found_dates.append(date(year, month, day))
            except (ValueError, KeyError):
                continue

    # Try to identify specific date types from context
    text_lower = text.lower()

    # Look for appeal deadline
    appeal_pattern = r"appeal.{0,50}(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})"
    appeal_match = re.search(appeal_pattern, text_lower)
    if appeal_match:
        try:
            m, d, y = (
                int(appeal_match.group(1)),
                int(appeal_match.group(2)),
                int(appeal_match.group(3)),
            )
            if y < 100:
                y += 2000
            dates.appeal_deadline = date(y, m, d)
        except ValueError:
            pass

    # Look for appeal deadline in days
    days_pattern = r"within\s+(\d+)\s*(?:calendar\s*)?days"
    days_match = re.search(days_pattern, text_lower)
    if days_match:
        dates.appeal_deadline_days = int(days_match.group(1))

    # Look for date of service
    dos_pattern = r"(?:date of service|service date|dos)[:\s]*(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})"
    dos_match = re.search(dos_pattern, text_lower)
    if dos_match:
        try:
            m, d, y = int(dos_match.group(1)), int(dos_match.group(2)), int(dos_match.group(3))
            if y < 100:
                y += 2000
            dates.date_of_service = date(y, m, d)
        except ValueError:
            pass

    # If we found dates but couldn't categorize them, use heuristics
    if found_dates and not dates.date_of_denial:
        # Latest date is probably the denial date
        dates.date_of_denial = max(found_dates)

    return dates


def _extract_codes(text: str) -> list[DenialCode]:
    """Extract medical and billing codes from text."""
    codes: list[DenialCode] = []
    seen: set[str] = set()

    # CPT codes (5 digits)
    cpt_pattern = r"\b(\d{5})\b"
    for match in re.finditer(cpt_pattern, text):
        code = match.group(1)
        # Filter out likely non-CPT numbers (years, zip codes, etc.)
        if code not in seen and not code.startswith("20") and not code.startswith("19"):
            codes.append(DenialCode(code=code, code_type=CodeType.CPT))
            seen.add(code)

    # ICD-10 codes (letter + 2 digits + optional decimal + more chars)
    icd_pattern = r"\b([A-Z]\d{2}\.?\d{0,4})\b"
    for match in re.finditer(icd_pattern, text):
        code = match.group(1)
        if code not in seen and len(code) >= 3:
            codes.append(DenialCode(code=code, code_type=CodeType.ICD_10_CM))
            seen.add(code)

    # HCPCS codes (letter + 4 digits)
    hcpcs_pattern = r"\b([A-Z]\d{4})\b"
    for match in re.finditer(hcpcs_pattern, text):
        code = match.group(1)
        if code not in seen:
            codes.append(DenialCode(code=code, code_type=CodeType.HCPCS))
            seen.add(code)

    # CARC/RARC codes
    carc_pattern = r"\b(CO|PR|OA|PI|CR)-?(\d{1,3})\b"
    for match in re.finditer(carc_pattern, text):
        code = f"{match.group(1)}-{match.group(2)}"
        if code not in seen:
            codes.append(DenialCode(code=code, code_type=CodeType.CARC))
            seen.add(code)

    return codes


def _extract_amounts(text: str) -> CaseAmounts:
    """Extract dollar amounts from text."""
    amounts = CaseAmounts()

    # Find all dollar amounts
    amount_pattern = r"\$\s*([\d,]+\.?\d{0,2})"
    found_amounts: list[float] = []
    for match in re.finditer(amount_pattern, text):
        try:
            amount = float(match.group(1).replace(",", ""))
            found_amounts.append(amount)
        except ValueError:
            continue

    # Try to categorize amounts by context
    text_lower = text.lower()

    # Billed amount
    billed_pattern = r"(?:billed|charged|total charges?)[:\s]*\$\s*([\d,]+\.?\d{0,2})"
    billed_match = re.search(billed_pattern, text_lower)
    if billed_match:
        amounts.billed_amount = float(billed_match.group(1).replace(",", ""))

    # Allowed amount
    allowed_pattern = r"(?:allowed|eligible)[:\s]*\$\s*([\d,]+\.?\d{0,2})"
    allowed_match = re.search(allowed_pattern, text_lower)
    if allowed_match:
        amounts.allowed_amount = float(allowed_match.group(1).replace(",", ""))

    # Patient responsibility
    patient_pattern = (
        r"(?:patient|member|your)\s*(?:responsibility|owes?|amount)[:\s]*\$\s*([\d,]+\.?\d{0,2})"
    )
    patient_match = re.search(patient_pattern, text_lower)
    if patient_match:
        amounts.patient_responsibility = float(patient_match.group(1).replace(",", ""))

    return amounts


def _extract_contact_info(text: str) -> ContactInfo:
    """Extract contact information from text."""
    contact = ContactInfo()

    # Phone numbers
    phone_pattern = r"(?:phone|call|tel)[:\s]*[(\s]*(\d{3})[)\s.-]*(\d{3})[\s.-]*(\d{4})"
    phone_match = re.search(phone_pattern, text, re.IGNORECASE)
    if phone_match:
        contact.phone = f"({phone_match.group(1)}) {phone_match.group(2)}-{phone_match.group(3)}"
    else:
        # Try standalone phone pattern
        standalone_phone = r"\b(\d{3})[\s.-](\d{3})[\s.-](\d{4})\b"
        phone_match = re.search(standalone_phone, text)
        if phone_match:
            contact.phone = (
                f"({phone_match.group(1)}) {phone_match.group(2)}-{phone_match.group(3)}"
            )

    # Fax numbers
    fax_pattern = r"(?:fax)[:\s]*[(\s]*(\d{3})[)\s.-]*(\d{3})[\s.-]*(\d{4})"
    fax_match = re.search(fax_pattern, text, re.IGNORECASE)
    if fax_match:
        contact.fax = f"({fax_match.group(1)}) {fax_match.group(2)}-{fax_match.group(3)}"

    return contact


def _extract_payer_info(text: str, user_context: dict[str, Any]) -> PayerInfo:
    """Extract payer information from text and user context."""
    payer = PayerInfo()

    # Use user context if provided
    if user_context.get("payer_name"):
        payer.name = user_context["payer_name"]
    if user_context.get("plan_name"):
        payer.plan_name = user_context["plan_name"]

    # Look for common payer names
    payer_patterns = [
        r"(Blue\s*Cross\s*Blue\s*Shield|BCBS)",
        r"(United\s*Health\s*(?:care|Group)?|UHC)",
        r"(Aetna)",
        r"(Cigna)",
        r"(Humana)",
        r"(Kaiser\s*Permanente)",
        r"(Anthem)",
        r"(Medicare)",
        r"(Medicaid)",
    ]

    if not payer.name:
        for pattern in payer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                payer.name = match.group(1).strip()
                break

    # Look for member ID
    member_pattern = r"(?:member|subscriber|id)[:\s#]*([A-Z0-9]{6,15})"
    member_match = re.search(member_pattern, text, re.IGNORECASE)
    if member_match:
        payer.member_id = member_match.group(1)

    # Look for claim number
    claim_pattern = r"(?:claim|reference)[:\s#]*([A-Z0-9]{8,20})"
    claim_match = re.search(claim_pattern, text, re.IGNORECASE)
    if claim_match:
        payer.claim_number = claim_match.group(1)

    return payer


def _extract_service_info(
    text: str,
    codes: list[DenialCode],
    dates: CaseDates,
    user_context: dict[str, Any],
) -> ServiceInfo:
    """Extract service information."""
    service = ServiceInfo()

    # Use user context
    if user_context.get("service_name"):
        service.description = user_context["service_name"]
    if user_context.get("provider_name"):
        service.provider_name = user_context["provider_name"]
    if user_context.get("date_of_service"):
        service.date_of_service = user_context["date_of_service"]

    # Copy date from dates if found
    if dates.date_of_service:
        service.date_of_service = dates.date_of_service

    # Separate CPT and ICD codes
    service.cpt_codes = [c.code for c in codes if c.code_type == CodeType.CPT]
    service.diagnosis_codes = [c.code for c in codes if c.code_type == CodeType.ICD_10_CM]

    # Look for NPI
    npi_pattern = r"(?:NPI|provider)[:\s#]*(\d{10})"
    npi_match = re.search(npi_pattern, text, re.IGNORECASE)
    if npi_match:
        service.provider_npi = npi_match.group(1)

    return service


def _extract_denial_reason_heuristic(text: str) -> tuple[str, str]:
    """Extract denial reason using heuristics (no LLM)."""
    text_lower = text.lower()

    # Look for common denial patterns
    denial_patterns = [
        (
            r"(?:denied|denial|not covered)\s*(?:because|due to|reason)[:\s]*([^.]+\.)",
            "pattern_match",
        ),
        (r"reason\s*(?:for\s*)?(?:denial|determination)[:\s]*([^.]+\.)", "reason_field"),
        (
            r"(?:this service|the claim|your claim)\s+(?:is|was|has been)\s+(?:denied|not covered)\s*(?:because|due to)?[:\s]*([^.]+\.)",
            "denial_statement",
        ),
    ]

    for pattern, _ in denial_patterns:
        match = re.search(pattern, text_lower)
        if match:
            reason = match.group(1).strip()
            # Clean up and capitalize
            reason = reason.capitalize()
            if len(reason) > 10:
                return reason, reason

    # Fallback: look for key phrases
    if "prior authorization" in text_lower or "pre-certification" in text_lower:
        return (
            "Prior authorization was not obtained for this service.",
            "Service requires prior authorization",
        )

    if "medical necessity" in text_lower or "not medically necessary" in text_lower:
        return (
            "The service was determined not to be medically necessary.",
            "Medical necessity not established",
        )

    if "not covered" in text_lower or "excluded" in text_lower:
        return "The service is not covered under your plan.", "Service not covered by plan"

    if "coding" in text_lower or "modifier" in text_lower:
        return (
            "There is an issue with the coding or billing of this claim.",
            "Coding or billing issue",
        )

    # Ultimate fallback
    return (
        "Denial reason could not be automatically determined. Please review the letter.",
        "Unknown denial reason",
    )


def _identify_missing_info(
    dates: CaseDates,
    codes: list[DenialCode],
    amounts: CaseAmounts,
    contact: ContactInfo,
    payer: PayerInfo,
) -> list[MissingInfo]:
    """Identify what information is missing."""
    missing: list[MissingInfo] = []

    if not dates.appeal_deadline and not dates.appeal_deadline_days:
        missing.append(
            MissingInfo(field="appeal_deadline", reason="Could not find appeal deadline in letter")
        )

    if not dates.date_of_service:
        missing.append(
            MissingInfo(field="date_of_service", reason="Could not find date of service")
        )

    if not codes:
        missing.append(
            MissingInfo(field="denial_codes", reason="No procedure or diagnosis codes found")
        )

    if not contact.phone and not contact.fax:
        missing.append(
            MissingInfo(field="contact_info", reason="No contact information found for appeals")
        )

    if not payer.name:
        missing.append(
            MissingInfo(field="payer_name", reason="Could not identify insurance company")
        )

    return missing


def _calculate_confidence(
    denial_reason: str,
    dates: CaseDates,
    codes: list[DenialCode],
    amounts: CaseAmounts,
    contact: ContactInfo,
) -> ExtractionConfidence:
    """Calculate confidence scores for extraction."""
    scores: list[float] = []

    # Denial reason confidence
    denial_conf = 0.5  # Base
    if len(denial_reason) > 50:
        denial_conf += 0.3
    if "could not" not in denial_reason.lower():
        denial_conf += 0.2
    scores.append(min(denial_conf, 1.0))

    # Dates confidence
    dates_conf = 0.3
    if dates.appeal_deadline or dates.appeal_deadline_days:
        dates_conf += 0.4
    if dates.date_of_service:
        dates_conf += 0.3
    scores.append(min(dates_conf, 1.0))

    # Codes confidence
    codes_conf = 0.2 if not codes else min(0.3 + len(codes) * 0.1, 1.0)
    scores.append(codes_conf)

    # Amounts confidence (lower weight - often not present)
    amounts_conf = 0.5
    if amounts.billed_amount:
        amounts_conf += 0.25
    if amounts.patient_responsibility:
        amounts_conf += 0.25
    scores.append(min(amounts_conf, 1.0))

    overall = sum(scores) / len(scores) if scores else 0.5

    return ExtractionConfidence(
        overall=round(overall, 2),
        denial_reason=round(scores[0], 2) if scores else 0.5,
        dates=round(scores[1], 2) if len(scores) > 1 else 0.5,
        codes=round(scores[2], 2) if len(scores) > 2 else 0.5,
        amounts=round(scores[3], 2) if len(scores) > 3 else 0.5,
    )
