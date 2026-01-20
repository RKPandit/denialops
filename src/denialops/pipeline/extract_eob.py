"""EOB (Explanation of Benefits) document extraction."""

import contextlib
import json
import re
from datetime import date
from decimal import Decimal

from denialops.llm import create_llm_client
from denialops.models.documents import ExtractedText
from denialops.models.eob import (
    AccumulatorInfo,
    ClaimStatus,
    EOBFacts,
    MemberCostSummary,
    ProviderInfo,
    ServiceLine,
)


def extract_eob_facts(
    case_id: str,
    text: ExtractedText,
    llm_api_key: str = "",
    llm_model: str = "",
    llm_provider: str = "openai",
) -> EOBFacts:
    """
    Extract facts from an EOB document.

    Uses LLM if available, falls back to heuristics.
    """
    if llm_api_key:
        try:
            return _extract_with_llm(
                case_id, text, llm_api_key, llm_model, llm_provider
            )
        except Exception:
            pass

    return _extract_with_heuristics(case_id, text)


EOB_SYSTEM_PROMPT = """You are an expert at analyzing Explanation of Benefits (EOB) documents.
Extract structured information from the EOB text provided.
Always respond with valid JSON. Be precise and accurate.
If information is not present in the document, use null for that field."""

EOB_USER_PROMPT = """Analyze this Explanation of Benefits (EOB) document and extract structured information.

Document text:
{text}

Extract and return a JSON object with these fields:
- eob_date: Date of the EOB (YYYY-MM-DD format or null)
- claim_number: Insurance claim reference number
- claim_status: One of "paid", "partially_paid", "denied", "pending", "adjusted"
- provider_name: Name of the healthcare provider
- provider_npi: NPI number if present
- is_in_network: true/false/null if unknown
- total_billed: Total billed amount (number)
- total_allowed: Total allowed amount (number)
- total_paid: Total paid by insurance (number)
- deductible_applied: Amount applied to deductible (number)
- copay: Copay amount (number)
- coinsurance: Coinsurance amount (number)
- not_covered: Amount not covered (number)
- total_member_responsibility: Total patient owes (number)
- denial_codes: List of denial/adjustment codes
- denial_reasons: List of denial reason descriptions
- appeal_deadline: Appeal deadline if mentioned (YYYY-MM-DD or null)
- service_lines: Array of services with procedure_code, description, billed_amount, paid_amount, denial_code

Return only valid JSON, no explanation."""


def _extract_with_llm(
    case_id: str,
    text: ExtractedText,
    llm_api_key: str,
    llm_model: str,
    llm_provider: str,
) -> EOBFacts:
    """Extract EOB facts using LLM."""
    # Create LLM client
    client = create_llm_client(
        provider=llm_provider,
        api_key=llm_api_key,
        model=llm_model if llm_model else None,
    )

    # Call LLM for extraction
    prompt = EOB_USER_PROMPT.format(text=text.full_text[:8000])
    llm_response = client.complete(
        prompt=prompt,
        system=EOB_SYSTEM_PROMPT,
        max_tokens=2000,
        temperature=0.0,
    )
    response = llm_response.content

    # Parse LLM response
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return _extract_with_heuristics(case_id, text)

    # Build EOBFacts from parsed data
    provider = ProviderInfo(
        name=data.get("provider_name"),
        npi=data.get("provider_npi"),
        is_in_network=data.get("is_in_network"),
    )

    member_costs = MemberCostSummary(
        deductible_applied=Decimal(str(data.get("deductible_applied", 0))),
        copay=Decimal(str(data.get("copay", 0))),
        coinsurance=Decimal(str(data.get("coinsurance", 0))),
        not_covered=Decimal(str(data.get("not_covered", 0))),
        total_member_responsibility=Decimal(
            str(data.get("total_member_responsibility", 0))
        ),
    )

    service_lines = []
    for sl in data.get("service_lines", []):
        service_lines.append(
            ServiceLine(
                description=sl.get("description", ""),
                procedure_code=sl.get("procedure_code"),
                billed_amount=Decimal(str(sl.get("billed_amount", 0)))
                if sl.get("billed_amount")
                else None,
                paid_amount=Decimal(str(sl.get("paid_amount", 0)))
                if sl.get("paid_amount")
                else None,
                denial_code=sl.get("denial_code"),
            )
        )

    # Parse dates
    eob_date = None
    if data.get("eob_date"):
        with contextlib.suppress(ValueError):
            eob_date = date.fromisoformat(data["eob_date"])

    appeal_deadline = None
    if data.get("appeal_deadline"):
        with contextlib.suppress(ValueError):
            appeal_deadline = date.fromisoformat(data["appeal_deadline"])

    # Parse claim status
    status_map = {
        "paid": ClaimStatus.PAID,
        "partially_paid": ClaimStatus.PARTIALLY_PAID,
        "denied": ClaimStatus.DENIED,
        "pending": ClaimStatus.PENDING,
        "adjusted": ClaimStatus.ADJUSTED,
    }
    claim_status = status_map.get(
        data.get("claim_status", "").lower(), ClaimStatus.DENIED
    )

    return EOBFacts(
        case_id=case_id,
        eob_date=eob_date,
        claim_number=data.get("claim_number"),
        claim_status=claim_status,
        provider=provider,
        service_lines=service_lines,
        total_billed=Decimal(str(data.get("total_billed", 0)))
        if data.get("total_billed")
        else None,
        total_allowed=Decimal(str(data.get("total_allowed", 0)))
        if data.get("total_allowed")
        else None,
        total_paid=Decimal(str(data.get("total_paid", 0)))
        if data.get("total_paid")
        else None,
        member_costs=member_costs,
        appeal_deadline=appeal_deadline,
        denial_codes=data.get("denial_codes", []),
        denial_reasons=data.get("denial_reasons", []),
        extraction_confidence=0.8,
    )


def _extract_with_heuristics(case_id: str, text: ExtractedText) -> EOBFacts:
    """Extract EOB facts using pattern matching heuristics."""
    full_text = text.full_text

    # Extract claim number
    claim_number = _extract_claim_number(full_text)

    # Extract provider info
    provider = _extract_provider_info(full_text)

    # Extract amounts
    total_billed, total_allowed, total_paid = _extract_amounts(full_text)

    # Extract member costs
    member_costs = _extract_member_costs(full_text)

    # Extract denial codes and reasons
    denial_codes, denial_reasons = _extract_denial_info(full_text)

    # Extract accumulator info
    accumulators = _extract_accumulators(full_text)

    # Determine claim status
    claim_status = _determine_claim_status(full_text, denial_codes)

    # Extract service lines
    service_lines = _extract_service_lines(full_text)

    return EOBFacts(
        case_id=case_id,
        claim_number=claim_number,
        claim_status=claim_status,
        provider=provider,
        service_lines=service_lines,
        total_billed=total_billed,
        total_allowed=total_allowed,
        total_paid=total_paid,
        member_costs=member_costs,
        accumulators=accumulators,
        denial_codes=denial_codes,
        denial_reasons=denial_reasons,
        extraction_confidence=0.5,
    )


def _extract_claim_number(text: str) -> str | None:
    """Extract claim number from EOB text."""
    patterns = [
        r"claim\s*(?:number|#|no\.?)[\s:]*([A-Z0-9-]+)",
        r"claim\s*id[\s:]*([A-Z0-9-]+)",
        r"reference[\s:]*([A-Z0-9-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _extract_provider_info(text: str) -> ProviderInfo:
    """Extract provider information from EOB text."""
    provider_name = None
    npi = None
    is_in_network = None

    # Provider name patterns
    name_patterns = [
        r"provider[\s:]*([A-Z][A-Za-z\s&,\.]+?)(?:\n|$)",
        r"from[\s:]*([A-Z][A-Za-z\s&,\.]+?)(?:\n|$)",
    ]
    for pattern in name_patterns:
        match = re.search(pattern, text)
        if match:
            provider_name = match.group(1).strip()[:100]
            break

    # NPI pattern
    npi_match = re.search(r"NPI[\s:]*(\d{10})", text, re.IGNORECASE)
    if npi_match:
        npi = npi_match.group(1)

    # Network status
    if re.search(r"in[-\s]*network", text, re.IGNORECASE):
        is_in_network = True
    elif re.search(r"out[-\s]*of[-\s]*network", text, re.IGNORECASE):
        is_in_network = False

    return ProviderInfo(name=provider_name, npi=npi, is_in_network=is_in_network)


def _extract_amounts(
    text: str,
) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
    """Extract billed, allowed, and paid amounts."""
    billed = None
    allowed = None
    paid = None

    # Amount patterns
    billed_match = re.search(
        r"(?:total\s+)?(?:billed|charges?)[\s:]*\$?([\d,]+\.?\d*)", text, re.IGNORECASE
    )
    if billed_match:
        billed = Decimal(billed_match.group(1).replace(",", ""))

    allowed_match = re.search(
        r"(?:total\s+)?allowed[\s:]*\$?([\d,]+\.?\d*)", text, re.IGNORECASE
    )
    if allowed_match:
        allowed = Decimal(allowed_match.group(1).replace(",", ""))

    paid_match = re.search(
        r"(?:total\s+)?(?:paid|payment)[\s:]*\$?([\d,]+\.?\d*)", text, re.IGNORECASE
    )
    if paid_match:
        paid = Decimal(paid_match.group(1).replace(",", ""))

    return billed, allowed, paid


def _extract_member_costs(text: str) -> MemberCostSummary:
    """Extract member cost breakdown from EOB."""
    deductible = Decimal("0")
    copay = Decimal("0")
    coinsurance = Decimal("0")
    not_covered = Decimal("0")
    total = Decimal("0")

    deductible_match = re.search(
        r"deductible[\s:]*\$?([\d,]+\.?\d*)", text, re.IGNORECASE
    )
    if deductible_match:
        deductible = Decimal(deductible_match.group(1).replace(",", ""))

    copay_match = re.search(r"copay[\s:]*\$?([\d,]+\.?\d*)", text, re.IGNORECASE)
    if copay_match:
        copay = Decimal(copay_match.group(1).replace(",", ""))

    coinsurance_match = re.search(
        r"coinsurance[\s:]*\$?([\d,]+\.?\d*)", text, re.IGNORECASE
    )
    if coinsurance_match:
        coinsurance = Decimal(coinsurance_match.group(1).replace(",", ""))

    not_covered_match = re.search(
        r"not\s+covered[\s:]*\$?([\d,]+\.?\d*)", text, re.IGNORECASE
    )
    if not_covered_match:
        not_covered = Decimal(not_covered_match.group(1).replace(",", ""))

    # Total member responsibility
    total_match = re.search(
        r"(?:your|member|patient)\s+(?:responsibility|amount|owes?)[\s:]*\$?([\d,]+\.?\d*)",
        text,
        re.IGNORECASE,
    )
    if total_match:
        total = Decimal(total_match.group(1).replace(",", ""))
    else:
        total = deductible + copay + coinsurance + not_covered

    return MemberCostSummary(
        deductible_applied=deductible,
        copay=copay,
        coinsurance=coinsurance,
        not_covered=not_covered,
        total_member_responsibility=total,
    )


def _extract_denial_info(text: str) -> tuple[list[str], list[str]]:
    """Extract denial codes and reasons from EOB."""
    codes: list[str] = []
    reasons: list[str] = []

    # Common denial code patterns
    code_patterns = [
        r"(?:denial|reason|remark)\s*code[\s:]*([A-Z0-9-]+)",
        r"\b(CO-\d+)\b",
        r"\b(PR-\d+)\b",
        r"\b(OA-\d+)\b",
    ]
    for pattern in code_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        codes.extend(matches)

    # Deduplicate codes
    codes = list(set(codes))

    # Common reason patterns
    reason_patterns = [
        r"(?:reason|denied|denial)[\s:]*([^\.]+(?:necessity|authorization|coverage|benefit)[^\.]*\.)",
        r"not\s+(?:covered|payable|eligible)[^\.]*\.",
    ]
    for pattern in reason_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        reasons.extend([m.strip() for m in matches])

    # Deduplicate reasons
    reasons = list(set(reasons))

    return codes, reasons


def _extract_accumulators(text: str) -> AccumulatorInfo:
    """Extract YTD accumulator information."""
    deductible_ytd = None
    deductible_max = None
    oop_ytd = None
    oop_max = None

    # Deductible patterns
    ded_ytd_match = re.search(
        r"deductible\s+(?:used|met|applied)[\s:]*\$?([\d,]+\.?\d*)", text, re.IGNORECASE
    )
    if ded_ytd_match:
        deductible_ytd = Decimal(ded_ytd_match.group(1).replace(",", ""))

    ded_max_match = re.search(
        r"(?:annual|yearly)\s+deductible[\s:]*\$?([\d,]+\.?\d*)", text, re.IGNORECASE
    )
    if ded_max_match:
        deductible_max = Decimal(ded_max_match.group(1).replace(",", ""))

    # Out-of-pocket patterns
    oop_ytd_match = re.search(
        r"out[- ]of[- ]pocket\s+(?:used|met|applied)[\s:]*\$?([\d,]+\.?\d*)",
        text,
        re.IGNORECASE,
    )
    if oop_ytd_match:
        oop_ytd = Decimal(oop_ytd_match.group(1).replace(",", ""))

    oop_max_match = re.search(
        r"(?:annual|yearly|maximum)\s+out[- ]of[- ]pocket[\s:]*\$?([\d,]+\.?\d*)",
        text,
        re.IGNORECASE,
    )
    if oop_max_match:
        oop_max = Decimal(oop_max_match.group(1).replace(",", ""))

    return AccumulatorInfo(
        deductible_ytd=deductible_ytd,
        deductible_max=deductible_max,
        oop_ytd=oop_ytd,
        oop_max=oop_max,
    )


def _determine_claim_status(text: str, denial_codes: list[str]) -> ClaimStatus:
    """Determine overall claim status from EOB."""
    text_lower = text.lower()

    if denial_codes or "denied" in text_lower or "not covered" in text_lower:
        if "partial" in text_lower:
            return ClaimStatus.PARTIALLY_PAID
        return ClaimStatus.DENIED

    if "pending" in text_lower or "processing" in text_lower:
        return ClaimStatus.PENDING

    if "adjusted" in text_lower:
        return ClaimStatus.ADJUSTED

    if "paid" in text_lower:
        return ClaimStatus.PAID

    return ClaimStatus.DENIED


def _extract_service_lines(text: str) -> list[ServiceLine]:
    """Extract individual service line items."""
    lines: list[ServiceLine] = []

    # Look for CPT codes with descriptions
    cpt_pattern = r"(\d{5})\s+([A-Za-z\s]+?)\s+\$?([\d,]+\.?\d*)"
    matches = re.findall(cpt_pattern, text)

    for code, desc, amount in matches[:10]:  # Limit to 10 lines
        lines.append(
            ServiceLine(
                procedure_code=code,
                description=desc.strip()[:100],
                billed_amount=Decimal(amount.replace(",", "")) if amount else None,
            )
        )

    return lines
