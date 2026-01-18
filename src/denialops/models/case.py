"""Pydantic models for case facts extraction."""

from datetime import date, datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class CaseMode(str, Enum):
    """Operating mode for case processing."""

    FAST = "fast"
    VERIFIED = "verified"


class CodeType(str, Enum):
    """Types of medical/billing codes."""

    CPT = "CPT"
    HCPCS = "HCPCS"
    ICD_10_CM = "ICD-10-CM"
    ICD_10_PCS = "ICD-10-PCS"
    CARC = "CARC"
    RARC = "RARC"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


class PlanType(str, Enum):
    """Types of insurance plans."""

    HMO = "HMO"
    PPO = "PPO"
    EPO = "EPO"
    POS = "POS"
    HDHP = "HDHP"
    MEDICARE = "Medicare"
    MEDICAID = "Medicaid"
    UNKNOWN = "unknown"


class DenialCode(BaseModel):
    """A code mentioned in the denial letter."""

    code: str = Field(..., description="The code value")
    code_type: CodeType = Field(..., description="Type of code")
    description: str | None = Field(None, description="Description if provided")


class ServiceInfo(BaseModel):
    """Information about the denied service."""

    description: str | None = Field(None, description="Service or procedure description")
    cpt_codes: list[str] = Field(default_factory=list, description="CPT/HCPCS codes")
    diagnosis_codes: list[str] = Field(default_factory=list, description="ICD-10 diagnosis codes")
    date_of_service: date | None = Field(None, description="Date service was provided")
    provider_name: str | None = Field(None, description="Name of the provider")
    provider_npi: Annotated[str | None, Field(pattern=r"^[0-9]{10}$")] = Field(
        None, description="Provider NPI (10 digits)"
    )
    facility_name: str | None = Field(None, description="Facility name")


class PayerInfo(BaseModel):
    """Insurance payer information."""

    name: str | None = Field(None, description="Payer/insurance company name")
    plan_name: str | None = Field(None, description="Specific plan name")
    plan_type: PlanType = Field(PlanType.UNKNOWN, description="Type of insurance plan")
    member_id: str | None = Field(None, description="Member/subscriber ID")
    group_number: str | None = Field(None, description="Group number")
    claim_number: str | None = Field(None, description="Claim reference number")


class CaseDates(BaseModel):
    """Key dates from the denial."""

    date_of_service: date | None = None
    date_of_denial: date | None = Field(None, description="Date the denial letter was issued")
    appeal_deadline: date | None = Field(None, description="Deadline to file appeal")
    appeal_deadline_days: int | None = Field(
        None, ge=0, description="Days from denial to appeal deadline"
    )
    timely_filing_deadline: date | None = Field(None, description="Timely filing deadline")


class CaseAmounts(BaseModel):
    """Financial amounts from the denial/EOB."""

    billed_amount: float | None = Field(None, ge=0, description="Amount billed by provider")
    allowed_amount: float | None = Field(None, ge=0, description="Amount allowed by payer")
    paid_amount: float | None = Field(None, ge=0, description="Amount paid by payer")
    patient_responsibility: float | None = Field(None, ge=0, description="Amount patient owes")
    deductible_applied: float | None = Field(None, ge=0)
    coinsurance_applied: float | None = Field(None, ge=0)
    copay_applied: float | None = Field(None, ge=0)


class ContactInfo(BaseModel):
    """Payer contact information for appeals."""

    phone: str | None = Field(None, description="Phone number for appeals")
    fax: str | None = Field(None, description="Fax number for appeals")
    address: str | None = Field(None, description="Mailing address for appeals")
    website: str | None = Field(None, description="Website for appeals")
    email: str | None = Field(None, description="Email address")


class PriorAuthInfo(BaseModel):
    """Prior authorization details."""

    was_required: bool | None = Field(None, description="Whether PA was required")
    was_obtained: bool | None = Field(None, description="Whether PA was obtained")
    auth_number: str | None = Field(None, description="PA number if provided")
    denial_reason: str | None = Field(None, description="Reason PA was denied or not valid")


class ExtractionConfidence(BaseModel):
    """Confidence scores for extracted fields."""

    overall: float = Field(..., ge=0, le=1, description="Overall extraction confidence")
    denial_reason: float | None = Field(None, ge=0, le=1)
    dates: float | None = Field(None, ge=0, le=1)
    amounts: float | None = Field(None, ge=0, le=1)
    codes: float | None = Field(None, ge=0, le=1)


class MissingInfo(BaseModel):
    """Information that could not be extracted."""

    field: str = Field(..., description="Name of the missing field")
    reason: str = Field(..., description="Why it could not be extracted")


class RawTextSnippets(BaseModel):
    """Key text excerpts for citation/grounding."""

    denial_statement: str | None = Field(None, description="Exact text of denial reason")
    appeal_instructions: str | None = Field(None, description="Appeal instructions as stated")


class CaseFacts(BaseModel):
    """Structured extraction from a denial letter."""

    case_id: str = Field(..., description="Unique identifier for this case")
    extraction_timestamp: datetime = Field(..., description="When extraction was performed")
    source_document: str | None = Field(None, description="Filename of the source denial letter")

    # Core denial info
    denial_reason: str = Field(..., min_length=1, description="Primary denial reason")
    denial_reason_summary: str | None = Field(None, description="Plain-language summary")
    denial_codes: list[DenialCode] = Field(
        default_factory=list, description="All codes mentioned in the denial"
    )

    # Structured info
    service: ServiceInfo | None = None
    payer: PayerInfo | None = None
    dates: CaseDates | None = None
    amounts: CaseAmounts | None = None
    contact_info: ContactInfo | None = None
    prior_authorization: PriorAuthInfo | None = None

    # Quality metadata
    extraction_confidence: ExtractionConfidence | None = None
    missing_info: list[MissingInfo] = Field(default_factory=list)
    raw_text_snippets: RawTextSnippets | None = None


class UserContext(BaseModel):
    """Optional user-provided context about the case."""

    payer_name: str | None = None
    plan_name: str | None = None
    state: Annotated[str | None, Field(pattern=r"^[A-Z]{2}$")] = None
    service_name: str | None = None
    date_of_service: date | None = None
    provider_name: str | None = None
    urgency: str = Field("standard", pattern=r"^(standard|expedited)$")
