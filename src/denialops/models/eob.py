"""Models for Explanation of Benefits (EOB) document extraction."""

from datetime import date
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field


class ClaimStatus(str, Enum):
    """Status of a claim in the EOB."""

    PAID = "paid"
    PARTIALLY_PAID = "partially_paid"
    DENIED = "denied"
    PENDING = "pending"
    ADJUSTED = "adjusted"


class ServiceLine(BaseModel):
    """A single service line item from the EOB."""

    service_date: date | None = Field(None, description="Date of service")
    description: str = Field("", description="Service description")
    procedure_code: str | None = Field(None, description="CPT/HCPCS code")
    diagnosis_codes: list[str] = Field(
        default_factory=list, description="ICD-10 diagnosis codes"
    )
    billed_amount: Decimal | None = Field(None, description="Amount billed by provider")
    allowed_amount: Decimal | None = Field(
        None, description="Amount allowed by insurance"
    )
    paid_amount: Decimal | None = Field(None, description="Amount paid by insurance")
    member_responsibility: Decimal | None = Field(
        None, description="Amount patient owes"
    )
    denial_code: str | None = Field(None, description="Denial/adjustment code if any")
    denial_reason: str | None = Field(None, description="Reason for denial/adjustment")


class ProviderInfo(BaseModel):
    """Provider information from the EOB."""

    name: str | None = Field(None, description="Provider name")
    npi: str | None = Field(None, description="National Provider Identifier")
    tax_id: str | None = Field(None, description="Tax ID (may be partially masked)")
    address: str | None = Field(None, description="Provider address")
    is_in_network: bool | None = Field(None, description="Whether provider is in-network")


class MemberCostSummary(BaseModel):
    """Summary of member costs from the EOB."""

    deductible_applied: Decimal = Field(
        Decimal("0"), description="Amount applied to deductible"
    )
    copay: Decimal = Field(Decimal("0"), description="Copay amount")
    coinsurance: Decimal = Field(Decimal("0"), description="Coinsurance amount")
    not_covered: Decimal = Field(
        Decimal("0"), description="Amount not covered by plan"
    )
    total_member_responsibility: Decimal = Field(
        Decimal("0"), description="Total amount patient owes"
    )


class AccumulatorInfo(BaseModel):
    """Year-to-date accumulator information."""

    deductible_ytd: Decimal | None = Field(
        None, description="Year-to-date deductible met"
    )
    deductible_max: Decimal | None = Field(
        None, description="Annual deductible maximum"
    )
    oop_ytd: Decimal | None = Field(None, description="Year-to-date out-of-pocket")
    oop_max: Decimal | None = Field(
        None, description="Annual out-of-pocket maximum"
    )


class EOBFacts(BaseModel):
    """Extracted facts from an Explanation of Benefits document."""

    # Document identifiers
    case_id: str = Field(..., description="Case this EOB belongs to")
    eob_date: date | None = Field(None, description="Date EOB was generated")
    claim_number: str | None = Field(None, description="Insurance claim number")

    # Claim status
    claim_status: ClaimStatus = Field(
        ClaimStatus.DENIED, description="Overall claim status"
    )

    # Provider
    provider: ProviderInfo = Field(
        default_factory=ProviderInfo, description="Provider information"
    )

    # Service details
    service_lines: list[ServiceLine] = Field(
        default_factory=list, description="Individual service line items"
    )

    # Financial summary
    total_billed: Decimal | None = Field(None, description="Total billed amount")
    total_allowed: Decimal | None = Field(None, description="Total allowed amount")
    total_paid: Decimal | None = Field(None, description="Total paid by insurance")
    member_costs: MemberCostSummary = Field(
        default_factory=MemberCostSummary, description="Member cost breakdown"
    )

    # Accumulator info
    accumulators: AccumulatorInfo = Field(
        default_factory=AccumulatorInfo, description="YTD accumulator status"
    )

    # Appeal information from EOB
    appeal_deadline: date | None = Field(
        None, description="Deadline to appeal (if shown)"
    )
    appeal_instructions: str | None = Field(
        None, description="Appeal instructions from EOB"
    )

    # Denial codes for reference
    denial_codes: list[str] = Field(
        default_factory=list, description="All denial/adjustment codes"
    )
    denial_reasons: list[str] = Field(
        default_factory=list, description="All denial reason descriptions"
    )

    # Extraction metadata
    extraction_confidence: float = Field(
        0.5, ge=0.0, le=1.0, description="Confidence in extraction accuracy"
    )
