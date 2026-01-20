"""Plan rules models extracted from SBC/EOC documents."""

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field


class PlanDocumentType(str, Enum):
    """Type of plan document."""

    SBC = "SBC"  # Summary of Benefits and Coverage
    EOC = "EOC"  # Evidence of Coverage
    COC = "COC"  # Certificate of Coverage
    SPD = "SPD"  # Summary Plan Description
    OTHER = "other"


class PlanType(str, Enum):
    """Type of insurance plan."""

    HMO = "HMO"
    PPO = "PPO"
    EPO = "EPO"
    POS = "POS"
    HDHP = "HDHP"
    MEDICARE = "Medicare"
    MEDICAID = "Medicaid"
    OTHER = "other"


# =============================================================================
# Citation Models (for grounding)
# =============================================================================


class SourceCitation(BaseModel):
    """Citation to source document for grounding."""

    source_page: int | None = Field(None, description="Page number in source document")
    source_section: str | None = Field(None, description="Section name in document")
    source_quote: str | None = Field(None, description="Direct quote from document")

    def format_citation(self) -> str:
        """Format citation for display."""
        parts = []
        if self.source_page:
            parts.append(f"p.{self.source_page}")
        if self.source_section:
            parts.append(self.source_section)
        if parts:
            return f"[SBC {', '.join(parts)}]"
        return ""


# =============================================================================
# Plan Info
# =============================================================================


class PlanInfo(BaseModel):
    """Basic plan identification information."""

    plan_name: str | None = Field(None, description="Name of the plan")
    plan_year: str | None = Field(None, description="Plan year (YYYY)")
    payer_name: str | None = Field(None, description="Insurance company name")
    plan_type: PlanType | None = Field(None, description="Type of plan")
    state: str | None = Field(None, description="State (2-letter code)")
    effective_date: date | None = Field(None, description="Plan effective date")
    termination_date: date | None = Field(None, description="Plan termination date")


# =============================================================================
# Cost Sharing
# =============================================================================


class Deductibles(BaseModel):
    """Deductible amounts."""

    individual_in_network: float | None = Field(None, ge=0)
    individual_out_of_network: float | None = Field(None, ge=0)
    family_in_network: float | None = Field(None, ge=0)
    family_out_of_network: float | None = Field(None, ge=0)
    source_page: int | None = None


class OutOfPocketMax(BaseModel):
    """Out-of-pocket maximum amounts."""

    individual_in_network: float | None = Field(None, ge=0)
    individual_out_of_network: float | None = Field(None, ge=0)
    family_in_network: float | None = Field(None, ge=0)
    family_out_of_network: float | None = Field(None, ge=0)
    source_page: int | None = None


# =============================================================================
# Prior Authorization Rules
# =============================================================================


class PriorAuthRule(BaseModel):
    """Rule for prior authorization requirements."""

    service_category: str = Field(..., description="Category of service")
    pa_required: bool = Field(..., description="Whether PA is required")
    conditions: str | None = Field(None, description="Conditions when PA is required")
    source_page: int | None = None
    source_section: str | None = None
    source_quote: str | None = None

    def get_citation(self) -> SourceCitation:
        """Get citation for this rule."""
        return SourceCitation(
            source_page=self.source_page,
            source_section=self.source_section,
            source_quote=self.source_quote,
        )


# =============================================================================
# Medical Necessity
# =============================================================================


class MedicalNecessityCriteria(BaseModel):
    """Medical necessity definition and criteria."""

    definition: str | None = Field(None, description="How plan defines medical necessity")
    criteria: list[str] = Field(default_factory=list, description="Specific criteria")
    source_page: int | None = None
    source_section: str | None = None
    source_quote: str | None = None

    def get_citation(self) -> SourceCitation:
        """Get citation for this criteria."""
        return SourceCitation(
            source_page=self.source_page,
            source_section=self.source_section,
            source_quote=self.source_quote,
        )


# =============================================================================
# Exclusions
# =============================================================================


class Exclusion(BaseModel):
    """Service excluded from coverage."""

    exclusion: str = Field(..., description="What is excluded")
    category: str | None = Field(None, description="Category (e.g., 'cosmetic')")
    exceptions: str | None = Field(None, description="Any exceptions to the exclusion")
    source_page: int | None = None
    source_section: str | None = None
    source_quote: str | None = None

    def get_citation(self) -> SourceCitation:
        """Get citation for this exclusion."""
        return SourceCitation(
            source_page=self.source_page,
            source_section=self.source_section,
            source_quote=self.source_quote,
        )


# =============================================================================
# Appeal Rights
# =============================================================================


class AppealRights(BaseModel):
    """Appeal process details from plan document."""

    internal_appeal_levels: int | None = Field(None, ge=1, description="Number of levels")
    internal_appeal_deadline_days: int | None = Field(
        None, description="Days to file internal appeal"
    )
    expedited_appeal_available: bool | None = None
    expedited_criteria: str | None = Field(
        None, description="When expedited appeal is available"
    )
    external_review_available: bool | None = None
    external_review_deadline_days: int | None = None
    appeal_address: str | None = None
    appeal_phone: str | None = None
    appeal_fax: str | None = None
    source_page: int | None = None
    source_section: str | None = None

    def get_citation(self) -> SourceCitation:
        """Get citation for appeal rights."""
        return SourceCitation(
            source_page=self.source_page,
            source_section=self.source_section,
        )


# =============================================================================
# Timely Filing
# =============================================================================


class TimelyFiling(BaseModel):
    """Timely filing requirements."""

    initial_claim_days: int | None = Field(None, description="Days to file initial claim")
    corrected_claim_days: int | None = Field(None, description="Days to file correction")
    source_page: int | None = None


# =============================================================================
# Extraction Quality
# =============================================================================


class ExtractionQuality(BaseModel):
    """Quality metrics for the extraction."""

    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Overall confidence")
    pages_processed: int = Field(0, ge=0)
    sections_identified: int = Field(0, ge=0)
    warnings: list[str] = Field(default_factory=list)


# =============================================================================
# Main PlanRules Model
# =============================================================================


class PlanRules(BaseModel):
    """Extracted rules from SBC/EOC documents (Verified mode)."""

    case_id: str = Field(..., description="Associated case ID")
    source_document: str = Field(..., description="Filename of source SBC/EOC")
    document_type: PlanDocumentType = Field(
        PlanDocumentType.SBC, description="Type of plan document"
    )
    extracted_at: datetime = Field(..., description="Extraction timestamp")

    # Plan identification
    plan_info: PlanInfo = Field(default_factory=PlanInfo)

    # Cost sharing
    deductibles: Deductibles | None = None
    out_of_pocket_max: OutOfPocketMax | None = None

    # Coverage rules
    prior_authorization_rules: list[PriorAuthRule] = Field(default_factory=list)
    medical_necessity_criteria: list[MedicalNecessityCriteria] = Field(default_factory=list)
    exclusions: list[Exclusion] = Field(default_factory=list)

    # Appeal process
    appeal_rights: AppealRights | None = None

    # Filing requirements
    timely_filing: TimelyFiling | None = None

    # Quality metrics
    extraction_quality: ExtractionQuality = Field(default_factory=ExtractionQuality)

    def get_pa_rule_for_service(self, service_category: str) -> PriorAuthRule | None:
        """Find PA rule for a given service category."""
        service_lower = service_category.lower()
        for rule in self.prior_authorization_rules:
            if service_lower in rule.service_category.lower():
                return rule
        return None

    def get_relevant_exclusions(self, service_description: str) -> list[Exclusion]:
        """Find exclusions that might apply to a service."""
        service_lower = service_description.lower()
        relevant = []
        for exclusion in self.exclusions:
            if any(
                word in service_lower
                for word in exclusion.exclusion.lower().split()
                if len(word) > 3
            ):
                relevant.append(exclusion)
        return relevant

    def has_appeal_deadline(self) -> bool:
        """Check if appeal deadline is specified."""
        return (
            self.appeal_rights is not None
            and self.appeal_rights.internal_appeal_deadline_days is not None
        )

    def get_appeal_deadline_days(self) -> int | None:
        """Get appeal deadline in days."""
        if self.appeal_rights:
            return self.appeal_rights.internal_appeal_deadline_days
        return None
