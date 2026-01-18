"""Pydantic models for action plan generation."""

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field

from denialops.models.route import RouteType


class ResponsibleParty(str, Enum):
    """Who should take an action."""

    PATIENT = "patient"
    PROVIDER = "provider"
    BILLING_STAFF = "billing_staff"
    CLINICIAN = "clinician"


class EvidencePriority(str, Enum):
    """Priority level for evidence items."""

    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


class ImportanceLevel(str, Enum):
    """Importance level for missing info."""

    CRITICAL = "critical"
    HELPFUL = "helpful"
    NICE_TO_HAVE = "nice_to_have"


class SuccessLikelihood(str, Enum):
    """Estimated likelihood of success."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class DocumentTypeGenerated(str, Enum):
    """Types of documents that can be generated."""

    APPEAL_LETTER = "appeal_letter"
    RECONSIDERATION_REQUEST = "reconsideration_request"
    PA_CHECKLIST = "pa_checklist"
    RESUBMIT_CHECKLIST = "resubmit_checklist"
    CALL_SCRIPT = "call_script"
    CLINICIAN_LETTER = "clinician_letter"
    FAX_COVER = "fax_cover"


class ActionSummary(BaseModel):
    """Summary of the situation and recommendation."""

    situation: str = Field(..., description="Plain-language explanation of denial")
    recommendation: str = Field(..., description="High-level recommended action")
    success_likelihood: SuccessLikelihood = Field(
        SuccessLikelihood.UNKNOWN, description="Estimated likelihood of success"
    )
    success_factors: list[str] = Field(
        default_factory=list, description="Factors that would increase success"
    )


class ActionStep(BaseModel):
    """A step in the action plan."""

    step_number: int = Field(..., ge=1, description="Step number")
    action: str = Field(..., description="Brief action title")
    description: str = Field(..., description="Detailed instructions")
    responsible_party: ResponsibleParty | None = Field(
        None, description="Who should take this action"
    )
    deadline: date | None = Field(None, description="Recommended completion date")
    deadline_source: str | None = Field(None, description="Where deadline came from")
    documents_needed: list[str] = Field(
        default_factory=list, description="Documents required for this step"
    )
    templates_available: list[str] = Field(
        default_factory=list, description="Generated templates for this step"
    )


class Timeline(BaseModel):
    """Key dates and deadlines."""

    appeal_deadline: date | None = None
    days_remaining: int | None = Field(None, description="Days until appeal deadline")
    recommended_submission_date: date | None = Field(None, description="Recommended date to submit")
    expected_response_time: str | None = Field(None, description="Typical response time")


class EvidenceItem(BaseModel):
    """An item in the evidence checklist."""

    item: str = Field(..., description="Evidence item description")
    priority: EvidencePriority = Field(..., description="Priority level")
    source: str | None = Field(None, description="Where to obtain this evidence")
    purpose: str | None = Field(None, description="Why this evidence helps")


class MissingInfoRequest(BaseModel):
    """Information needed from user to improve the plan."""

    question: str = Field(..., description="Question to ask the user")
    field: str = Field(..., description="Which case_facts field this would populate")
    importance: ImportanceLevel = Field(..., description="How important this info is")


class GeneratedDocument(BaseModel):
    """A document generated as part of the plan."""

    filename: str = Field(..., description="Filename of the document")
    document_type: DocumentTypeGenerated = Field(..., description="Type of document")
    description: str | None = Field(None, description="What this document is for")


class Citation(BaseModel):
    """A policy citation (Verified mode)."""

    claim: str = Field(..., description="The claim being made")
    source: str = Field(..., description="Document and section reference")
    quote: str | None = Field(None, description="Direct quote if available")


class Assumption(BaseModel):
    """An assumption made due to missing information."""

    assumption: str = Field(..., description="The assumption")
    impact: str = Field(..., description="How this assumption affects the plan")
    how_to_verify: str | None = Field(None, description="How user can verify/correct")


DEFAULT_DISCLAIMERS = [
    "This is not legal advice. Consult an attorney for legal questions.",
    "This is not medical advice. Consult your healthcare provider for medical questions.",
    "Plan-specific coverage details may vary. Verify with your insurance company.",
]


class ActionPlan(BaseModel):
    """Generated action plan for resolving a denial."""

    case_id: str = Field(..., description="Case ID")
    generated_at: datetime = Field(..., description="When plan was generated")
    route: RouteType = Field(..., description="Route this plan addresses")
    mode: str = Field(..., pattern=r"^(fast|verified)$", description="Mode used")

    # Content
    summary: ActionSummary = Field(..., description="Situation summary")
    steps: list[ActionStep] = Field(..., min_length=1, description="Ordered steps")
    timeline: Timeline | None = None
    evidence_checklist: list[EvidenceItem] = Field(default_factory=list)
    missing_info_requests: list[MissingInfoRequest] = Field(default_factory=list)

    # Generated outputs
    generated_documents: list[GeneratedDocument] = Field(default_factory=list)

    # Grounding
    citations: list[Citation] = Field(
        default_factory=list, description="Policy citations (Verified mode)"
    )
    assumptions: list[Assumption] = Field(default_factory=list)
    disclaimers: list[str] = Field(default_factory=lambda: DEFAULT_DISCLAIMERS.copy())
