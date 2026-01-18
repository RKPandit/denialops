"""Pydantic models for case routing."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class RouteType(str, Enum):
    """Available action routes."""

    PRIOR_AUTH_NEEDED = "prior_auth_needed"
    CLAIM_CORRECTION_RESUBMIT = "claim_correction_resubmit"
    MEDICAL_NECESSITY_APPEAL = "medical_necessity_appeal"


class Urgency(str, Enum):
    """Urgency levels for case handling."""

    STANDARD = "standard"
    EXPEDITED = "expedited"
    URGENT = "urgent"


class RouteSignal(BaseModel):
    """A signal that contributed to the routing decision."""

    signal: str = Field(..., description="The signal detected (e.g., 'prior_auth_mentioned')")
    weight: float = Field(..., ge=-1, le=1, description="Influence on decision (-1 to 1)")
    source: str = Field(..., description="Where signal was found (e.g., 'denial_reason')")
    evidence: str | None = Field(None, description="Text/code that triggered this signal")


class AlternativeRoute(BaseModel):
    """An alternative route that was considered."""

    route: RouteType = Field(..., description="The route considered")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    reason_not_selected: str | None = Field(None, description="Why this wasn't chosen")


class RouteDecision(BaseModel):
    """Routing decision for a denial case."""

    case_id: str = Field(..., description="Case this routing applies to")
    timestamp: datetime = Field(..., description="When routing decision was made")

    # Decision
    route: RouteType = Field(..., description="Selected action route")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in the decision (0-1)")
    reasoning: str = Field(..., min_length=1, description="Explanation of why this route")

    # Supporting info
    signals: list[RouteSignal] = Field(
        default_factory=list, description="Signals that contributed to decision"
    )
    alternative_routes: list[AlternativeRoute] = Field(
        default_factory=list, description="Other routes considered"
    )

    # Recommendations
    requires_verified_mode: bool = Field(
        False, description="Whether this route benefits from Verified mode"
    )
    urgency: Urgency = Field(Urgency.STANDARD, description="Recommended urgency level")
    urgency_reason: str | None = Field(None, description="Why non-standard urgency")
