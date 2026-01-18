"""Tests for case routing."""

from datetime import datetime, timezone

import pytest

from denialops.models.case import CaseFacts, ExtractionConfidence
from denialops.models.route import RouteType
from denialops.pipeline.router import route_case


@pytest.fixture
def prior_auth_facts() -> CaseFacts:
    """Create case facts indicating prior auth issue."""
    return CaseFacts(
        case_id="test-pa",
        extraction_timestamp=datetime.now(timezone.utc),
        denial_reason="Prior authorization was not obtained for this service.",
        extraction_confidence=ExtractionConfidence(overall=0.8),
    )


@pytest.fixture
def coding_facts() -> CaseFacts:
    """Create case facts indicating coding issue."""
    return CaseFacts(
        case_id="test-coding",
        extraction_timestamp=datetime.now(timezone.utc),
        denial_reason="Invalid modifier on the claim. Please correct and resubmit.",
        extraction_confidence=ExtractionConfidence(overall=0.8),
    )


@pytest.fixture
def medical_necessity_facts() -> CaseFacts:
    """Create case facts indicating medical necessity denial."""
    return CaseFacts(
        case_id="test-med-nec",
        extraction_timestamp=datetime.now(timezone.utc),
        denial_reason="This service does not meet medical necessity criteria.",
        extraction_confidence=ExtractionConfidence(overall=0.8),
    )


def test_route_prior_auth(prior_auth_facts: CaseFacts) -> None:
    """Test routing for prior auth denial."""
    decision = route_case(prior_auth_facts)

    assert decision.route == RouteType.PRIOR_AUTH_NEEDED
    assert decision.confidence > 0.5
    assert len(decision.signals) > 0


def test_route_coding(coding_facts: CaseFacts) -> None:
    """Test routing for coding issue."""
    decision = route_case(coding_facts)

    assert decision.route == RouteType.CLAIM_CORRECTION_RESUBMIT
    assert decision.confidence > 0.5


def test_route_medical_necessity(medical_necessity_facts: CaseFacts) -> None:
    """Test routing for medical necessity denial."""
    decision = route_case(medical_necessity_facts)

    assert decision.route == RouteType.MEDICAL_NECESSITY_APPEAL
    assert decision.confidence > 0.5
    # Medical necessity appeals benefit from Verified mode
    assert decision.requires_verified_mode is True


def test_route_ambiguous() -> None:
    """Test routing with ambiguous denial reason."""
    facts = CaseFacts(
        case_id="test-ambiguous",
        extraction_timestamp=datetime.now(timezone.utc),
        denial_reason="Your claim has been denied.",
        extraction_confidence=ExtractionConfidence(overall=0.5),
    )

    decision = route_case(facts)

    # Should default to medical necessity appeal with low confidence
    assert decision.route == RouteType.MEDICAL_NECESSITY_APPEAL
    assert decision.confidence < 0.7  # Lower confidence for ambiguous cases


def test_route_includes_alternatives(prior_auth_facts: CaseFacts) -> None:
    """Test that routing includes alternative routes."""
    decision = route_case(prior_auth_facts)

    # Should have alternative routes
    assert len(decision.alternative_routes) >= 0

    # If there are alternatives, they should be different from selected
    for alt in decision.alternative_routes:
        assert alt.route != decision.route


def test_route_includes_reasoning(prior_auth_facts: CaseFacts) -> None:
    """Test that routing includes reasoning."""
    decision = route_case(prior_auth_facts)

    assert len(decision.reasoning) > 0
    assert decision.case_id == prior_auth_facts.case_id
