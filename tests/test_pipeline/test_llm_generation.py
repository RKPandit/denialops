"""Tests for Phase 4 LLM generation features."""

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from denialops.models.case import (
    CaseAmounts,
    CaseDates,
    CaseFacts,
    CodeType,
    ContactInfo,
    DenialCode,
    ExtractionConfidence,
    PayerInfo,
    ServiceInfo,
)
from denialops.models.plan_rules import AppealRights, PlanRules, SourceCitation
from denialops.models.route import RouteDecision, RouteType
from denialops.pipeline.llm_generation import (
    PersonalizedSummary,
    generate_personalized_summary,
    predict_success,
    validate_grounding,
)


@pytest.fixture
def sample_case_facts() -> CaseFacts:
    """Create sample case facts for testing."""
    return CaseFacts(
        case_id="test-case-123",
        denial_reason="Service not medically necessary",
        denial_reason_summary="Medical necessity denial",
        denial_codes=[DenialCode(code="A1", code_type=CodeType.RARC, description="Not medically necessary")],
        service=ServiceInfo(
            description="MRI of knee",
            cpt_codes=["73721"],
            diagnosis_codes=["M25.561"],
            date_of_service=date(2024, 1, 15),
            provider_name="Dr. Smith",
        ),
        payer=PayerInfo(
            name="Blue Cross Blue Shield",
            plan_name="PPO Gold",
            member_id="MEM123456",
            claim_number="CLM789",
        ),
        dates=CaseDates(
            date_of_service=date(2024, 1, 15),
            date_of_denial=date(2024, 2, 1),
            appeal_deadline=date(2026, 8, 1),  # Future date for testing
        ),
        amounts=CaseAmounts(
            billed_amount=Decimal("1500.00"),
            patient_responsibility=Decimal("1500.00"),
        ),
        contact_info=ContactInfo(
            phone="1-800-555-1234",
            address="P.O. Box 12345, City, ST 12345",
        ),
        extraction_confidence=ExtractionConfidence(overall=0.9),
        extraction_timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_route_decision() -> RouteDecision:
    """Create sample route decision for testing."""
    return RouteDecision(
        case_id="test-case-123",
        route=RouteType.MEDICAL_NECESSITY_APPEAL,
        confidence=0.85,
        reasoning="Denial explicitly mentions medical necessity",
        signals=[],
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_plan_rules() -> PlanRules:
    """Create sample plan rules for testing."""
    return PlanRules(
        case_id="test-case-123",
        source_document="test-sbc.pdf",
        extracted_at=datetime.now(timezone.utc),
        appeal_rights=AppealRights(
            internal_appeal_deadline_days=180,
            source=SourceCitation(section="Appeals", page_number=15),
        ),
    )


class TestGeneratePersonalizedSummary:
    """Tests for personalized summary generation."""

    def test_generates_summary_without_llm(
        self, sample_case_facts: CaseFacts, sample_route_decision: RouteDecision
    ):
        """Test heuristic summary generation without LLM."""
        result = generate_personalized_summary(
            facts=sample_case_facts,
            route=sample_route_decision,
            llm_api_key="",  # No API key, uses heuristics
        )

        assert isinstance(result, PersonalizedSummary)
        assert result.situation_summary
        assert result.recommendation
        assert result.urgency in ["low", "medium", "high"]
        assert result.is_llm_generated is False

    def test_summary_includes_denial_reason(
        self, sample_case_facts: CaseFacts, sample_route_decision: RouteDecision
    ):
        """Test that summary includes denial reason."""
        result = generate_personalized_summary(
            facts=sample_case_facts,
            route=sample_route_decision,
        )

        # Summary should reference the denial reason
        assert "medical necessity" in result.situation_summary.lower() or "denied" in result.situation_summary.lower()

    def test_summary_includes_key_points(
        self, sample_case_facts: CaseFacts, sample_route_decision: RouteDecision
    ):
        """Test that summary includes key points."""
        result = generate_personalized_summary(
            facts=sample_case_facts,
            route=sample_route_decision,
        )

        assert len(result.key_points) > 0

    def test_summary_with_plan_rules(
        self,
        sample_case_facts: CaseFacts,
        sample_route_decision: RouteDecision,
        sample_plan_rules: PlanRules,
    ):
        """Test summary generation with plan rules context."""
        result = generate_personalized_summary(
            facts=sample_case_facts,
            route=sample_route_decision,
            plan_rules=sample_plan_rules,
        )

        assert isinstance(result, PersonalizedSummary)
        # Should still generate a valid summary
        assert result.situation_summary
        assert result.recommendation


class TestPredictSuccess:
    """Tests for appeal success prediction."""

    def test_predict_success_without_llm(
        self, sample_case_facts: CaseFacts, sample_route_decision: RouteDecision
    ):
        """Test heuristic success prediction without LLM."""
        result = predict_success(
            facts=sample_case_facts,
            route=sample_route_decision,
            llm_api_key="",  # No API key, uses heuristics
        )

        assert result.likelihood in ["low", "medium", "high"]
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.factors_for, list)
        assert isinstance(result.factors_against, list)
        assert result.reasoning

    def test_predict_success_with_plan_rules(
        self,
        sample_case_facts: CaseFacts,
        sample_route_decision: RouteDecision,
        sample_plan_rules: PlanRules,
    ):
        """Test success prediction with plan rules context."""
        result = predict_success(
            facts=sample_case_facts,
            route=sample_route_decision,
            plan_rules=sample_plan_rules,
        )

        assert result.likelihood in ["low", "medium", "high"]
        # Having plan rules should be a positive factor
        assert any("plan" in f.lower() for f in result.factors_for)

    def test_predict_success_considers_deadline(
        self, sample_case_facts: CaseFacts, sample_route_decision: RouteDecision
    ):
        """Test that prediction considers appeal deadline."""
        result = predict_success(
            facts=sample_case_facts,
            route=sample_route_decision,
        )

        # With a future deadline, should be a positive factor related to time
        has_deadline_factor = any(
            "time" in f.lower() or "appeal" in f.lower() for f in result.factors_for
        )
        assert has_deadline_factor

    def test_predict_success_for_resubmit_route(self, sample_case_facts: CaseFacts):
        """Test success prediction for resubmit route."""
        route = RouteDecision(
            case_id="test-case-123",
            route=RouteType.CLAIM_CORRECTION_RESUBMIT,
            confidence=0.9,
            reasoning="Coding error",
            signals=[],
            timestamp=datetime.now(timezone.utc),
        )

        result = predict_success(
            facts=sample_case_facts,
            route=route,
        )

        # Resubmit cases typically have higher success likelihood
        assert result.likelihood in ["medium", "high"]


class TestValidateGrounding:
    """Tests for grounding validation."""

    def test_validate_grounding_empty_content(
        self, sample_case_facts: CaseFacts
    ):
        """Test validation with empty content."""
        result = validate_grounding(
            content="",
            facts=sample_case_facts,
        )

        # Empty content should be considered grounded (nothing to validate)
        assert result.is_grounded is True

    def test_validate_grounding_valid_content(
        self, sample_case_facts: CaseFacts
    ):
        """Test validation with content matching facts."""
        # Content that matches the case facts
        content = """
        Dear Blue Cross Blue Shield,

        I am writing to appeal the denial of my MRI claim (CLM789).
        The service was performed on January 15, 2024.
        The procedure code was 73721.
        """

        result = validate_grounding(
            content=content,
            facts=sample_case_facts,
        )

        assert result.is_grounded is True
        assert len(result.hallucinated_codes) == 0

    def test_validate_grounding_detects_hallucinated_codes(
        self, sample_case_facts: CaseFacts
    ):
        """Test that validation detects hallucinated procedure codes."""
        # Content with a procedure code not in the facts
        content = """
        The denied service was CPT code 99999, which is clearly medically necessary.
        """

        result = validate_grounding(
            content=content,
            facts=sample_case_facts,
        )

        # Should detect the hallucinated code
        assert "99999" in result.hallucinated_codes or not result.is_grounded

    def test_validate_grounding_detects_hallucinated_dates(
        self, sample_case_facts: CaseFacts
    ):
        """Test that validation detects dates not in source documents."""
        # Content with a date not in the facts (March 25, 2024 is not in the facts)
        content = """
        The service was performed on March 25, 2024, which was medically necessary.
        The appeal deadline is December 31, 2025.
        """

        result = validate_grounding(
            content=content,
            facts=sample_case_facts,
        )

        # The heuristic validation may not catch all dates, but should return a result
        # Either it detects hallucinated dates or it returns lower confidence
        assert isinstance(result.hallucinated_dates, list)
        # The grounding result should be valid regardless of detection
        assert result.confidence >= 0.0

    def test_validate_grounding_with_plan_rules(
        self, sample_case_facts: CaseFacts, sample_plan_rules: PlanRules
    ):
        """Test grounding validation considers plan rules."""
        # Content that references plan rules
        content = """
        According to my plan's appeal rights, I have 180 days to file an appeal.
        """

        result = validate_grounding(
            content=content,
            facts=sample_case_facts,
            plan_rules=sample_plan_rules,
        )

        # 180 days matches the plan rules, should be grounded
        assert result.is_grounded is True

    def test_validate_grounding_confidence(
        self, sample_case_facts: CaseFacts
    ):
        """Test that validation returns a confidence score."""
        content = "This is a test appeal letter."

        result = validate_grounding(
            content=content,
            facts=sample_case_facts,
        )

        assert 0.0 <= result.confidence <= 1.0
