"""Tests for plan rules extraction from SBC documents."""

from denialops.models.documents import ExtractedPage, ExtractedText
from denialops.models.plan_rules import (
    AppealRights,
    PlanDocumentType,
    PlanRules,
    PriorAuthRule,
)
from denialops.pipeline.extract_plan_rules import (
    _parse_llm_response,
    extract_plan_rules,
)


def make_extracted_text(document_id: str, text: str) -> ExtractedText:
    """Helper to create ExtractedText with all required fields."""
    return ExtractedText(
        document_id=document_id,
        total_pages=1,
        pages=[
            ExtractedPage(
                page_number=1,
                text=text,
                char_offset_start=0,
                char_offset_end=len(text),
            )
        ],
        full_text=text,
        extraction_method="test",
    )


class TestExtractPlanRulesHeuristics:
    """Test heuristic-based extraction (fallback when no LLM)."""

    def test_extracts_deductible(self, sample_sbc_text: str):
        """Test that deductible is extracted from SBC text."""
        text = make_extracted_text("test-sbc.pdf", sample_sbc_text)

        result = extract_plan_rules(
            case_id="test-case",
            text=text,
            llm_api_key="",  # No LLM, use heuristics
        )

        assert result is not None
        assert result.case_id == "test-case"
        assert result.source_document == "test-sbc.pdf"
        assert result.document_type == PlanDocumentType.SBC

    def test_extracts_prior_auth_mention(self, sample_sbc_text: str):
        """Test that prior authorization is detected."""
        text = make_extracted_text("test-sbc.pdf", sample_sbc_text)

        result = extract_plan_rules(
            case_id="test-case",
            text=text,
            llm_api_key="",
        )

        # Should detect prior auth mention
        assert len(result.prior_authorization_rules) > 0

    def test_extracts_appeal_deadline(self, sample_sbc_text: str):
        """Test that appeal deadline is extracted."""
        text = make_extracted_text("test-sbc.pdf", sample_sbc_text)

        result = extract_plan_rules(
            case_id="test-case",
            text=text,
            llm_api_key="",
        )

        # Should extract appeal deadline (180 days from sample)
        assert result.appeal_rights is not None
        assert result.appeal_rights.internal_appeal_deadline_days == 180

    def test_heuristic_extraction_has_low_confidence(self, sample_sbc_text: str):
        """Test that heuristic extraction is marked as low confidence."""
        text = make_extracted_text("test-sbc.pdf", sample_sbc_text)

        result = extract_plan_rules(
            case_id="test-case",
            text=text,
            llm_api_key="",
        )

        assert result.extraction_quality.confidence <= 0.5
        assert "heuristics" in result.extraction_quality.warnings[0].lower()


class TestParseLLMResponse:
    """Test LLM response parsing."""

    def test_parses_json_directly(self):
        """Test parsing direct JSON response."""
        response = '{"plan_info": {"payer_name": "Test Insurance"}}'
        result = _parse_llm_response(response)
        assert result["plan_info"]["payer_name"] == "Test Insurance"

    def test_parses_json_in_markdown_code_block(self):
        """Test parsing JSON wrapped in markdown code block."""
        response = """```json
{"plan_info": {"payer_name": "Test Insurance"}}
```"""
        result = _parse_llm_response(response)
        assert result["plan_info"]["payer_name"] == "Test Insurance"

    def test_parses_json_in_plain_code_block(self):
        """Test parsing JSON wrapped in plain code block."""
        response = """```
{"plan_info": {"payer_name": "Test Insurance"}}
```"""
        result = _parse_llm_response(response)
        assert result["plan_info"]["payer_name"] == "Test Insurance"


class TestPlanRulesModel:
    """Test PlanRules model methods."""

    def test_get_pa_rule_for_service(self):
        """Test finding PA rule for a service category."""
        from datetime import datetime, timezone

        rules = PlanRules(
            case_id="test",
            source_document="test.pdf",
            extracted_at=datetime.now(timezone.utc),
            prior_authorization_rules=[
                PriorAuthRule(
                    service_category="Advanced Imaging (MRI, CT)",
                    pa_required=True,
                    conditions="All non-emergency imaging",
                ),
                PriorAuthRule(
                    service_category="Inpatient Hospital",
                    pa_required=True,
                    conditions="Non-emergency admissions",
                ),
            ],
        )

        # Should find imaging rule
        imaging_rule = rules.get_pa_rule_for_service("MRI")
        assert imaging_rule is not None
        assert "imaging" in imaging_rule.service_category.lower()

        # Should find hospital rule
        hospital_rule = rules.get_pa_rule_for_service("inpatient")
        assert hospital_rule is not None

        # Should not find rule for uncovered service
        dental_rule = rules.get_pa_rule_for_service("dental")
        assert dental_rule is None

    def test_has_appeal_deadline(self):
        """Test appeal deadline detection."""
        from datetime import datetime, timezone

        # Without appeal rights
        rules_no_appeal = PlanRules(
            case_id="test",
            source_document="test.pdf",
            extracted_at=datetime.now(timezone.utc),
        )
        assert rules_no_appeal.has_appeal_deadline() is False

        # With appeal rights but no deadline
        rules_no_deadline = PlanRules(
            case_id="test",
            source_document="test.pdf",
            extracted_at=datetime.now(timezone.utc),
            appeal_rights=AppealRights(),
        )
        assert rules_no_deadline.has_appeal_deadline() is False

        # With appeal deadline
        rules_with_deadline = PlanRules(
            case_id="test",
            source_document="test.pdf",
            extracted_at=datetime.now(timezone.utc),
            appeal_rights=AppealRights(internal_appeal_deadline_days=180),
        )
        assert rules_with_deadline.has_appeal_deadline() is True
        assert rules_with_deadline.get_appeal_deadline_days() == 180

    def test_get_relevant_exclusions(self):
        """Test finding relevant exclusions for a service."""
        from datetime import datetime, timezone

        from denialops.models.plan_rules import Exclusion

        rules = PlanRules(
            case_id="test",
            source_document="test.pdf",
            extracted_at=datetime.now(timezone.utc),
            exclusions=[
                Exclusion(
                    exclusion="Cosmetic surgery",
                    category="cosmetic",
                    exceptions="Reconstruction after mastectomy",
                ),
                Exclusion(
                    exclusion="Experimental treatments",
                    category="experimental",
                ),
                Exclusion(
                    exclusion="Weight loss surgery",
                    category="other",
                    exceptions="BMI over 40 with comorbidities",
                ),
            ],
        )

        # Should find cosmetic exclusion
        cosmetic = rules.get_relevant_exclusions("cosmetic rhinoplasty")
        assert len(cosmetic) > 0

        # Should find weight loss exclusion
        weight = rules.get_relevant_exclusions("gastric bypass surgery for weight loss")
        assert len(weight) > 0

        # Should not find exclusion for covered service
        covered = rules.get_relevant_exclusions("appendectomy")
        assert len(covered) == 0
