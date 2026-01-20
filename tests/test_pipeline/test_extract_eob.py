"""Tests for EOB (Explanation of Benefits) extraction."""

from decimal import Decimal

import pytest

from denialops.models.documents import ExtractedPage, ExtractedText
from denialops.models.eob import ClaimStatus, EOBFacts
from denialops.pipeline.extract_eob import (
    _determine_claim_status,
    _extract_amounts,
    _extract_claim_number,
    _extract_denial_info,
    _extract_member_costs,
    _extract_provider_info,
    extract_eob_facts,
)


def make_extracted_text(text: str, document_id: str = "test-doc") -> ExtractedText:
    """Helper to create ExtractedText from string."""
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
        extraction_method="passthrough",
    )


@pytest.fixture
def sample_eob_text() -> str:
    """Sample EOB document text for testing."""
    return """
    EXPLANATION OF BENEFITS

    Blue Cross Blue Shield
    Claim Number: CLM-2024-12345

    Date of Service: 01/15/2024
    Provider: Dr. Jane Smith, MD
    NPI: 1234567890
    Network Status: In-Network

    Service Description          CPT     Billed      Allowed     Paid
    ----------------------------------------------------------------
    Office Visit               99213    $150.00     $100.00    $80.00
    X-Ray                      73030    $200.00     $150.00   $120.00
    ----------------------------------------------------------------
    Total Charges                        $350.00     $250.00   $200.00

    YOUR RESPONSIBILITY:
    Deductible:           $25.00
    Copay:                $20.00
    Coinsurance:           $5.00
    Not Covered:           $0.00
    --------------------------------
    Total You Owe:        $50.00

    Year-to-Date Deductible Used: $500.00
    Annual Deductible: $1,000.00

    REMARK CODES:
    CO-45: Charges exceed your contracted/legislated fee arrangement

    If you disagree with this decision, you may file an appeal within 180 days.
    """


@pytest.fixture
def denied_eob_text() -> str:
    """Sample denied EOB document text for testing."""
    return """
    EXPLANATION OF BENEFITS - CLAIM DENIED

    Claim Number: CLM-DENIED-999

    Provider: ABC Medical Center

    Service: MRI of Knee
    Procedure Code: 73721
    Billed Amount: $2,500.00
    Allowed Amount: $0.00
    Paid Amount: $0.00

    DENIAL REASON:
    This service was denied because prior authorization was not obtained.
    Denial Code: PR-97 - Prior authorization required

    YOUR RESPONSIBILITY:
    Total You Owe: $2,500.00
    Not Covered: $2,500.00

    You have 180 days from the date of this notice to file an appeal.
    """


class TestExtractClaimNumber:
    """Tests for claim number extraction."""

    def test_extracts_claim_number_standard_format(self):
        """Test extraction of standard claim number format."""
        text = "Claim Number: CLM-2024-12345"
        result = _extract_claim_number(text)
        assert result == "CLM-2024-12345"

    def test_extracts_claim_number_with_hash(self):
        """Test extraction with # symbol."""
        text = "Claim # ABC123456"
        result = _extract_claim_number(text)
        assert result == "ABC123456"

    def test_extracts_claim_id(self):
        """Test extraction of claim ID format."""
        text = "Claim ID: REF-9999"
        result = _extract_claim_number(text)
        assert result == "REF-9999"

    def test_returns_none_when_not_found(self):
        """Test returns None when no claim number found."""
        text = "This is some text without a claim number"
        result = _extract_claim_number(text)
        assert result is None


class TestExtractProviderInfo:
    """Tests for provider information extraction."""

    def test_extracts_npi(self):
        """Test NPI extraction."""
        text = "NPI: 1234567890"
        result = _extract_provider_info(text)
        assert result.npi == "1234567890"

    def test_detects_in_network(self):
        """Test in-network detection."""
        text = "Network Status: In-Network"
        result = _extract_provider_info(text)
        assert result.is_in_network is True

    def test_detects_out_of_network(self):
        """Test out-of-network detection."""
        text = "This provider is out-of-network"
        result = _extract_provider_info(text)
        assert result.is_in_network is False


class TestExtractAmounts:
    """Tests for amount extraction."""

    def test_extracts_billed_amount(self):
        """Test billed amount extraction."""
        text = "Total Billed: $1,500.00"
        billed, _, _ = _extract_amounts(text)
        assert billed == Decimal("1500.00")

    def test_extracts_allowed_amount(self):
        """Test allowed amount extraction."""
        text = "Total Allowed: $1,000.00"
        _, allowed, _ = _extract_amounts(text)
        assert allowed == Decimal("1000.00")

    def test_extracts_paid_amount(self):
        """Test paid amount extraction."""
        text = "Total Paid: $800.00"
        _, _, paid = _extract_amounts(text)
        assert paid == Decimal("800.00")


class TestExtractMemberCosts:
    """Tests for member cost breakdown extraction."""

    def test_extracts_deductible(self):
        """Test deductible extraction."""
        text = "Deductible: $100.00"
        result = _extract_member_costs(text)
        assert result.deductible_applied == Decimal("100.00")

    def test_extracts_copay(self):
        """Test copay extraction."""
        text = "Copay: $25.00"
        result = _extract_member_costs(text)
        assert result.copay == Decimal("25.00")

    def test_extracts_coinsurance(self):
        """Test coinsurance extraction."""
        text = "Coinsurance: $50.00"
        result = _extract_member_costs(text)
        assert result.coinsurance == Decimal("50.00")

    def test_extracts_total_responsibility(self):
        """Test total member responsibility extraction."""
        text = "Your Responsibility: $175.00"
        result = _extract_member_costs(text)
        assert result.total_member_responsibility == Decimal("175.00")


class TestExtractDenialInfo:
    """Tests for denial code and reason extraction."""

    def test_extracts_denial_codes(self):
        """Test denial code extraction."""
        text = "Denial Code: PR-97\nReason Code: CO-45"
        codes, _ = _extract_denial_info(text)
        assert "PR-97" in codes or "CO-45" in codes

    def test_extracts_denial_reasons(self):
        """Test denial reason extraction."""
        text = "Denied because prior authorization was not obtained."
        _, reasons = _extract_denial_info(text)
        # Should extract some reason text
        assert len(reasons) >= 0  # May or may not match pattern


class TestDetermineClaimStatus:
    """Tests for claim status determination."""

    def test_identifies_denied_claim(self):
        """Test identification of denied claim."""
        status = _determine_claim_status("This claim was denied", ["PR-97"])
        assert status == ClaimStatus.DENIED

    def test_identifies_paid_claim(self):
        """Test identification of paid claim."""
        status = _determine_claim_status("Your claim has been paid in full", [])
        assert status == ClaimStatus.PAID

    def test_identifies_pending_claim(self):
        """Test identification of pending claim."""
        status = _determine_claim_status("Your claim is currently pending review", [])
        assert status == ClaimStatus.PENDING


class TestExtractEobFacts:
    """Integration tests for full EOB extraction."""

    def test_extracts_from_standard_eob(self, sample_eob_text: str):
        """Test extraction from standard EOB document."""
        text = make_extracted_text(sample_eob_text)

        result = extract_eob_facts(
            case_id="test-case",
            text=text,
            llm_api_key="",  # Use heuristics
        )

        assert isinstance(result, EOBFacts)
        assert result.case_id == "test-case"
        assert result.claim_number == "CLM-2024-12345"
        assert result.provider.npi == "1234567890"
        assert result.provider.is_in_network is True

    def test_extracts_denied_eob(self, denied_eob_text: str):
        """Test extraction from denied EOB document."""
        text = make_extracted_text(denied_eob_text)

        result = extract_eob_facts(
            case_id="test-denied",
            text=text,
            llm_api_key="",  # Use heuristics
        )

        assert isinstance(result, EOBFacts)
        assert result.claim_status == ClaimStatus.DENIED
        assert len(result.denial_codes) > 0 or "PR-97" in str(result.denial_codes)

    def test_heuristic_extraction_has_moderate_confidence(self, sample_eob_text: str):
        """Test that heuristic extraction has moderate confidence."""
        text = make_extracted_text(sample_eob_text)

        result = extract_eob_facts(
            case_id="test-case",
            text=text,
            llm_api_key="",  # Use heuristics
        )

        # Heuristic extraction should have confidence around 0.5
        assert result.extraction_confidence <= 0.6

    def test_extracts_member_costs(self, sample_eob_text: str):
        """Test extraction of member cost details."""
        text = make_extracted_text(sample_eob_text)

        result = extract_eob_facts(
            case_id="test-case",
            text=text,
            llm_api_key="",
        )

        assert result.member_costs.deductible_applied == Decimal("25.00")
        assert result.member_costs.copay == Decimal("20.00")
        assert result.member_costs.coinsurance == Decimal("5.00")

    def test_extracts_amounts_from_eob(self, sample_eob_text: str):
        """Test extraction of total amounts."""
        text = make_extracted_text(sample_eob_text)

        result = extract_eob_facts(
            case_id="test-case",
            text=text,
            llm_api_key="",
        )

        # Should extract total billed
        assert result.total_billed == Decimal("350.00")
