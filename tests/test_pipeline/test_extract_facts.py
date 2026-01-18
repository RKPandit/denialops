"""Tests for case facts extraction."""

import pytest

from denialops.models.documents import ExtractedPage, ExtractedText
from denialops.pipeline.extract_facts import (
    _extract_amounts,
    _extract_codes,
    _extract_contact_info,
    _extract_dates,
    extract_case_facts,
)


@pytest.fixture
def extracted_text(sample_denial_text: str) -> ExtractedText:
    """Create extracted text from sample denial."""
    return ExtractedText(
        document_id="test",
        total_pages=1,
        pages=[
            ExtractedPage(
                page_number=1,
                text=sample_denial_text,
                char_offset_start=0,
                char_offset_end=len(sample_denial_text),
            )
        ],
        full_text=sample_denial_text,
        extraction_method="test",
    )


def test_extract_dates(sample_denial_text: str) -> None:
    """Test date extraction."""
    dates = _extract_dates(sample_denial_text)

    # Should find appeal deadline days
    assert dates.appeal_deadline_days == 180


def test_extract_codes(sample_denial_text: str) -> None:
    """Test code extraction."""
    codes = _extract_codes(sample_denial_text)

    # Should find CPT code
    cpt_codes = [c for c in codes if c.code_type.value == "CPT"]
    assert any(c.code == "72148" for c in cpt_codes)


def test_extract_amounts(sample_denial_text: str) -> None:
    """Test amount extraction."""
    amounts = _extract_amounts(sample_denial_text)

    assert amounts.billed_amount == 2500.00
    assert amounts.patient_responsibility == 2500.00


def test_extract_contact_info(sample_denial_text: str) -> None:
    """Test contact info extraction."""
    contact = _extract_contact_info(sample_denial_text)

    assert contact.phone is not None
    assert contact.fax is not None


def test_extract_case_facts(extracted_text: ExtractedText) -> None:
    """Test full case facts extraction."""
    facts = extract_case_facts(
        case_id="test-case",
        text=extracted_text,
    )

    assert facts.case_id == "test-case"
    assert facts.denial_reason is not None
    assert len(facts.denial_reason) > 0

    # Check that payer was identified
    assert facts.payer is not None

    # Check confidence scores
    assert facts.extraction_confidence is not None
    assert 0 <= facts.extraction_confidence.overall <= 1


def test_extract_prior_auth_denial(sample_denial_text: str) -> None:
    """Test that prior auth denial is detected."""
    text = ExtractedText(
        document_id="test",
        total_pages=1,
        pages=[
            ExtractedPage(
                page_number=1,
                text=sample_denial_text,
                char_offset_start=0,
                char_offset_end=len(sample_denial_text),
            )
        ],
        full_text=sample_denial_text,
        extraction_method="test",
    )

    facts = extract_case_facts(case_id="test", text=text)

    # Should detect prior auth in denial reason
    assert "prior authorization" in facts.denial_reason.lower()
