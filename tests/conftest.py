"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from denialops.main import app
from denialops.utils.storage import CaseStorage


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture
def temp_storage() -> CaseStorage:
    """Create a temporary storage for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield CaseStorage(Path(tmpdir))


@pytest.fixture
def sample_denial_text() -> str:
    """Sample denial letter text for testing."""
    return """
BLUE CROSS BLUE SHIELD OF NORTH CAROLINA
P.O. Box 2291
Durham, NC 27702

January 15, 2024

John Smith
123 Main Street
Raleigh, NC 27601

Member ID: ABC123456789
Group Number: 98765
Claim Number: CLM2024011500001

RE: Explanation of Benefits - Claim Denial

Dear John Smith,

We have reviewed your claim for MRI of the lumbar spine (CPT code 72148)
performed on January 5, 2024, by Dr. Jane Wilson at Raleigh Medical Center.

YOUR CLAIM HAS BEEN DENIED

Reason for Denial: Prior authorization was not obtained for this service.
According to your plan benefits, MRI services require prior authorization
before the service is performed.

Denial Code: CO-197

Amount Billed: $2,500.00
Amount Allowed: $0.00
Patient Responsibility: $2,500.00

YOUR APPEAL RIGHTS

You have the right to appeal this decision. To file an appeal, you must
submit your request within 180 days of this notice.

To file an appeal:
Phone: (800) 555-1234
Fax: (800) 555-4321
Mail: BCBS Appeals Department, P.O. Box 2291, Durham, NC 27702

Please include:
- A copy of this denial letter
- A letter explaining why you believe the claim should be paid
- Any supporting documentation

Sincerely,
Claims Department
Blue Cross Blue Shield of North Carolina
"""


@pytest.fixture
def sample_medical_necessity_denial() -> str:
    """Sample medical necessity denial letter."""
    return """
UNITED HEALTHCARE
P.O. Box 740800
Atlanta, GA 30374

February 1, 2024

Jane Doe
456 Oak Avenue
Atlanta, GA 30301

Member ID: UHC987654321
Claim Number: CLM2024020100002

RE: Claim Denial Notice

Dear Jane Doe,

This letter is regarding your claim for Humira (adalimumab) injection
prescribed by Dr. Robert Johnson for treatment of rheumatoid arthritis.
Date of service: January 20, 2024.

DENIAL DETERMINATION

After medical review, we have determined that this service does not meet
the criteria for medical necessity under your plan. The documentation
provided does not demonstrate that conventional DMARD therapy was tried
and failed before initiating biologic therapy.

ICD-10 Code: M05.79 (Rheumatoid arthritis)
CPT Code: J0135

Amount Billed: $6,500.00
Your Responsibility: $6,500.00

APPEAL INFORMATION

You may appeal this decision within 60 days. Include:
- Medical records showing treatment history
- Letter of medical necessity from your physician
- Documentation of prior treatments tried

Appeals Address: UHC Appeals, P.O. Box 740800, Atlanta, GA 30374
Appeals Phone: (800) 555-9876
Appeals Fax: (800) 555-6789

Sincerely,
Medical Review Department
United Healthcare
"""
