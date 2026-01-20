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


@pytest.fixture
def sample_sbc_text() -> str:
    """Sample Summary of Benefits and Coverage (SBC) document for testing."""
    return """
SUMMARY OF BENEFITS AND COVERAGE
What this Plan Covers & What You Pay for Covered Services

HealthFirst Insurance Company
Gold PPO Plan
Coverage Period: 01/01/2024 - 12/31/2024

This is only a summary. For complete details, see the Evidence of Coverage.

IMPORTANT QUESTIONS          ANSWERS
--------------------------- ------------------------------------------
What is the overall         $1,500 Individual / $3,000 Family (In-Network)
deductible?                 $3,000 Individual / $6,000 Family (Out-of-Network)

Are there services covered  Yes. Preventive care, prenatal care, and
before you meet your        well-child visits are covered before you
deductible?                 meet your deductible.

Are there other             Yes. $30 copay for primary care visits.
deductibles for specific    $50 copay for specialist visits.
services?                   Copays do not apply to deductible.

What is the out-of-pocket   $6,500 Individual / $13,000 Family (In-Network)
limit for this plan?        $13,000 Individual / $26,000 Family (Out-of-Network)

What is not included in     Premiums, balance-billing charges, and health
the out-of-pocket limit?    care this plan doesn't cover.

Will you pay less if you    Yes. See www.healthfirst.com or call
use a network provider?     1-800-555-0100 for a list of network providers.

Do you need a referral to   No. You can see any specialist without a referral,
see a specialist?           but staying in-network saves you money.


PRIOR AUTHORIZATION REQUIREMENTS
--------------------------------
The following services require prior authorization:

- Inpatient hospital stays (all non-emergency admissions)
- Outpatient surgery at ambulatory surgical centers
- Advanced imaging (MRI, CT scan, PET scan)
- Durable medical equipment (DME) over $500
- Home health care services
- Skilled nursing facility care
- Physical therapy beyond 20 visits per year
- Mental health/substance abuse inpatient treatment
- Specialty medications and biologics

Failure to obtain prior authorization may result in denial of the claim
or reduced benefits (50% reduction in payment).

To request prior authorization:
Phone: 1-800-555-0101
Fax: 1-800-555-0102
Online: www.healthfirst.com/priorauth


MEDICAL NECESSITY
-----------------
A service is considered medically necessary when it is:
1. Required for the diagnosis or treatment of your condition
2. Appropriate for the symptoms and consistent with the diagnosis
3. Not primarily for convenience of patient, physician, or facility
4. The most appropriate level of care that can safely be provided
5. Consistent with generally accepted standards of medical practice
6. Not experimental or investigational

Medical necessity determinations are made by our medical directors
based on clinical evidence and peer-reviewed literature.


EXCLUDED SERVICES & OTHER COVERED SERVICES
------------------------------------------

Services Your Plan Generally DOES NOT Cover:
- Cosmetic surgery (unless medically necessary for reconstruction)
- Dental care (except for accidental injury)
- Vision care (routine eye exams)
- Hearing aids
- Long-term care
- Weight loss programs (non-surgical)
- Infertility treatments
- Experimental or investigational treatments
- Services received outside the United States (except emergencies)
- Acupuncture (except for chronic pain management)

Some services require a waiting period. Please refer to the Evidence
of Coverage for details on waiting periods.


APPEAL RIGHTS
-------------
If we deny coverage for a service, you have the right to appeal.

INTERNAL APPEALS:
You may file up to 2 levels of internal appeal.

Level 1 Appeal: Must be filed within 180 calendar days of the
denial notice. We will respond within 30 days (72 hours for urgent).

Level 2 Appeal: If Level 1 is denied, you may request a second
review within 60 days. We will respond within 30 days.

EXPEDITED APPEALS:
If your situation is urgent (imminent harm to health), you may
request an expedited review. Decisions are made within 72 hours.
Expedited appeals are available when delay could seriously jeopardize
your life, health, or ability to regain maximum function.

EXTERNAL REVIEW:
After exhausting internal appeals, you may request an independent
external review within 4 months. External review is conducted by
an independent review organization (IRO).

HOW TO FILE AN APPEAL:
Mail: HealthFirst Appeals Department
      P.O. Box 12345
      Chicago, IL 60601
Phone: 1-800-555-0103
Fax: 1-800-555-0104
Email: appeals@healthfirst.com

Include with your appeal:
- Copy of denial letter
- Written explanation of why you disagree
- Any supporting medical records or documentation
- Letter from your treating physician (recommended)


TIMELY FILING REQUIREMENTS
--------------------------
Claims must be submitted within:
- Initial claims: 365 days from date of service
- Corrected claims: 365 days from date of original claim denial
- Coordination of Benefits: 365 days from date other insurer processes claim

Late submissions may result in denial of the claim.


QUESTIONS?
----------
Call us at 1-800-555-0100
Visit www.healthfirst.com
TTY: 1-800-555-0105 (hearing impaired)

State: IL
Plan Type: PPO
Effective Date: 01/01/2024
"""
