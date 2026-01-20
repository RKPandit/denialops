"""Action plan generation."""

from datetime import date, datetime, timedelta, timezone

from denialops.models.action_plan import (
    ActionPlan,
    ActionStep,
    ActionSummary,
    Assumption,
    DocumentTypeGenerated,
    EvidenceItem,
    EvidencePriority,
    GeneratedDocument,
    ImportanceLevel,
    MissingInfoRequest,
    ResponsibleParty,
    SuccessLikelihood,
    Timeline,
)
from denialops.models.case import CaseFacts
from denialops.models.plan_rules import PlanRules
from denialops.models.route import RouteDecision, RouteType


def generate_action_plan(
    facts: CaseFacts,
    route: RouteDecision,
    mode: str = "fast",
    plan_rules: PlanRules | None = None,
) -> ActionPlan:
    """
    Generate an action plan based on case facts and route.

    Args:
        facts: Extracted case facts
        route: Routing decision
        mode: Processing mode ('fast' or 'verified')
        plan_rules: Optional PlanRules extracted from SBC (for verified mode)

    Returns:
        ActionPlan with steps, evidence checklist, and generated documents
    """
    # Generate route-specific plan
    if route.route == RouteType.PRIOR_AUTH_NEEDED:
        return _generate_pa_plan(facts, route, mode, plan_rules)
    elif route.route == RouteType.CLAIM_CORRECTION_RESUBMIT:
        return _generate_resubmit_plan(facts, route, mode, plan_rules)
    else:  # MEDICAL_NECESSITY_APPEAL
        return _generate_appeal_plan(facts, route, mode, plan_rules)


def _generate_pa_plan(
    facts: CaseFacts,
    route: RouteDecision,
    mode: str,
    plan_rules: PlanRules | None = None,
) -> ActionPlan:
    """Generate action plan for prior authorization route."""
    summary = ActionSummary(
        situation=(
            f"Your claim was denied because prior authorization was not obtained. "
            f"Reason: {facts.denial_reason}"
        ),
        recommendation=(
            "You need to obtain prior authorization and then either request "
            "reconsideration or resubmit the claim."
        ),
        success_likelihood=SuccessLikelihood.MEDIUM,
        success_factors=[
            "Service meets medical necessity criteria",
            "Prior authorization obtained promptly",
            "Complete documentation submitted",
        ],
    )

    steps = [
        ActionStep(
            step_number=1,
            action="Contact your provider",
            description=(
                "Call your healthcare provider's office and inform them that the claim was "
                "denied due to missing prior authorization. Ask them to submit a prior "
                "authorization request on your behalf."
            ),
            responsible_party=ResponsibleParty.PATIENT,
            documents_needed=["Denial letter"],
        ),
        ActionStep(
            step_number=2,
            action="Gather medical records",
            description=(
                "Request copies of medical records that support the medical necessity of "
                "the service. This may include clinical notes, test results, and treatment history."
            ),
            responsible_party=ResponsibleParty.PATIENT,
            documents_needed=["Medical records request form"],
        ),
        ActionStep(
            step_number=3,
            action="Provider submits PA request",
            description=(
                "Your provider will submit the prior authorization request to your insurance. "
                "They should include all supporting documentation for medical necessity."
            ),
            responsible_party=ResponsibleParty.PROVIDER,
            documents_needed=["PA form", "Clinical documentation", "Medical records"],
        ),
        ActionStep(
            step_number=4,
            action="Follow up on PA status",
            description=(
                "Call the insurance company to check on the status of the prior authorization. "
                "Use the call script provided to guide your conversation."
            ),
            responsible_party=ResponsibleParty.PATIENT,
            templates_available=["call_script.md"],
        ),
        ActionStep(
            step_number=5,
            action="Request claim reconsideration",
            description=(
                "Once prior authorization is approved, contact your insurance to request "
                "reconsideration of the original claim with the new PA number."
            ),
            responsible_party=ResponsibleParty.PATIENT,
        ),
    ]

    timeline = _build_timeline(facts)

    evidence_checklist = [
        EvidenceItem(
            item="Prior authorization approval letter",
            priority=EvidencePriority.REQUIRED,
            source="Insurance company after PA approved",
            purpose="Proves authorization was obtained",
        ),
        EvidenceItem(
            item="Medical records supporting necessity",
            priority=EvidencePriority.REQUIRED,
            source="Healthcare provider",
            purpose="Demonstrates medical necessity for service",
        ),
        EvidenceItem(
            item="Original denial letter",
            priority=EvidencePriority.REQUIRED,
            source="Already received",
            purpose="Reference for appeal/reconsideration",
        ),
    ]

    missing_info = _identify_missing_info_requests(facts)
    assumptions = _identify_assumptions(facts, mode, plan_rules)

    return ActionPlan(
        case_id=facts.case_id,
        generated_at=datetime.now(timezone.utc),
        route=route.route,
        mode=mode,
        summary=summary,
        steps=steps,
        timeline=timeline,
        evidence_checklist=evidence_checklist,
        missing_info_requests=missing_info,
        generated_documents=[
            GeneratedDocument(
                filename="pa_checklist.md",
                document_type=DocumentTypeGenerated.PA_CHECKLIST,
                description="Checklist for prior authorization process",
            ),
            GeneratedDocument(
                filename="call_script.md",
                document_type=DocumentTypeGenerated.CALL_SCRIPT,
                description="Script for calling insurance company",
            ),
        ],
        assumptions=assumptions,
    )


def _generate_resubmit_plan(
    facts: CaseFacts,
    route: RouteDecision,
    mode: str,
    plan_rules: PlanRules | None = None,
) -> ActionPlan:
    """Generate action plan for claim correction/resubmission route."""
    summary = ActionSummary(
        situation=(
            f"Your claim was denied due to a coding or billing issue. Reason: {facts.denial_reason}"
        ),
        recommendation=(
            "The claim needs to be corrected and resubmitted. Contact your provider's "
            "billing department to address the issue."
        ),
        success_likelihood=SuccessLikelihood.HIGH,
        success_factors=[
            "Billing error is corrected",
            "Claim resubmitted within timely filing",
            "All required information included",
        ],
    )

    steps = [
        ActionStep(
            step_number=1,
            action="Contact provider billing",
            description=(
                "Call your healthcare provider's billing department. Inform them of the "
                "denial and ask them to review and correct the claim."
            ),
            responsible_party=ResponsibleParty.PATIENT,
            documents_needed=["Denial letter", "EOB if available"],
        ),
        ActionStep(
            step_number=2,
            action="Identify the error",
            description=(
                "Work with the billing department to identify the specific error. "
                "Common issues include incorrect procedure codes, missing modifiers, "
                "wrong diagnosis codes, or NPI errors."
            ),
            responsible_party=ResponsibleParty.BILLING_STAFF,
        ),
        ActionStep(
            step_number=3,
            action="Correct and resubmit",
            description=(
                "The billing department will correct the claim and resubmit it to the "
                "insurance company. Ask for a confirmation number."
            ),
            responsible_party=ResponsibleParty.BILLING_STAFF,
        ),
        ActionStep(
            step_number=4,
            action="Follow up",
            description=(
                "After 2-3 weeks, call the insurance company to verify the corrected "
                "claim was received and is being processed."
            ),
            responsible_party=ResponsibleParty.PATIENT,
            templates_available=["call_script.md"],
        ),
    ]

    timeline = _build_timeline(facts)

    evidence_checklist = [
        EvidenceItem(
            item="Corrected claim form (CMS-1500 or UB-04)",
            priority=EvidencePriority.REQUIRED,
            source="Provider billing department",
            purpose="Resubmission with corrections",
        ),
        EvidenceItem(
            item="Original denial letter with denial codes",
            priority=EvidencePriority.REQUIRED,
            source="Already received",
            purpose="Reference for corrections needed",
        ),
    ]

    missing_info = _identify_missing_info_requests(facts)
    assumptions = _identify_assumptions(facts, mode, plan_rules)

    return ActionPlan(
        case_id=facts.case_id,
        generated_at=datetime.now(timezone.utc),
        route=route.route,
        mode=mode,
        summary=summary,
        steps=steps,
        timeline=timeline,
        evidence_checklist=evidence_checklist,
        missing_info_requests=missing_info,
        generated_documents=[
            GeneratedDocument(
                filename="resubmit_checklist.md",
                document_type=DocumentTypeGenerated.RESUBMIT_CHECKLIST,
                description="Checklist for claim resubmission",
            ),
            GeneratedDocument(
                filename="call_script.md",
                document_type=DocumentTypeGenerated.CALL_SCRIPT,
                description="Script for calling provider and insurance",
            ),
        ],
        assumptions=assumptions,
    )


def _generate_appeal_plan(
    facts: CaseFacts,
    route: RouteDecision,
    mode: str,
    plan_rules: PlanRules | None = None,
) -> ActionPlan:
    """Generate action plan for medical necessity appeal route."""
    # Build recommendation with plan-specific citations if available
    recommendation = (
        "You should file a formal appeal with supporting clinical documentation "
        "that demonstrates the medical necessity of the service."
    )
    success_factors = [
        "Strong clinical documentation",
        "Letter of medical necessity from provider",
        "Appeal filed within deadline",
        "Policy criteria addressed directly",
    ]

    # Enhance with plan rules if available (Verified mode)
    if plan_rules:
        # Add appeal deadline from plan rules
        if plan_rules.appeal_rights and plan_rules.appeal_rights.internal_appeal_deadline_days:
            deadline_days = plan_rules.appeal_rights.internal_appeal_deadline_days
            citation = plan_rules.appeal_rights.get_citation().format_citation()
            recommendation += (
                f" Your plan allows {deadline_days} days to file an appeal. {citation}"
            )

        # Add medical necessity criteria if available
        if plan_rules.medical_necessity_criteria:
            criteria = plan_rules.medical_necessity_criteria[0]
            if criteria.definition:
                citation = criteria.get_citation().format_citation()
                success_factors.append(
                    f"Service meets plan's medical necessity definition {citation}"
                )

    summary = ActionSummary(
        situation=(
            f"Your claim was denied based on medical necessity. Reason: {facts.denial_reason}"
        ),
        recommendation=recommendation,
        success_likelihood=SuccessLikelihood.MEDIUM,
        success_factors=success_factors,
    )

    steps = [
        ActionStep(
            step_number=1,
            action="Request medical records",
            description=(
                "Contact your healthcare provider to request complete medical records "
                "related to this service, including clinical notes, test results, "
                "and treatment history."
            ),
            responsible_party=ResponsibleParty.PATIENT,
        ),
        ActionStep(
            step_number=2,
            action="Request letter of medical necessity",
            description=(
                "Ask your provider to write a letter of medical necessity explaining "
                "why this service is needed for your condition. The letter should "
                "address the specific denial reason."
            ),
            responsible_party=ResponsibleParty.CLINICIAN,
            templates_available=["clinician_letter.md"],
        ),
        ActionStep(
            step_number=3,
            action="Prepare appeal letter",
            description=(
                "Write your appeal letter using the provided template. Include your "
                "personal statement, reference the denial, and explain why the service "
                "is medically necessary for you."
            ),
            responsible_party=ResponsibleParty.PATIENT,
            templates_available=["appeal_letter.md"],
        ),
        ActionStep(
            step_number=4,
            action="Submit appeal",
            description=(
                f"Submit your appeal package to the insurance company. "
                f"{'Use the contact information in the denial letter.' if facts.contact_info else 'Find the appeals address on your insurance card or EOB.'}"
            ),
            responsible_party=ResponsibleParty.PATIENT,
            documents_needed=[
                "Appeal letter",
                "Letter of medical necessity",
                "Medical records",
                "Copy of denial letter",
            ],
            deadline=facts.dates.appeal_deadline if facts.dates else None,
        ),
        ActionStep(
            step_number=5,
            action="Follow up on appeal",
            description=(
                "After submitting, call to confirm receipt. Note the confirmation number. "
                "Appeals typically take 30-60 days to process."
            ),
            responsible_party=ResponsibleParty.PATIENT,
            templates_available=["call_script.md"],
        ),
    ]

    timeline = _build_timeline(facts)

    evidence_checklist = [
        EvidenceItem(
            item="Letter of medical necessity from provider",
            priority=EvidencePriority.REQUIRED,
            source="Healthcare provider/clinician",
            purpose="Clinical justification for the service",
        ),
        EvidenceItem(
            item="Complete medical records",
            priority=EvidencePriority.REQUIRED,
            source="Healthcare provider",
            purpose="Documentation of condition and treatment history",
        ),
        EvidenceItem(
            item="Lab results or imaging reports",
            priority=EvidencePriority.RECOMMENDED,
            source="Healthcare provider",
            purpose="Objective evidence supporting necessity",
        ),
        EvidenceItem(
            item="Prior treatment records",
            priority=EvidencePriority.RECOMMENDED,
            source="Healthcare provider",
            purpose="Shows what treatments have been tried",
        ),
        EvidenceItem(
            item="Clinical guidelines or studies",
            priority=EvidencePriority.OPTIONAL,
            source="Medical literature",
            purpose="Evidence that service is standard of care",
        ),
    ]

    missing_info = _identify_missing_info_requests(facts)
    assumptions = _identify_assumptions(facts, mode, plan_rules)

    return ActionPlan(
        case_id=facts.case_id,
        generated_at=datetime.now(timezone.utc),
        route=route.route,
        mode=mode,
        summary=summary,
        steps=steps,
        timeline=timeline,
        evidence_checklist=evidence_checklist,
        missing_info_requests=missing_info,
        generated_documents=[
            GeneratedDocument(
                filename="appeal_letter.md",
                document_type=DocumentTypeGenerated.APPEAL_LETTER,
                description="Template appeal letter to insurance",
            ),
            GeneratedDocument(
                filename="call_script.md",
                document_type=DocumentTypeGenerated.CALL_SCRIPT,
                description="Script for following up on appeal",
            ),
        ],
        assumptions=assumptions,
    )


def _build_timeline(facts: CaseFacts) -> Timeline:
    """Build timeline from case facts."""
    timeline = Timeline()

    if facts.dates:
        timeline.appeal_deadline = facts.dates.appeal_deadline

        if facts.dates.appeal_deadline:
            days = (facts.dates.appeal_deadline - date.today()).days
            timeline.days_remaining = max(days, 0)

            # Recommend submitting 5 days before deadline
            if days > 5:
                timeline.recommended_submission_date = facts.dates.appeal_deadline - timedelta(
                    days=5
                )
            else:
                timeline.recommended_submission_date = date.today()

    timeline.expected_response_time = "30-60 days for standard appeals"

    return timeline


def _identify_missing_info_requests(facts: CaseFacts) -> list[MissingInfoRequest]:
    """Identify what information would improve the plan."""
    requests: list[MissingInfoRequest] = []

    if not facts.payer or not facts.payer.name:
        requests.append(
            MissingInfoRequest(
                question="What is your insurance company name?",
                field="payer.name",
                importance=ImportanceLevel.HELPFUL,
            )
        )

    if not facts.service or not facts.service.description:
        requests.append(
            MissingInfoRequest(
                question="What service or procedure was denied?",
                field="service.description",
                importance=ImportanceLevel.HELPFUL,
            )
        )

    if not facts.dates or not facts.dates.appeal_deadline:
        requests.append(
            MissingInfoRequest(
                question="What is the deadline to file an appeal? (Check your denial letter)",
                field="dates.appeal_deadline",
                importance=ImportanceLevel.CRITICAL,
            )
        )

    return requests


def _identify_assumptions(
    facts: CaseFacts, mode: str, plan_rules: PlanRules | None = None
) -> list[Assumption]:
    """Identify assumptions made due to missing information."""
    assumptions: list[Assumption] = []

    if mode == "fast" and not plan_rules:
        assumptions.append(
            Assumption(
                assumption="Plan-specific coverage details not verified",
                impact="Recommendations are general and may not reflect your specific plan rules",
                how_to_verify="Upload your Summary of Benefits and Coverage (SBC) for verified guidance",
            )
        )
    elif plan_rules and plan_rules.extraction_quality.confidence < 0.7:
        # In verified mode with low confidence, note the extraction quality
        assumptions.append(
            Assumption(
                assumption="Plan rules extracted with moderate confidence",
                impact="Some plan details may need verification",
                how_to_verify="Review your SBC document directly for exact policy language",
            )
        )

    if not facts.dates or not facts.dates.appeal_deadline:
        assumptions.append(
            Assumption(
                assumption="Standard 180-day appeal deadline assumed",
                impact="Your actual deadline may be shorter",
                how_to_verify="Check your denial letter or call your insurance for exact deadline",
            )
        )

    if not facts.payer or not facts.payer.plan_type:
        assumptions.append(
            Assumption(
                assumption="Standard commercial insurance plan assumed",
                impact="Process may differ for Medicare, Medicaid, or self-funded plans",
                how_to_verify="Check your insurance card for plan type",
            )
        )

    return assumptions
